"""
=============================================================================
37_api_auth_oauth2_and_jwt.py: OAuth2 Client Credentials & JWT from Scratch
=============================================================================
Senior GenAI Engineer Context:
Module 36's agent service has zero auth — anyone who can reach it can call
`/chat` and spend your LLM budget. This module implements, from the standard
library only, the two mechanisms that actually protect a production agent
API:

1. JWT (JSON Web Token): base64url-encoded `header.payload.signature`, HMAC-
   SHA256 signed, with expiry (`exp`) and scope claims — implemented and
   verified by hand so the "it's just a signed, expiring claim" mental model
   is concrete, not magic.
2. OAuth2 Client-Credentials Grant: the machine-to-machine flow (no human
   login) — a client_id/client_secret pair is exchanged at a `/token`
   endpoint for a short-lived JWT access token, which is then presented as
   `Authorization: Bearer <token>` on every subsequent call.

The auth guard below plugs directly into module 36's `MiniAPI` as a nested
`Depends(...)` dependency, protecting a new `/secure-chat` route — the real
FastAPI pattern of `Depends(get_current_user)` guarding an endpoint.

Run directly:
  python3 37_api_auth_oauth2_and_jwt.py
=============================================================================
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

fastapi_module = import_module("36_fastapi_service_layer_and_dependency_injection")
app = fastapi_module.app
Depends = fastapi_module.Depends
HTTPException = fastapi_module.HTTPException
ChatRequest = fastapi_module.ChatRequest
get_llm_client = fastapi_module.get_llm_client

JWT_SECRET = "dev-only-signing-secret-do-not-use-in-prod"  # real: load from a secrets manager (see module 35)


# =====================================================================
# 1. JWT: encode/sign/verify, from scratch
# =====================================================================
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def issue_jwt(claims: Dict[str, Any], secret: str, ttl_seconds: int = 300) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {**claims, "iat": int(time.time()), "exp": int(time.time()) + ttl_seconds}

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()

    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_jwt(token: str, secret: str) -> Dict[str, Any]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError:
        raise ValueError("Malformed token (expected 3 dot-separated segments)")

    expected_signature = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
    actual_signature = _b64url_decode(signature_b64)
    # Constant-time comparison — a naive `==` leaks timing information an
    # attacker can use to forge a valid signature byte-by-byte.
    if not hmac.compare_digest(expected_signature, actual_signature):
        raise ValueError("Invalid signature")

    payload = json.loads(_b64url_decode(payload_b64))
    if payload.get("exp", 0) < time.time():
        raise ValueError("Token has expired")
    return payload


# =====================================================================
# 2. OAUTH2 CLIENT-CREDENTIALS GRANT (machine-to-machine auth)
# =====================================================================
class OAuth2TokenServer:
    """Real equivalent: Auth0 / Keycloak / Cognito's `/oauth/token` endpoint."""

    def __init__(self):
        # client_id -> (client_secret, granted_scopes)
        self.registered_clients: Dict[str, Dict[str, Any]] = {
            "sre-dashboard": {"secret": "sre-dashboard-secret", "scopes": ["chat:read", "chat:write"]},
            "readonly-bot": {"secret": "readonly-bot-secret", "scopes": ["chat:read"]},
        }

    def issue_token(self, client_id: str, client_secret: str) -> Dict[str, Any]:
        client = self.registered_clients.get(client_id)
        if client is None or not hmac.compare_digest(client["secret"], client_secret):
            raise ValueError("Invalid client credentials")

        access_token = issue_jwt(
            {"sub": client_id, "scopes": client["scopes"], "token_type": "access_token"},
            JWT_SECRET,
            ttl_seconds=300,
        )
        return {"access_token": access_token, "token_type": "Bearer", "expires_in": 300}


token_server = OAuth2TokenServer()


# =====================================================================
# 3. AUTH GUARD wired into module 36's MiniAPI via nested Depends
# =====================================================================
def get_request_headers() -> Dict[str, str]:
    return app.current_headers


def require_scope(required_scope: str):
    """Returns a Depends-compatible guard closed over the required scope —
    the same "parameterized dependency" pattern real FastAPI auth uses.
    """
    def guard(headers: Dict[str, str] = Depends(get_request_headers)) -> Dict[str, Any]:
        auth_header = (headers or {}).get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")

        token = auth_header[len("Bearer "):]
        try:
            claims = verify_jwt(token, JWT_SECRET)
        except ValueError as e:
            raise HTTPException(401, f"Invalid token: {e}")

        if required_scope not in claims.get("scopes", []):
            raise HTTPException(403, f"Token missing required scope '{required_scope}'")
        return claims
    return guard


@app.post("/secure-chat", body_model=ChatRequest)
def secure_chat_endpoint(
    body: ChatRequest,
    claims: Dict[str, Any] = Depends(require_scope("chat:write")),
    llm: LLMClient = Depends(get_llm_client),
):
    reply = llm.chat(
        [{"role": "user", "content": body.message}],
        simulated_response=f"[authenticated as {claims['sub']}] Acknowledged: '{body.message}'.",
    )
    return {"authenticated_client": claims["sub"], "scopes": claims["scopes"], "reply": reply}


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== OAuth2 Client Credentials & JWT ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Client credentials grant — exchange client_id/secret for a JWT{Colors.RESET}")
    token_response = token_server.issue_token("sre-dashboard", "sre-dashboard-secret")
    access_token = token_response["access_token"]
    print(f"  {token_response}\n")

    print(f"{Colors.YELLOW}2. Inspecting the JWT's own claims (decoded, not just opaque){Colors.RESET}")
    decoded = verify_jwt(access_token, JWT_SECRET)
    print(f"  {decoded}\n")

    print(f"{Colors.YELLOW}3. Calling the auth-protected /secure-chat route WITH a valid bearer token{Colors.RESET}")
    resp = app.request(
        "POST", "/secure-chat",
        json_body={"message": "What's the deploy status?", "agent_id": "n/a"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}4. Calling it WITHOUT a token -> 401{Colors.RESET}")
    resp = app.request("POST", "/secure-chat", json_body={"message": "hi"})
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}5. Calling it with a read-only client's token -> 403 (wrong scope){Colors.RESET}")
    readonly_token = token_server.issue_token("readonly-bot", "readonly-bot-secret")["access_token"]
    resp = app.request(
        "POST", "/secure-chat", json_body={"message": "hi"},
        headers={"Authorization": f"Bearer {readonly_token}"},
    )
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}6. Calling it with an EXPIRED token -> 401{Colors.RESET}")
    expired_token = issue_jwt({"sub": "sre-dashboard", "scopes": ["chat:write"]}, JWT_SECRET, ttl_seconds=-10)
    resp = app.request(
        "POST", "/secure-chat", json_body={"message": "hi"},
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.GREEN}✔ OAuth2 token issuance, JWT verification, and scope-based access control all verified.{Colors.RESET}")

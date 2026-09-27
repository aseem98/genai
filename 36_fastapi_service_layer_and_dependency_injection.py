"""
=============================================================================
36_fastapi_service_layer_and_dependency_injection.py: Serving Agents with FastAPI
=============================================================================
Senior GenAI Engineer Context:
Every production agent eventually needs an HTTP service layer so other
systems (frontends, other microservices, orchestrators) can call it. FastAPI
is the de-facto standard for this in Python because it gives you, for free:
  1. Declarative routing (`@app.get`, `@app.post`) with path/query params.
  2. Request/response validation via Pydantic-style type-annotated models.
  3. Dependency Injection (`Depends`) for shared resources (DB pools, LLM
     clients, auth) without global state or manual wiring.
  4. Automatic OpenAPI (Swagger) schema generation for every route.

This module re-implements those four ideas from scratch in the standard
library (`MiniAPI`) so the mechanics are 100% transparent and the script
runs with zero pip installs. The real FastAPI equivalent for every piece is
shown in the docstrings/comments alongside it.

Run directly:
  python3 36_fastapi_service_layer_and_dependency_injection.py
=============================================================================
"""

import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


# =====================================================================
# 1. REQUEST/RESPONSE MODELS (Pydantic-style validation via dataclasses)
# =====================================================================
@dataclass
class ChatRequest:
    """Equivalent of a Pydantic BaseModel used as a FastAPI request body."""
    message: str
    agent_id: str = "default"

    def __post_init__(self):
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("'message' must be a non-empty string")


class HTTPException(Exception):
    """Mirrors fastapi.HTTPException(status_code, detail)."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


# =====================================================================
# 2. DEPENDENCY INJECTION CONTAINER (mirrors FastAPI's `Depends`)
# =====================================================================
class Depends:
    """Wraps a provider callable. MiniAPI resolves it lazily per-request,
    exactly like `def endpoint(llm: LLMClient = Depends(get_llm_client))`.
    """
    def __init__(self, provider: Callable[[], Any]):
        self.provider = provider


def get_llm_client() -> LLMClient:
    """A dependency provider — swap this for a pooled/cached client in prod."""
    return LLMClient()


_AGENT_REGISTRY_SINGLETON = {"total_requests_served": 0}


def get_agent_registry() -> Dict[str, Any]:
    """Simulates a shared, app-lifetime resource (e.g. a connection pool)."""
    return _AGENT_REGISTRY_SINGLETON


# =====================================================================
# 3. MINI-FASTAPI: routing, path params, DI resolution, OpenAPI export
# =====================================================================
class Route:
    def __init__(self, method: str, path: str, handler: Callable, body_model=None):
        self.method = method
        self.path_parts = [p for p in path.split("/") if p != ""]
        self.raw_path = path
        self.handler = handler
        self.body_model = body_model

    def match(self, method: str, path: str) -> Optional[Dict[str, str]]:
        if method != self.method:
            return None
        parts = [p for p in path.split("/") if p != ""]
        if len(parts) != len(self.path_parts):
            return None
        params = {}
        for expected, actual in zip(self.path_parts, parts):
            if expected.startswith("{") and expected.endswith("}"):
                params[expected[1:-1]] = actual
            elif expected != actual:
                return None
        return params


class MiniAPI:
    """A dependency-injecting, path-matching micro web framework.
    Real equivalent: `app = FastAPI()`.
    """

    def __init__(self, title: str = "Agent Service"):
        self.title = title
        self.routes: List[Route] = []
        # Stashed per-call so a `Depends` provider can read the inbound
        # headers (see module 37's auth guard). Not thread-safe — fine for
        # this single-threaded teaching demo; a real ASGI app scopes this
        # per-request instead of on the app instance.
        self.current_headers: Dict[str, str] = {}

    def get(self, path: str):
        return self._register("GET", path)

    def post(self, path: str, body_model=None):
        return self._register("POST", path, body_model)

    def _register(self, method: str, path: str, body_model=None):
        def decorator(func: Callable):
            self.routes.append(Route(method, path, func, body_model))
            return func
        return decorator

    def _resolve_provider(self, provider: Callable) -> Any:
        """Recursively resolves a `Depends` provider, so a provider can
        itself declare `Depends(...)` defaults — FastAPI's "sub-dependency"
        chains (e.g. an auth guard that depends on a raw-headers provider).
        """
        defaults = provider.__defaults__ or ()
        varnames = provider.__code__.co_varnames[:provider.__code__.co_argcount]
        default_map = dict(zip(varnames[len(varnames) - len(defaults):], defaults))
        kwargs = {
            name: self._resolve_provider(default_map[name].provider)
            for name in varnames if isinstance(default_map.get(name), Depends)
        }
        return provider(**kwargs)

    def _resolve_kwargs(self, handler: Callable, path_params: Dict[str, str], body: Optional[dict]) -> Dict[str, Any]:
        """Inspects the handler's default values to resolve `Depends(...)`
        providers and path params — a tiny version of FastAPI's signature
        introspection.
        """
        kwargs = {}
        defaults = handler.__defaults__ or ()
        varnames = handler.__code__.co_varnames[:handler.__code__.co_argcount]
        default_map = dict(zip(varnames[len(varnames) - len(defaults):], defaults))
        for name in varnames:
            if name in path_params:
                kwargs[name] = path_params[name]
            elif isinstance(default_map.get(name), Depends):
                kwargs[name] = self._resolve_provider(default_map[name].provider)
            elif name == "body" and body is not None:
                kwargs[name] = body
        return kwargs

    def request(self, method: str, path: str, json_body: Optional[dict] = None, headers: Optional[Dict[str, str]] = None) -> "Response":
        """Simulates an incoming HTTP request without a real socket/ASGI
        server — enough to prove out routing + validation + DI end to end.
        """
        self.current_headers = headers or {}
        for route in self.routes:
            params = route.match(method, path)
            if params is None:
                continue
            try:
                body_obj = None
                if route.body_model is not None:
                    if json_body is None:
                        raise HTTPException(422, "Request body required")
                    body_obj = route.body_model(**json_body)
                kwargs = self._resolve_kwargs(route.handler, params, body_obj)
                result = route.handler(**kwargs)
                return Response(200, result)
            except HTTPException as e:
                return Response(e.status_code, {"detail": e.detail})
            except (TypeError, ValueError) as e:
                return Response(422, {"detail": f"Validation error: {e}"})
        return Response(404, {"detail": f"No route for {method} {path}"})

    def openapi_schema(self) -> dict:
        """Auto-generated API contract — real equivalent: GET /openapi.json."""
        paths = {}
        for route in self.routes:
            paths.setdefault(route.raw_path, {})[route.method.lower()] = {
                "operationId": route.handler.__name__,
                "requestBody": (
                    {"schema": [f.name for f in fields(route.body_model)]}
                    if route.body_model and is_dataclass(route.body_model)
                    else None
                ),
            }
        return {"title": self.title, "openapi": "3.1.0", "paths": paths}


class Response:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body

    def json(self) -> str:
        return json.dumps(self.body, default=str, indent=2)


# =====================================================================
# 4. AGENT SERVICE — routes wired with dependency injection
# =====================================================================
app = MiniAPI(title="Agentic Chat Service")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/agents/{agent_id}/status")
def agent_status(agent_id: str, registry: dict = Depends(get_agent_registry)):
    return {"agent_id": agent_id, "requests_served": registry["total_requests_served"]}


@app.post("/chat", body_model=ChatRequest)
def chat_endpoint(
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
    registry: dict = Depends(get_agent_registry),
):
    registry["total_requests_served"] += 1
    reply = llm.chat(
        [{"role": "user", "content": body.message}],
        simulated_response=f"[{body.agent_id}] Acknowledged: '{body.message}'. How else can I help?",
    )
    return {"agent_id": body.agent_id, "reply": reply}


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== FastAPI-style Agent Service Layer ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. GET /health{Colors.RESET}")
    resp = app.request("GET", "/health")
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}2. POST /chat (valid body -> DI-injected LLMClient){Colors.RESET}")
    resp = app.request("POST", "/chat", {"message": "What's the incident status?", "agent_id": "sre-bot"})
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}3. POST /chat (invalid body -> 422 validation error){Colors.RESET}")
    resp = app.request("POST", "/chat", {"message": "   "})
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}4. GET /agents/{{agent_id}}/status (path param + shared DI singleton){Colors.RESET}")
    resp = app.request("GET", "/agents/sre-bot/status")
    print(f"  {resp.status_code} -> {resp.json()}\n")

    print(f"{Colors.YELLOW}5. Auto-generated OpenAPI schema{Colors.RESET}")
    print(f"  {json.dumps(app.openapi_schema(), indent=2)}\n")

    print(f"{Colors.GREEN}✔ Service layer, DI container, and OpenAPI export are working.{Colors.RESET}")
    print(f"{Colors.DIM}Real deployment: `uvicorn service:app --host 0.0.0.0 --port 8000`{Colors.RESET}")

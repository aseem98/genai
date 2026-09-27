"""
=============================================================================
01_unified_llm_client.py: Your First LLM Call — the Client Every Module Uses
=============================================================================
Start here. Every later module talks to a language model through this one
client, so it's the first thing to understand:

1. CONFIGURATION: a hand-rolled `.env` loader (stdlib only, no python-dotenv)
   reads OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL. Any OpenAI-compatible
   endpoint works: OpenAI, Groq, DeepSeek, Ollama, or a local gateway like
   OmniRoute.
2. THE CHAT COMPLETIONS PROTOCOL: a POST to `/chat/completions` with a list
   of role-tagged messages; the reply carries the generated text (or
   tool_calls) plus a `usage` block with real prompt/completion token counts.
3. RESILIENCE: retries with exponential backoff on transient failures (429,
   5xx, timeouts) — and fail-fast on everything else (401, 400).
4. STREAMING: Server-Sent Events, yielding tokens as they're generated.
5. EMBEDDINGS: `embed()` calls a real `/embeddings` endpoint when
   EMBEDDING_MODEL is configured; otherwise it falls back to feature hashing
   (the "hashing trick" — a real, production lexical embedding, the same idea
   as scikit-learn's HashingVectorizer). Feature hashing captures shared
   WORDS, not meaning: "refund" and "money back" won't match. Module 05
   explains the difference in depth.
6. OFFLINE MODE: with no API key, calls return the scripted
   `simulated_response` each module provides, so every module still runs.

Run directly:
  python3 01_unified_llm_client.py
=============================================================================
"""

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import zlib
from typing import Any, Dict, Iterator, List, Optional


# =====================================================================
# 0. MINIMAL .ENV LOADER (stdlib only — no `python-dotenv` dependency)
# =====================================================================
def load_dotenv(dotenv_path: Optional[str] = None) -> None:
    """Parses `KEY=VALUE` lines (with `#` comments) into `os.environ`. A value
    already exported in the shell always wins over the `.env` file.
    """
    path = dotenv_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


load_dotenv()


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    RESET = '\033[0m'


# Transient failures worth retrying; anything else (401, 400, 404) fails fast.
RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_STREAM_ERROR_TYPES = {"server_error", "service_unavailable", "rate_limit_error", "overloaded_error", "timeout"}


class StreamError(RuntimeError):
    """An error delivered INSIDE an HTTP-200 SSE stream (`data: {"error": ...}`)
    — easy to miss, since the HTTP status already said success."""

    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable

HASHING_EMBEDDING_DIM = 64


def hashing_embedding(text: str, dim: int = HASHING_EMBEDDING_DIM) -> List[float]:
    """Feature hashing: each lowercase word is hashed (crc32 — stable across
    processes, unlike Python's randomized `hash()`) into one of `dim`
    buckets, then the vector is L2-normalized. Documents sharing words get
    similar vectors; synonyms with no shared words do not.
    """
    vec = [0.0] * dim
    for word in re.findall(r"\b\w+\b", text.lower()):
        vec[zlib.crc32(word.encode("utf-8")) % dim] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


class LLMClient:
    """One client for chat, streaming, and embeddings — live or offline."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 5,
        timeout_s: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.last_usage: Optional[Dict[str, int]] = None

        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", self.base_url)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.api_key)

        if provider == "live" or (provider is None and self.api_key):
            self.mode = "live"
        else:
            self.mode = "simulation"

    @property
    def embedding_mode(self) -> str:
        return "api" if (self.mode == "live" and self.embedding_model) else "hashing"

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        simulated_response: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Returns the reply text, or a JSON `{"tool_calls": [...]}` string
        when the model decided to call a tool. `stop` sequences end
        generation early; the text is also cut client-side at the first stop
        sequence, because not every provider honors the parameter."""
        if self.mode == "live":
            text = self._call_live_api(messages, tools, stream=False, temperature=temperature, max_tokens=max_tokens, stop=stop)
        else:
            text = self._simulate_response(messages, tools, simulated_response)
        for s in stop or []:
            if s in text:
                text = text[:text.index(s)]
        return text

    def stream_chat(self, messages: List[Dict[str, Any]], simulated_response: Optional[str] = None) -> Iterator[str]:
        """Yields the reply token-by-token (a real SSE stream when live).
        A transient in-stream error is retried only if NO token was yielded
        yet — once text has reached the caller, a silent retry would repeat it."""
        if self.mode == "live":
            for attempt in range(1, self.max_retries + 1):
                yielded = False
                try:
                    for token in self._call_live_api(messages, tools=None, stream=True):
                        yielded = True
                        yield token
                    return
                except StreamError as e:
                    if yielded or not e.retryable or attempt == self.max_retries:
                        raise
                    time.sleep(self._backoff_seconds(attempt))
            return
        text = simulated_response or self._simulate_response(messages, None, None)
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Real embeddings from `/embeddings` when EMBEDDING_MODEL is set and
        the client is live; feature-hashing embeddings otherwise."""
        if self.embedding_mode == "hashing":
            return [hashing_embedding(t) for t in texts]
        payload = {"model": self.embedding_model, "input": texts}
        result = self._post_json(self.embedding_base_url, "/embeddings", payload, self.embedding_api_key)
        return [item["embedding"] for item in sorted(result["data"], key=lambda d: d["index"])]

    # -----------------------------------------------------------------
    # LIVE API TRANSPORT (with retry/backoff)
    # -----------------------------------------------------------------
    def _post_json(self, base_url: str, path: str, payload: Dict[str, Any], api_key: Optional[str]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}" if api_key else ""},
            method="POST",
        )
        with self._open_with_retries(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _backoff_seconds(attempt: int, retry_after: Optional[str] = None) -> float:
        """Honor the server's Retry-After header if it sent one; otherwise
        exponential backoff (1s, 2s, 4s... capped at 16s) with random jitter,
        so many clients retrying at once don't stampede the server in sync."""
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        return min(2.0 ** (attempt - 1), 16.0) * random.uniform(0.75, 1.25)

    def _open_with_retries(self, req: urllib.request.Request):
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            retry_after = None
            try:
                return urllib.request.urlopen(req, timeout=self.timeout_s)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Live LLM API HTTP {e.code} Error: {body}")
                if e.code not in RETRYABLE_HTTP_STATUSES or attempt == self.max_retries:
                    raise last_error
                retry_after = e.headers.get("Retry-After") if e.headers else None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = RuntimeError(f"Failed to communicate with LLM API: {e}")
                if attempt == self.max_retries:
                    raise last_error
            time.sleep(self._backoff_seconds(attempt, retry_after))
        raise last_error or RuntimeError("Live LLM API call failed.")

    def _call_live_api(self, messages, tools, stream: bool, temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None, stop: Optional[List[str]] = None):
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        if stream:
            payload["stream_options"] = {"include_usage": True}  # final SSE chunk carries token usage

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}" if self.api_key else ""},
            method="POST",
        )
        resp = self._open_with_retries(req)
        if stream:
            return self._consume_sse_stream(resp)
        with resp:
            result = json.loads(resp.read().decode("utf-8"))
        self.last_usage = result.get("usage")
        choice = result["choices"][0]["message"]
        if choice.get("tool_calls"):
            return json.dumps({"tool_calls": choice["tool_calls"]})
        return choice.get("content", "") or ""

    def _consume_sse_stream(self, resp) -> Iterator[str]:
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    err = chunk["error"] if isinstance(chunk["error"], dict) else {"message": str(chunk["error"])}
                    retryable = bool({err.get("type"), err.get("code")} & RETRYABLE_STREAM_ERROR_TYPES)
                    raise StreamError(f"Stream error: {err.get('message')}", retryable=retryable)
                if chunk.get("usage"):
                    self.last_usage = chunk["usage"]
                choices = chunk.get("choices") or [{}]
                token = choices[0].get("delta", {}).get("content")
                if token:
                    yield token

    # -----------------------------------------------------------------
    # OFFLINE SIMULATION
    # -----------------------------------------------------------------
    def _simulate_response(self, messages, tools=None, simulated_response: Optional[str] = None) -> str:
        if simulated_response:
            return simulated_response
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, list):  # multimodal content parts
            last = " ".join(p.get("text", "") for p in last if p.get("type") == "text")
        if tools:
            first_tool = tools[0].get("function", tools[0])
            return json.dumps({"tool_calls": [{
                "id": "call_sim_0", "type": "function",
                "function": {"name": first_tool.get("name", "unknown_tool"), "arguments": "{}"},
            }]})
        return f"[Simulated Response to: {last[:50]}...]"


_default_client: Optional[LLMClient] = None
_embedding_cache: Dict[str, List[float]] = {}


def embed_text(text: str) -> List[float]:
    """Module-level convenience used across the curriculum: one shared
    client, with an in-process cache so a demo never pays for the same
    embedding twice."""
    global _default_client
    if text not in _embedding_cache:
        if _default_client is None:
            _default_client = LLMClient()
        _embedding_cache[text] = _default_client.embed([text])[0]
    return _embedding_cache[text]


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== 01: Your First LLM Call ==={Colors.RESET}\n")
    client = LLMClient()
    print(f"Chat mode:      {Colors.GREEN}{client.mode.upper()}{Colors.RESET}")
    print(f"Embedding mode: {Colors.GREEN}{client.embedding_mode.upper()}{Colors.RESET}"
          f"{'' if client.embedding_mode == 'api' else '  (set EMBEDDING_MODEL in .env to use a real embeddings API)'}")
    if client.mode == "live":
        print(f"Endpoint:       {client.base_url}  |  model: {client.model}")
    print()

    print(f"{Colors.YELLOW}1. A single chat completion (and the real token usage it reports){Colors.RESET}")
    reply = client.chat(
        [{"role": "system", "content": "You are a concise technical assistant."},
         {"role": "user", "content": "In one sentence, what is a large language model?"}],
        simulated_response="A large language model is a neural network trained on vast text to predict the next token, which lets it generate and reason over language.",
    )
    print(f"  Reply: {reply}")
    print(f"  Usage: {client.last_usage or '(not available offline)'}\n")

    print(f"{Colors.YELLOW}2. Multi-turn conversation — the model only 'remembers' what you resend{Colors.RESET}")
    history = [
        {"role": "user", "content": "My favorite database is PostgreSQL. Remember that."},
        {"role": "assistant", "content": "Noted — PostgreSQL."},
        {"role": "user", "content": "Which database did I say was my favorite? One word."},
    ]
    print(f"  Reply: {client.chat(history, simulated_response='PostgreSQL')}\n")

    print(f"{Colors.YELLOW}3. Streaming — tokens arrive incrementally over Server-Sent Events{Colors.RESET}")
    print("  ", end="")
    chunk_count = 0
    for token in client.stream_chat(
        [{"role": "user", "content": "Count from 1 to 5, separated by commas."}],
        simulated_response="1, 2, 3, 4, 5",
    ):
        print(token, end="", flush=True)
        chunk_count += 1
    print(f"\n  ({chunk_count} stream chunks received)\n")

    print(f"{Colors.YELLOW}4. Embeddings — text in, fixed-length vector out{Colors.RESET}")
    vectors = client.embed(["reset my password", "password reset help", "quarterly revenue report"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    print(f"  Dimension: {len(vectors[0])}")
    print(f"  sim('reset my password', 'password reset help')      = {dot(vectors[0], vectors[1]):.3f}")
    print(f"  sim('reset my password', 'quarterly revenue report') = {dot(vectors[0], vectors[2]):.3f}\n")

    print(f"{Colors.YELLOW}5. Error handling — a non-retryable 4xx fails fast with the provider's real error{Colors.RESET}")
    if client.mode == "live":
        # A model id whose provider isn't configured returns a real 4xx.
        # 4xx isn't in RETRYABLE_HTTP_STATUSES, so there's exactly one attempt.
        bad_client = LLMClient(model="unconfigured-provider/some-model")
        t0 = time.time()
        try:
            bad_client.chat([{"role": "user", "content": "hi"}])
            print("  (this endpoint accepted the request — some gateways fall back to a default model)")
        except RuntimeError as e:
            print(f"  {Colors.RED}Caught after {time.time() - t0:.2f}s (no retries): {str(e)[:110]}{Colors.RESET}")
    else:
        print("  (skipped offline — needs a reachable endpoint to demonstrate a real auth failure)")

    print(f"\n{Colors.GREEN}✔ Client ready — every later module builds on this.{Colors.RESET}")

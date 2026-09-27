"""
=============================================================================
42_redis_caching_and_agent_state.py: Redis for LLM Caching & Agent Session State
=============================================================================
Senior GenAI Engineer Context:
Redis shows up twice in almost every production agent stack:
  1. CACHE-ASIDE for LLM calls: identical (or near-identical) prompts are
     extremely common (e.g. "summarize this ticket" across a support queue).
     Hashing the prompt and caching the response with a TTL can cut LLM
     spend and P99 latency dramatically.
  2. SESSION / SHORT-TERM MEMORY: an agent's working memory for a given
     conversation needs to live somewhere shared and fast so any replica of
     a horizontally-scaled service can pick up the conversation.

This module implements an `InMemoryRedis` client with the same core surface
as `redis-py` (`set`, `get`, `setex`, `ttl`, `delete`, `exists`, `incr`) so
every call site below is a drop-in match for the real thing. If the `redis`
package is installed AND `REDIS_URL` is set, it transparently uses a real
Redis server instead — same call sites, same code.

Real equivalent:
  import redis
  r = redis.Redis.from_url(os.environ["REDIS_URL"])
  r.setex(key, ttl_seconds, value)

Run directly:
  python3 42_redis_caching_and_agent_state.py
"""

import os
import time
import json
import hashlib
from typing import Any, Callable, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


# =====================================================================
# 1. IN-MEMORY REDIS-COMPATIBLE CLIENT (zero external dependency)
# =====================================================================
class InMemoryRedis:
    """Mirrors the subset of the redis-py API used by agent services."""

    def __init__(self):
        self._store: dict = {}
        self._expiry: dict = {}

    def _is_expired(self, key: str) -> bool:
        expires_at = self._expiry.get(key)
        return expires_at is not None and time.time() >= expires_at

    def _purge_if_expired(self, key: str):
        if key in self._store and self._is_expired(key):
            del self._store[key]
            del self._expiry[key]

    def set(self, key: str, value: str):
        self._store[key] = value
        self._expiry.pop(key, None)

    def setex(self, key: str, ttl_seconds: float, value: str):
        self._store[key] = value
        self._expiry[key] = time.time() + ttl_seconds

    def get(self, key: str) -> Optional[str]:
        self._purge_if_expired(key)
        return self._store.get(key)

    def ttl(self, key: str) -> float:
        self._purge_if_expired(key)
        if key not in self._store:
            return -2  # redis convention: key does not exist
        if key not in self._expiry:
            return -1  # redis convention: no expiry set
        return max(0.0, self._expiry[key] - time.time())

    def exists(self, key: str) -> bool:
        self._purge_if_expired(key)
        return key in self._store

    def delete(self, key: str):
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    def incr(self, key: str) -> int:
        current = int(self.get(key) or 0)
        current += 1
        self.set(key, str(current))
        return current


def get_redis_client():
    """Live-mode switch: real redis-py if configured, else the in-memory stub."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis  # type: ignore
            return redis.Redis.from_url(redis_url, decode_responses=True), "live"
        except ImportError:
            pass
    return InMemoryRedis(), "simulation"


# =====================================================================
# 2. CACHE-ASIDE PATTERN FOR LLM RESPONSES
# =====================================================================
def prompt_cache_key(prompt: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"llm_cache:{model}:{digest}"


def cached_llm_call(
    cache: InMemoryRedis,
    llm: LLMClient,
    prompt: str,
    simulated_response: str,
    ttl_seconds: float = 300,
) -> Any:
    """Cache-aside: check cache -> on miss call LLM -> populate cache -> return."""
    key = prompt_cache_key(prompt, llm.model)
    hit = cache.get(key)
    if hit is not None:
        return json.loads(hit)["response"], "CACHE HIT"

    response = llm.chat([{"role": "user", "content": prompt}], simulated_response=simulated_response)
    cache.setex(key, ttl_seconds, json.dumps({"response": response}))
    return response, "CACHE MISS"


# =====================================================================
# 3. AGENT SESSION STATE (short-term working memory, TTL-bound)
# =====================================================================
class RedisAgentSession:
    """Conversation state keyed per session_id, backed by Redis so any
    stateless service replica can resume the same conversation.
    """

    def __init__(self, cache: InMemoryRedis, session_id: str, ttl_seconds: float = 1800):
        self.cache = cache
        self.key = f"session:{session_id}:history"
        self.ttl_seconds = ttl_seconds

    def append(self, role: str, content: str):
        history = self.get_history()
        history.append({"role": role, "content": content})
        self.cache.setex(self.key, self.ttl_seconds, json.dumps(history))

    def get_history(self) -> list:
        raw = self.cache.get(self.key)
        return json.loads(raw) if raw else []


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== Redis: LLM Caching & Agent Session State ==={Colors.RESET}\n")

    cache, mode = get_redis_client()
    print(f"Active Redis Mode: {Colors.GREEN}{mode.upper()}{Colors.RESET}\n")

    llm = LLMClient()
    prompt = "Summarize: 'Payment service returned 500s for 3 minutes at 02:14 UTC.'"
    canned = "Brief P1 payment outage (~3 min) at 02:14 UTC; auto-recovered."

    print(f"{Colors.YELLOW}1. Cache-aside pattern for repeated LLM prompts{Colors.RESET}")
    resp1, status1 = cached_llm_call(cache, llm, prompt, canned)
    print(f"  Call #1 -> {status1}: {resp1}")
    resp2, status2 = cached_llm_call(cache, llm, prompt, canned)
    print(f"  Call #2 -> {status2}: {resp2}")
    key = prompt_cache_key(prompt, llm.model)
    print(f"  Key: {key} | TTL remaining: {cache.ttl(key):.0f}s\n")

    print(f"{Colors.YELLOW}2. Agent session state shared via Redis{Colors.RESET}")
    session = RedisAgentSession(cache, session_id="user-42", ttl_seconds=1800)
    session.append("user", "What's the status of my refund?")
    session.append("assistant", "Your refund of $49.00 was issued and should post in 2-3 days.")
    for turn in session.get_history():
        print(f"  [{turn['role']:<9}] {turn['content']}")

    print(f"\n{Colors.YELLOW}3. Rate-limit counter via INCR (sliding-window building block){Colors.RESET}")
    for _ in range(3):
        count = cache.incr("rate_limit:user-42:minute")
    print(f"  Requests this window: {count}")

    print(f"\n{Colors.GREEN}✔ Cache-aside and session-state patterns verified.{Colors.RESET}")
    print(f"{Colors.DIM}Live mode: export REDIS_URL=\"redis://localhost:6379/0\" (requires `pip install redis`){Colors.RESET}")

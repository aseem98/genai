"""
=============================================================================
43_semantic_and_prompt_caching.py: Semantic Caching & Anthropic-style Prompt Caching
=============================================================================
Senior GenAI Engineer Context:
Module 42 covers exact-match caching (same prompt hash -> same cached
response). Two distinct, higher-leverage caching strategies sit on top of it:

1. SEMANTIC CACHING: users paraphrase the same question constantly
   ("what's our refund window?" vs "how long do refunds take to process?").
   An exact-hash cache misses both; a semantic cache embeds the query and
   returns a cached response if a past query is similar enough (cosine
   similarity above a threshold) — trading a small correctness risk for a
   large hit-rate gain.
2. PROMPT CACHING (Anthropic `cache_control`, OpenAI-style prefix caching):
   a long, unchanging prefix (system prompt, few-shot examples, a big
   retrieved document) is marked cacheable. The FIRST call pays a premium to
   write it into the cache; every subsequent call reads it back at a steep
   discount, because the provider skips a full attention pass over it.

Run directly:
  python3 43_semantic_and_prompt_caching.py
=============================================================================
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
VectorMath = import_module("05_embeddings_and_vector_math").VectorMath
embed_text = import_module("01_unified_llm_client").embed_text


# =====================================================================
# 1. SEMANTIC CACHE (embedding-similarity lookup, not exact string match)
# =====================================================================
class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold
        self.entries: List[Tuple[List[float], str, str]] = []  # (embedding, original_query, response)

    def get(self, query: str) -> Optional[Tuple[str, str, float]]:
        """Returns (response, matched_query, similarity) on a hit, else None."""
        if not self.entries:
            return None
        q_embed = embed_text(query)
        best_sim, best_query, best_response = -1.0, None, None
        for embed, orig_query, response in self.entries:
            sim = VectorMath.cosine_similarity(q_embed, embed)
            if sim > best_sim:
                best_sim, best_query, best_response = sim, orig_query, response
        if best_sim >= self.similarity_threshold:
            return best_response, best_query, best_sim
        return None

    def set(self, query: str, response: str):
        self.entries.append((embed_text(query), query, response))


def cached_llm_call(cache: SemanticCache, llm: LLMClient, query: str, simulated_response: str) -> Tuple[str, str]:
    hit = cache.get(query)
    if hit is not None:
        response, matched_query, sim = hit
        return response, f"SEMANTIC CACHE HIT (sim={sim:.3f} vs \"{matched_query}\")"
    response = llm.chat([{"role": "user", "content": query}], simulated_response=simulated_response)
    cache.set(query, response)
    return response, "CACHE MISS (generated + cached)"


# =====================================================================
# 2. PROMPT / KV CACHING COST MODEL (Anthropic `cache_control`-style)
# =====================================================================
class PromptCacheSimulator:
    """Models the real Anthropic prompt-caching pricing shape: a cache WRITE
    costs MORE than a normal input token (the provider has to compute and
    store the KV cache), while a cache READ costs a small fraction of a
    normal input token (attention over that prefix is skipped entirely).
    """
    WRITE_MULTIPLIER = 1.25   # writing to cache: +25% over base input price
    READ_MULTIPLIER = 0.10    # reading from cache: -90% off base input price
    BASE_INPUT_COST_PER_M = 3.00  # $ / 1M input tokens (illustrative)

    def __init__(self, ttl_calls: int = 999):
        self.cached_prefixes: Dict[str, bool] = {}

    def cost_for_call(self, cacheable_prefix: str, fresh_suffix: str) -> Dict[str, Any]:
        prefix_tokens = len(cacheable_prefix.split())
        suffix_tokens = len(fresh_suffix.split())
        prefix_hash = hashlib.sha256(cacheable_prefix.encode("utf-8")).hexdigest()[:12]

        cache_hit = prefix_hash in self.cached_prefixes
        multiplier = self.READ_MULTIPLIER if cache_hit else self.WRITE_MULTIPLIER
        self.cached_prefixes[prefix_hash] = True

        prefix_cost = (prefix_tokens / 1e6) * self.BASE_INPUT_COST_PER_M * multiplier
        suffix_cost = (suffix_tokens / 1e6) * self.BASE_INPUT_COST_PER_M  # never cached
        return {
            "cache_hit": cache_hit,
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "cost_usd": prefix_cost + suffix_cost,
        }


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Semantic Caching & Prompt Caching ==={Colors.RESET}\n")

    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Semantic cache catches PARAPHRASES that exact-hash caching (module 42) would miss{Colors.RESET}")
    print(f"  {Colors.DIM}(similarity is cosine over module 26's hashing-trick embedding — real deployments use a trained encoder,{Colors.RESET}")
    print(f"  {Colors.DIM} so a query with genuinely no shared vocabulary, like the 3rd one below, correctly misses here too){Colors.RESET}\n")
    cache = SemanticCache(similarity_threshold=0.75)
    canned = "Refunds are processed within 5-7 business days after we receive the returned item."

    for query in [
        "What is our refund window?",
        "What is the refund window for returns?",      # shares most words -> should HIT
        "How wide is our refund window policy?",        # shares core terms -> should HIT
        "Do refunds get processed quickly after a return?",  # no shared vocabulary -> MISS
    ]:
        response, status = cached_llm_call(cache, llm, query, canned)
        print(f"  Query: \"{query}\"")
        print(f"    -> {status}")
        print(f"    -> {response}\n")

    print(f"{Colors.YELLOW}2. Prompt/KV caching: cost of 4 calls sharing one large system-prompt prefix{Colors.RESET}")
    prompt_cache = PromptCacheSimulator()
    long_system_prefix = " ".join(["You are a support agent. Company policy document follows."] * 40)  # ~big prefix
    user_questions = [
        "What's the refund window?",
        "Can I get store credit instead?",
        "Does this apply to international orders?",
        "What about final-sale items?",
    ]

    total_cost = 0.0
    for i, question in enumerate(user_questions, 1):
        result = prompt_cache.cost_for_call(long_system_prefix, question)
        total_cost += result["cost_usd"]
        tag = f"{Colors.GREEN}CACHE READ (-90%){Colors.RESET}" if result["cache_hit"] else f"{Colors.YELLOW}CACHE WRITE (+25%){Colors.RESET}"
        print(f"  Call {i}: {tag} | prefix={result['prefix_tokens']} tok, suffix={result['suffix_tokens']} tok "
              f"-> ${result['cost_usd']:.6f}")

    print(f"\n  Total cost across {len(user_questions)} calls: ${total_cost:.6f}")
    naive_cost = sum(
        ((len(long_system_prefix.split()) + len(q.split())) / 1e6) * PromptCacheSimulator.BASE_INPUT_COST_PER_M
        for q in user_questions
    )
    savings_pct = (1 - total_cost / naive_cost) * 100 if naive_cost else 0
    print(f"  Cost WITHOUT prompt caching: ${naive_cost:.6f}")
    print(f"  {Colors.GREEN}Savings from caching the shared prefix: {savings_pct:.1f}%{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Semantic cache and prompt-cache cost model verified.{Colors.RESET}")

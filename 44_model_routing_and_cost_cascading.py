"""
=============================================================================
44_model_routing_and_cost_cascading.py: Multi-Provider Routing & Cost Cascading
=============================================================================
Senior GenAI Engineer Context:
This is the exact pattern behind gateways like OpenRouter/OmniRoute: never
depend on one model or provider, and never pay premium-model prices for a
question a cheap model already answers well.

Three router strategies implemented from scratch:
1. WEIGHTED ROUND ROBIN: spreads load across N backends by weight (e.g. 70%
   of traffic to a cheap primary, 30% to a backup) — mirrors a "combo" with
   multiple models behind one alias.
2. FAILOVER: tries backends in priority order; if one raises (rate-limited,
   down, timed out), falls through to the next without failing the request.
3. COST CASCADING: always tries the cheap/fast model first; only escalates
   to the expensive model if a confidence heuristic on the cheap answer
   fails — the actual mechanism behind "model cascades" in production cost
   optimization.

Run directly:
  python3 44_model_routing_and_cost_cascading.py
=============================================================================
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


@dataclass
class ModelBackend:
    name: str
    client: LLMClient
    cost_per_call_usd: float
    weight: int = 1
    # Injected failure for the failover demo: after this many calls, this
    # backend starts raising (simulates a rate limit / outage).
    fail_after_calls: int = -1
    _call_count: int = field(default=0, init=False)

    def chat(self, messages: List[Dict[str, str]], simulated_response: str) -> str:
        self._call_count += 1
        if self.fail_after_calls >= 0 and self._call_count > self.fail_after_calls:
            raise RuntimeError(f"Backend '{self.name}' is rate-limited / unavailable.")
        return self.client.chat(messages, simulated_response=simulated_response)


# =====================================================================
# 1. WEIGHTED ROUND ROBIN
# =====================================================================
class WeightedRoundRobinRouter:
    def __init__(self, backends: List[ModelBackend]):
        # Expand into a schedule proportional to weight, e.g. weights [3,1]
        # -> ['A','A','A','B'], so 3-in-4 calls hit the heavier backend.
        self.schedule = [b for b in backends for _ in range(b.weight)]
        self._i = 0

    def pick(self) -> ModelBackend:
        backend = self.schedule[self._i % len(self.schedule)]
        self._i += 1
        return backend


# =====================================================================
# 2. FAILOVER ROUTER
# =====================================================================
class FailoverRouter:
    def __init__(self, backends_in_priority_order: List[ModelBackend]):
        self.backends = backends_in_priority_order

    def call(self, messages: List[Dict[str, str]], simulated_response: str) -> Tuple[str, str]:
        last_error = None
        for backend in self.backends:
            try:
                return backend.chat(messages, simulated_response), backend.name
            except RuntimeError as e:
                last_error = e
                print(f"    {Colors.RED}✖ {backend.name} failed ({e}) — falling through...{Colors.RESET}")
        raise RuntimeError(f"All backends exhausted. Last error: {last_error}")


# =====================================================================
# 3. COST CASCADE (cheap-first, escalate on low confidence)
# =====================================================================
class CascadingRouter:
    def __init__(self, cheap: ModelBackend, expensive: ModelBackend, confidence_check: Callable[[str], bool]):
        self.cheap = cheap
        self.expensive = expensive
        self.confidence_check = confidence_check

    def call(self, messages: List[Dict[str, str]], cheap_sim: str, expensive_sim: str) -> Dict[str, Any]:
        cheap_answer = self.cheap.chat(messages, simulated_response=cheap_sim)
        if self.confidence_check(cheap_answer):
            return {"answer": cheap_answer, "escalated": False, "cost_usd": self.cheap.cost_per_call_usd, "model": self.cheap.name}

        print(f"    {Colors.YELLOW}⚠ Cheap model answer failed the confidence check — escalating...{Colors.RESET}")
        expensive_answer = self.expensive.chat(messages, simulated_response=expensive_sim)
        return {
            "answer": expensive_answer,
            "escalated": True,
            "cost_usd": self.cheap.cost_per_call_usd + self.expensive.cost_per_call_usd,
            "model": self.expensive.name,
        }


def low_confidence(answer: str) -> bool:
    """A cheap, deterministic proxy for 'the model wasn't sure'. Real systems
    use logprob thresholds or a judge model instead of a keyword list — this
    demo's own live output ("impossible... to predict") shows exactly why:
    a fixed keyword list is brittle against phrasing it wasn't written for.
    """
    uncertainty_markers = [
        "i'm not sure", "i don't know", "unclear", "cannot determine",
        "impossible", "cannot predict", "can't predict", "cannot know",
        "no way to know", "hard to say", "difficult to predict", "unpredictable",
    ]
    return any(marker in answer.lower() for marker in uncertainty_markers)


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Multi-Provider Routing & Cost Cascading ==={Colors.RESET}\n")

    # All backends share one real LLMClient under the hood (there's only one
    # live gateway configured in this curriculum) — the cost/weight/failure
    # metadata is what actually differs, exactly like naming several models
    # behind one "combo" alias in a real gateway.
    base_client = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{base_client.mode.upper()}{Colors.RESET}\n")

    primary = ModelBackend("primary-cheap", base_client, cost_per_call_usd=0.0002, weight=3)
    backup = ModelBackend("backup-cheap", base_client, cost_per_call_usd=0.0002, weight=1)
    flaky = ModelBackend("flaky-provider", base_client, cost_per_call_usd=0.0002, fail_after_calls=1)
    reliable = ModelBackend("reliable-provider", base_client, cost_per_call_usd=0.0005)
    cheap_model = ModelBackend("gpt-4o-mini-tier", base_client, cost_per_call_usd=0.0002)
    expensive_model = ModelBackend("gpt-4o-tier", base_client, cost_per_call_usd=0.0050)

    print(f"{Colors.YELLOW}1. Weighted Round Robin — 3:1 split across two backends over 8 calls{Colors.RESET}")
    rr_router = WeightedRoundRobinRouter([primary, backup])
    picks = [rr_router.pick().name for _ in range(8)]
    print(f"  {picks}")
    print(f"  {Colors.DIM}(primary chosen {picks.count('primary-cheap')}/8, backup {picks.count('backup-cheap')}/8){Colors.RESET}\n")

    print(f"{Colors.YELLOW}2. Failover — the first backend fails after 1 call, router falls through{Colors.RESET}")
    failover_router = FailoverRouter([flaky, reliable])
    for i in range(1, 3):
        answer, used = failover_router.call(
            [{"role": "user", "content": "What's the capital of France?"}],
            simulated_response="Paris is the capital of France.",
        )
        print(f"  Call {i}: served by {Colors.CYAN}{used}{Colors.RESET} -> {answer}")
    print()

    print(f"{Colors.YELLOW}3. Cost Cascade — cheap model first, escalate only on low confidence{Colors.RESET}")
    cascade = CascadingRouter(cheap_model, expensive_model, confidence_check=lambda a: not low_confidence(a))

    easy_result = cascade.call(
        [{"role": "user", "content": "What is 12 * 12?"}],
        cheap_sim="144.",
        expensive_sim="144 (escalated answer, should not be needed here).",
    )
    print(f"  Easy question -> model={easy_result['model']}, escalated={easy_result['escalated']}, "
          f"cost=${easy_result['cost_usd']:.4f}, answer=\"{easy_result['answer']}\"")

    hard_result = cascade.call(
        [{"role": "user", "content": "What will the S&P 500 close at exactly one year from today?"}],
        cheap_sim="I'm not sure — that depends on unknowable future market conditions.",
        expensive_sim="No model can predict an exact future index value with certainty; here's a reasoned range instead...",
    )
    print(f"  Hard question -> model={hard_result['model']}, escalated={hard_result['escalated']}, "
          f"cost=${hard_result['cost_usd']:.4f}, answer=\"{hard_result['answer']}\"")

    print(f"\n{Colors.GREEN}✔ Weighted round robin, failover, and cost cascading all verified.{Colors.RESET}")

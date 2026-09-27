"""
=============================================================================
13_langchain_callbacks_and_streaming.py: Callbacks, Streaming & Streaming Metrics
=============================================================================
How do production systems stream, monitor, and audit LLM pipelines without
polluting business logic with logging code? LangChain's answer is the
callback system: the pipeline emits lifecycle events, and independent
handlers subscribe to them.

1. LIFECYCLE HOOKS: on_chain_start / on_llm_start / on_llm_new_token /
   on_llm_end / on_chain_end.
2. STREAMING: tokens are pushed to handlers as they arrive over a real SSE
   stream (module 01), so a UI can render them immediately.
3. STREAMING METRICS that actually matter in production:
   - TTFT (time to first token): what the user perceives as responsiveness.
   - total latency and output throughput (tokens/sec).
   - cost, from the API's REAL reported token usage.
4. AUDIT LOGGING: a handler that appends structured JSONL events to a file —
   the same pipeline, a completely independent concern.
5. STREAMING vs NON-STREAMING, measured: identical total work, but the user
   sees the first words after TTFT instead of after the full response.

Run directly:
  python3 13_langchain_callbacks_and_streaming.py
=============================================================================
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from importlib import import_module

lcel_module = import_module("10_langchain_lcel_runnables_and_pipes")
Runnable = lcel_module.Runnable
PromptTemplate = lcel_module.PromptTemplate
Colors = lcel_module.Colors
LLMClient = import_module("01_unified_llm_client").LLMClient

PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT = 3.00, 15.00  # illustrative $/1M tokens — set to your model's real pricing


# =====================================================================
# 1. BASE CALLBACK HANDLER
# =====================================================================
class BaseCallbackHandler:
    def on_chain_start(self, chain_name: str, inputs: Dict[str, Any]): ...
    def on_chain_end(self, chain_name: str, outputs: Any): ...
    def on_llm_start(self, prompt: str): ...
    def on_llm_new_token(self, token: str): ...
    def on_llm_end(self, output: str, usage: Optional[Dict[str, int]]): ...


# =====================================================================
# 2. PRODUCTION HANDLERS
# =====================================================================
class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str):
        sys.stdout.write(f"{Colors.GREEN}{token}{Colors.RESET}")
        sys.stdout.flush()


class StreamingMetricsCallback(BaseCallbackHandler):
    def __init__(self):
        self.t_start = 0.0
        self.t_first_token: Optional[float] = None
        self.t_end = 0.0
        self.chunks = 0
        self.usage: Optional[Dict[str, int]] = None

    def on_llm_start(self, prompt: str):
        self.t_start, self.t_first_token, self.chunks = time.perf_counter(), None, 0

    def on_llm_new_token(self, token: str):
        if self.t_first_token is None:
            self.t_first_token = time.perf_counter()
        self.chunks += 1

    def on_llm_end(self, output: str, usage: Optional[Dict[str, int]]):
        self.t_end, self.usage = time.perf_counter(), usage

    MIN_CHUNKS_FOR_THROUGHPUT = 10

    def report(self, output_text: str) -> Dict[str, Any]:
        ttft = (self.t_first_token - self.t_start) if self.t_first_token else float("nan")
        total = self.t_end - self.t_start
        usage = self.usage or {}
        out_tokens, in_tokens = usage.get("completion_tokens"), usage.get("prompt_tokens")
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        # If the endpoint buffers and sends the answer in a handful of chunks,
        # "tokens / (total - ttft)" divides by ~0 and is meaningless.
        buffered = self.chunks < self.MIN_CHUNKS_FOR_THROUGHPUT
        return {
            "ttft_s": ttft, "total_s": total, "stream_chunks": self.chunks, "buffered": buffered,
            "prompt_tokens": in_tokens, "completion_tokens": out_tokens, "reasoning_tokens": reasoning,
            "visible_words": len(output_text.split()),
            "tokens_per_s": None if (buffered or not out_tokens) else out_tokens / max(total - ttft, 1e-9),
            "cost_usd": ((in_tokens or 0) / 1e6 * PRICE_PER_M_INPUT + (out_tokens or 0) / 1e6 * PRICE_PER_M_OUTPUT) if usage else None,
        }


class JsonlAuditLogCallback(BaseCallbackHandler):
    """Appends one JSON object per lifecycle event — the audit trail a
    compliance team asks for, added without touching the pipeline code."""

    def __init__(self, path: str):
        self.path = path

    def _write(self, event: str, **fields):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}) + "\n")

    def on_chain_start(self, chain_name: str, inputs: Dict[str, Any]):
        self._write("chain_start", chain=chain_name, inputs=inputs)

    def on_llm_end(self, output: str, usage: Optional[Dict[str, int]]):
        self._write("llm_end", output_chars=len(output), usage=usage)

    def on_chain_end(self, chain_name: str, outputs: Any):
        self._write("chain_end", chain=chain_name, output_preview=str(outputs)[:80])


# =====================================================================
# 3. INSTRUMENTED CHAT MODEL (fires callbacks on real stream deltas)
# =====================================================================
class InstrumentedChatModel(Runnable):
    def __init__(self, llm: LLMClient, simulated_text: str):
        self.llm = llm
        self.simulated_text = simulated_text

    def invoke(self, prompt_str: str, config: Optional[Dict[str, Any]] = None) -> str:
        callbacks: List[BaseCallbackHandler] = (config or {}).get("callbacks", [])
        self.llm.last_usage = None
        for cb in callbacks:
            cb.on_llm_start(prompt_str)
        full_text = ""
        for token in self.llm.stream_chat([{"role": "user", "content": prompt_str}], simulated_response=self.simulated_text):
            full_text += token
            for cb in callbacks:
                cb.on_llm_new_token(token)
        for cb in callbacks:
            cb.on_llm_end(full_text, self.llm.last_usage)
        return full_text


def run_chain(name: str, prompt: PromptTemplate, model: InstrumentedChatModel, inputs: Dict[str, Any], callbacks: List[BaseCallbackHandler]) -> str:
    for cb in callbacks:
        cb.on_chain_start(name, inputs)
    output = model.invoke(prompt.invoke(inputs), config={"callbacks": callbacks})
    for cb in callbacks:
        cb.on_chain_end(name, output)
    return output


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 13: Callbacks, Streaming & Streaming Metrics ==={Colors.RESET}")
    client = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{client.mode.upper()}{Colors.RESET}\n")

    audit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_assets", "llm_audit_log.jsonl")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    if os.path.exists(audit_path):
        os.remove(audit_path)

    prompt = PromptTemplate("In about 60 words, describe the role of a {role} on an engineering team.")
    model = InstrumentedChatModel(client, simulated_text=(
        "A Senior GenAI Engineer designs robust orchestration graphs, implements resilient tool calling, "
        "and ensures strict observability, latency budgets, and security guardrails across LLM applications."))
    metrics = StreamingMetricsCallback()
    callbacks = [StreamingStdOutCallbackHandler(), metrics, JsonlAuditLogCallback(audit_path)]

    print(f"{Colors.YELLOW}1. Streamed output (tokens printed the moment they arrive){Colors.RESET}\n  ", end="")
    output = run_chain("RoleDescriberChain", prompt, model, {"role": "Senior GenAI Engineer"}, callbacks)

    r = metrics.report(output)
    print(f"\n\n{Colors.YELLOW}2. Streaming metrics (from the metrics callback){Colors.RESET}")
    print(f"  time to first token : {r['ttft_s']:.2f}s")
    print(f"  total latency       : {r['total_s']:.2f}s over {r['stream_chunks']} stream chunks")
    if r["buffered"]:
        print(f"  {Colors.YELLOW}⚠ only {r['stream_chunks']} chunks arrived: this endpoint (or a proxy in front of it) is buffering,")
        print(f"    so TTFT ≈ total latency and output throughput can't be measured from the stream.{Colors.RESET}")
    elif r["tokens_per_s"]:
        print(f"  output throughput   : {r['tokens_per_s']:.1f} tokens/s")
    if r["completion_tokens"]:
        print(f"  real token usage    : {r['prompt_tokens']} prompt + {r['completion_tokens']} completion "
              f"for {r['visible_words']} visible words")
        hidden = r["reasoning_tokens"] or max(0, r["completion_tokens"] - int(r["visible_words"] * 1.4))
        if hidden > 0.5 * r["completion_tokens"]:
            print(f"  {Colors.YELLOW}⚠ ~{hidden} completion tokens aren't visible text — hidden reasoning ('thinking') tokens"
                  f" that are still BILLED.{Colors.RESET}")
        print(f"  cost                : ${r['cost_usd']:.6f} at ${PRICE_PER_M_INPUT}/${PRICE_PER_M_OUTPUT} per 1M in/out tokens")
    else:
        print("  real token usage    : not reported by this endpoint for streamed calls (or offline)")

    print(f"\n{Colors.YELLOW}3. Audit log written by an independent handler{Colors.RESET}")
    with open(audit_path, encoding="utf-8") as f:
        for line in f:
            print(f"  {line.strip()[:130]}")

    print(f"\n{Colors.YELLOW}4. Why stream? Same request, non-streaming, for comparison{Colors.RESET}")
    t0 = time.perf_counter()
    client.chat([{"role": "user", "content": prompt.invoke({"role": "Senior GenAI Engineer"})}], simulated_response=model.simulated_text)
    non_stream_s = time.perf_counter() - t0
    print(f"  non-streaming: user sees NOTHING for {non_stream_s:.2f}s, then the whole answer")
    print(f"  streaming:     user sees the first words after {r['ttft_s']:.2f}s")
    print(f"  {Colors.DIM}(Live latencies vary call to call through a multi-model gateway; TTFT is the number users feel.){Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Streaming, real-usage cost tracking, and audit logging via decoupled callbacks.{Colors.RESET}")

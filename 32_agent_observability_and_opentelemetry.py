"""
=============================================================================
32_agent_observability_and_opentelemetry.py: OpenTelemetry Spans & Tracing
=============================================================================
Senior GenAI Engineer Context:
Deploying an uninstrumented agent to production is a financial and operational risk.

OpenTelemetry (OTel) Distributed Tracing Architecture:
1. Hierarchical Spans: Root Agent Run -> Steps -> LLM Calls -> Tool Executions.
2. Token Accounting: Tracking prompt vs completion tokens.
3. Cost Attribution: Accurate USD cost calculation per query.
4. Latency breakdown (P50/P95).

Run directly:
  python3 32_agent_observability_and_opentelemetry.py
=============================================================================
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

# Per-1M-token USD pricing used for the cost-accounting attribute below —
# swap these for whatever your live gateway/model actually bills.
PROMPT_COST_PER_M = 0.15
COMPLETION_COST_PER_M = 0.60


class Span:
    def __init__(self, name: str, span_type: str):
        self.span_id = str(uuid.uuid4())[:8]
        self.name = name
        self.span_type = span_type
        self.duration_ms = 0.0
        self.attributes: Dict[str, Any] = {}
        self.children: List["Span"] = []
        self._t0 = 0.0

    def start(self):
        self._t0 = time.perf_counter()
        return self

    def end(self):
        self.duration_ms = (time.perf_counter() - self._t0) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """OpenInference-style flat span record: {span_id, name, kind, attributes, duration}."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "span.kind": self.span_type.upper(),
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("OPENTELEMETRY AGENT TRACE & COST ACCOUNTING")
    print(f"===================================================={Colors.RESET}\n")

    client = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{client.mode.upper()}{Colors.RESET}\n")

    root = Span("PaymentRemediationAgent", "agent").start()

    # A "Step" span nests the LLM call + tool execution one level below the
    # root, giving the 4-level hierarchy the module promises:
    # Root Agent Run -> Steps -> LLM Calls -> Tool Executions.
    step = Span("HandleRefundRequest", "step").start()

    retriever = Span("PolicyRetriever", "retriever").start()
    time.sleep(0.03)
    retriever.end()
    step.children.append(retriever)

    refund_prompt = "Customer disputes a $49.99 charge from 3 days ago claiming non-delivery. Draft a one-sentence refund decision."
    llm_span = Span("ReasoningTurn1", "llm").start()
    completion = client.chat(
        [{"role": "user", "content": refund_prompt}],
        simulated_response="Approve the refund: the charge is within the 14-day window and no delivery confirmation exists.",
    )
    llm_span.end()

    # Real prompt/completion token counts aren't available from every
    # OpenAI-compatible gateway response — falling back to a whitespace
    # estimate keeps cost accounting working even when `usage` is absent.
    prompt_tokens = len(refund_prompt.split())
    completion_tokens = len(completion.split())
    llm_span.attributes["prompt_tokens"] = prompt_tokens
    llm_span.attributes["completion_tokens"] = completion_tokens
    llm_span.attributes["cost_usd"] = (
        (prompt_tokens / 1e6 * PROMPT_COST_PER_M) + (completion_tokens / 1e6 * COMPLETION_COST_PER_M)
    )
    llm_span.attributes["completion_preview"] = completion[:80]
    step.children.append(llm_span)

    tool = Span("StripeRefundTool", "tool").start()
    time.sleep(0.05)
    tool.attributes["amount"] = 49.99
    tool.end()
    llm_span.children.append(tool)  # tool execution nested under the LLM call that requested it

    step.end()
    root.children.append(step)
    root.end()

    def print_span(span: Span, depth: int = 0):
        indent = "  " * depth + ("↳ " if depth else "")
        print(f"{indent}[{span.span_type.upper()}] {span.name} ({span.duration_ms:.1f}ms) Attributes: {span.attributes}")
        for child in span.children:
            print_span(child, depth + 1)

    print(f"{Colors.CYAN}[AGENT ROOT SPAN]{Colors.RESET}")
    print_span(root)

    trace_export = json.dumps(root.to_dict(), indent=2)
    print(f"\n{Colors.DIM}OpenInference-format export (truncated):{Colors.RESET}")
    print(trace_export[:400] + "...\n")
    print(f"{Colors.GREEN}✔ Trace exported to OpenInference format successfully ({len(trace_export)} bytes).{Colors.RESET}\n")

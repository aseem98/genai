"""
=============================================================================
35_llm_app_ci_testing_and_golden_datasets.py: CI Testing for LLM Applications
=============================================================================
Senior GenAI Engineer Context:
Module 33 is an eval HARNESS you run manually to score agent trajectories.
This module is the CI TESTING layer that runs automatically on every pull
request, and it needs different tools because LLM output is non-deterministic:

1. MOCKING THE LLM: unit tests must not call a real (paid, slow, flaky)
   model. A `FakeLLMClient` swaps in for `LLMClient` with scripted
   responses, so prompt/parsing logic gets tested deterministically —
   exactly the "don't hit the real network in unit tests" pattern applied
   to LLMs instead of HTTP APIs or databases.
2. GOLDEN-DATASET REGRESSION TESTS: instead of asserting exact string
   equality (impossible against a live model), each golden case asserts a
   PREDICATE — "does the output contain X", "is it valid JSON with field Y"
   — the same reason snapshot testing frameworks use approximate/structural
   diffs for anything non-deterministic.
3. PROMPT-DRIFT SNAPSHOT DIFFING: stores a reference ("golden") output and
   flags when a NEW prompt/model version's output structurally diverges from
   it — the regression signal that catches "I tweaked the prompt and it
   quietly broke a downstream parser."

Run directly:
  python3 35_llm_app_ci_testing_and_golden_datasets.py
=============================================================================
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
JsonOutputParser = import_module("11_langchain_output_parsers_and_schemas").JsonOutputParser


# =====================================================================
# 1. MOCKING THE LLM FOR CI (never hits the real network)
# =====================================================================
class FakeLLMClient:
    """Drop-in replacement for `LLMClient` in unit tests: same `.chat()`
    signature, but returns scripted responses keyed by a substring match on
    the prompt — deterministic, free, and instant.
    """

    def __init__(self, scripted_responses: Dict[str, str]):
        self.scripted_responses = scripted_responses
        self.call_log: List[str] = []
        self.mode = "mocked"

    def chat(self, messages: List[Dict[str, str]], tools=None, simulated_response: Optional[str] = None) -> str:
        prompt = messages[-1]["content"]
        self.call_log.append(prompt)
        for key_substring, response in self.scripted_responses.items():
            if key_substring in prompt:
                return response
        raise AssertionError(f"FakeLLMClient received an un-mocked prompt: {prompt!r}")


# The actual "app" under test: a tiny function that prompts an LLM and
# parses structured output — standing in for a real agent endpoint.
def classify_support_ticket(llm, json_parser: JsonOutputParser, ticket_text: str) -> Dict[str, Any]:
    prompt = (
        f"Classify this support ticket as JSON with keys category (billing/technical/account) "
        f"and urgency (low/medium/high).\nTicket: {ticket_text}\n{json_parser.get_format_instructions()}"
    )
    raw = llm.chat([{"role": "user", "content": prompt}])
    return json_parser.invoke(raw)


# =====================================================================
# 2. GOLDEN-DATASET REGRESSION TESTS (predicate-based, not exact match)
# =====================================================================
@dataclass
class GoldenCase:
    name: str
    ticket_text: str
    mock_response: str
    predicate: Callable[[Dict[str, Any]], bool]
    predicate_description: str


GOLDEN_CASES = [
    GoldenCase(
        name="billing_ticket_classified_correctly",
        ticket_text="I was charged twice for my subscription this month.",
        mock_response=json.dumps({"category": "billing", "urgency": "medium"}),
        predicate=lambda r: r.get("category") == "billing",
        predicate_description="category must be 'billing'",
    ),
    GoldenCase(
        name="outage_ticket_flagged_high_urgency",
        ticket_text="Production API is returning 500s for all customers right now.",
        mock_response=json.dumps({"category": "technical", "urgency": "high"}),
        predicate=lambda r: r.get("urgency") == "high",
        predicate_description="urgency must be 'high' for an active outage",
    ),
    GoldenCase(
        name="malformed_llm_output_is_caught_not_silently_swallowed",
        ticket_text="Can't log in to my account.",
        mock_response="Sure! This looks like an account issue.",  # NOT valid JSON — simulates prompt drift
        predicate=lambda r: False,  # never reached; ValueError should fire first
        predicate_description="parser must raise, not silently return garbage",
        ) ,
]


def run_golden_suite(cases: List[GoldenCase]) -> Dict[str, Any]:
    json_parser = JsonOutputParser()
    results = {"passed": 0, "failed": 0, "details": []}

    for case in cases:
        fake_llm = FakeLLMClient({case.ticket_text: case.mock_response})
        try:
            parsed = classify_support_ticket(fake_llm, json_parser, case.ticket_text)
            ok = case.predicate(parsed)
            error = None if ok else f"predicate failed ({case.predicate_description}): got {parsed}"
        except ValueError as e:
            # The one deliberately-malformed case is EXPECTED to raise —
            # anything else raising is a genuine test failure.
            ok = "malformed" in case.name
            error = None if ok else f"unexpected parse failure: {e}"

        results["passed" if ok else "failed"] += 1
        results["details"].append({"case": case.name, "passed": ok, "error": error})

    return results


# =====================================================================
# 3. SNAPSHOT DIFFING (prompt-drift detection across runs)
# =====================================================================
def diff_against_snapshot(golden_snapshot: Dict[str, Any], new_output: Dict[str, Any]) -> List[str]:
    """Flags structural drift: new/missing keys or a changed value type —
    the actual signal a prompt-drift snapshot test cares about, since exact
    value equality is too strict for non-deterministic LLM output.
    """
    drifts = []
    for key in golden_snapshot:
        if key not in new_output:
            drifts.append(f"key '{key}' present in golden snapshot but MISSING from new output")
        elif type(new_output[key]) is not type(golden_snapshot[key]):
            drifts.append(f"key '{key}' changed type: {type(golden_snapshot[key]).__name__} -> {type(new_output[key]).__name__}")
    for key in new_output:
        if key not in golden_snapshot:
            drifts.append(f"key '{key}' is NEW — not present in the golden snapshot")
    return drifts


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== CI Testing for LLM Applications ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Golden-dataset regression suite (runs against a MOCKED LLM — no network, no cost){Colors.RESET}")
    report = run_golden_suite(GOLDEN_CASES)
    for detail in report["details"]:
        icon = f"{Colors.GREEN}PASS{Colors.RESET}" if detail["passed"] else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  [{icon}] {detail['case']}")
        if detail["error"]:
            print(f"         {Colors.RED}↳ {detail['error']}{Colors.RESET}")
    print(f"\n  Summary: {Colors.GREEN}{report['passed']} passed{Colors.RESET}, {Colors.RED}{report['failed']} failed{Colors.RESET}\n")

    print(f"{Colors.YELLOW}2. Snapshot diffing — catching prompt drift between two model/prompt versions{Colors.RESET}")
    golden_snapshot = {"category": "billing", "urgency": "medium", "confidence": 0.91}
    drifted_output = {"category": "billing", "urgency_level": "medium"}  # key renamed + one dropped
    drifts = diff_against_snapshot(golden_snapshot, drifted_output)
    print(f"  Golden snapshot: {golden_snapshot}")
    print(f"  New output:      {drifted_output}")
    for d in drifts:
        print(f"  {Colors.RED}⚠ DRIFT: {d}{Colors.RESET}")
    if not drifts:
        print(f"  {Colors.GREEN}No structural drift detected.{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Mocked-LLM unit testing, golden regression suite, and snapshot drift detection all verified.{Colors.RESET}")
    print(f"{Colors.DIM}CI equivalent: fail the pipeline if `report['failed'] > 0` or if `drifts` is non-empty on a protected prompt.{Colors.RESET}")

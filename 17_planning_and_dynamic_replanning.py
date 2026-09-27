"""
=============================================================================
17_planning_and_dynamic_replanning.py: Plan-and-Solve & Dynamic Replanning
=============================================================================
Theory:
The Plan-and-Solve pattern (Wang et al., 2023) decouples reasoning:
1. Planning Phase: A dedicated Planner breaks the goal into structured subtasks.
2. Execution Phase: An Executor takes each subtask one by one.
3. Replanning Phase: If an intermediate step fails, a Replanner dynamically
   mutates the remaining plan without restarting from scratch.

Run directly:
  python3 17_planning_and_dynamic_replanning.py
=============================================================================
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
JsonOutputParser = import_module("11_langchain_output_parsers_and_schemas").JsonOutputParser

VALID_TOOLS = ["check_git", "run_tests", "deploy"]


@dataclass
class PlanStep:
    step_id: Any  # int for original steps, "N.1"/"N.2" for injected remediation steps
    description: str
    tool: str
    status: str = "PENDING"


def mock_environment(tool: str, state: Dict[str, Any]) -> Dict[str, Any]:
    if tool == "check_git":
        return {"status": "success", "branch": "main"}
    elif tool == "run_tests":
        # Fails until the legacy test has actually been auto-patched, so a
        # replanned rerun of this step can genuinely verify the fix worked.
        if state.get("patched"):
            return {"status": "success", "passed": 42}
        return {"status": "failure", "error": "AssertionError on line 42 in test_legacy.py"}
    elif tool == "auto_patch":
        state["patched"] = True
        return {"status": "success", "patched": "test_legacy.py"}
    elif tool == "deploy":
        return {"status": "success", "url": "https://staging.internal"}
    return {"status": "error"}


def generate_plan(llm: LLMClient, parser: JsonOutputParser, goal: str) -> List[PlanStep]:
    """Planner phase: asks the LLM to decompose the goal into an ordered
    sequence of steps, each bound to one of the tools the executor actually
    understands. Falls back to a sane default plan if a (likely smaller/free)
    live model doesn't return valid, schema-conforming JSON.
    """
    prompt = (
        f"Break this goal into an ordered JSON array of steps to reach it. "
        f"Each step is an object with 'description' (string) and 'tool' "
        f"(must be exactly one of {VALID_TOOLS}).\n"
        f"Goal: {goal}\n\n{parser.get_format_instructions()}"
    )
    default_plan = [
        {"description": "Verify git status", "tool": "check_git"},
        {"description": "Run automated test suite", "tool": "run_tests"},
        {"description": "Deploy to staging", "tool": "deploy"},
    ]
    raw = llm.chat([{"role": "user", "content": prompt}], simulated_response=json.dumps(default_plan))

    try:
        steps = parser.invoke(raw)
        assert isinstance(steps, list) and all(s.get("tool") in VALID_TOOLS for s in steps)
    except (ValueError, AssertionError):
        print(f"  {Colors.YELLOW}⚠ Planner returned an invalid plan — falling back to the default 3-step plan.{Colors.RESET}")
        steps = default_plan

    return [PlanStep(i + 1, s["description"], s["tool"]) for i, s in enumerate(steps)]


def generate_remediation_description(llm: LLMClient, failure_error: str) -> str:
    """Replanner phase: asks the LLM to explain, in one sentence, the
    corrective step it's about to inject — the tool itself stays the
    deterministic `auto_patch` action the executor actually knows how to run.
    """
    prompt = (
        f"A pipeline step just failed with this error:\n{failure_error}\n"
        "In one short sentence, describe the auto-patch remediation step we're about to take."
    )
    return llm.chat(
        [{"role": "user", "content": prompt}],
        simulated_response="Self-heal: Auto-patch legacy test",
    ).strip()


if __name__ == "__main__":
    llm = LLMClient()
    json_parser = JsonOutputParser()

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("PLANNER: LLM decomposes the goal into an executable plan")
    print(f"===================================================={Colors.RESET}\n")
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    goal = "Ship the current branch: confirm git state, run the test suite, then deploy to staging."
    plan = generate_plan(llm, json_parser, goal)
    for step in plan:
        print(f"  [{step.step_id}] {step.description}  (tool: {step.tool})")

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("EXECUTING PLAN-AND-SOLVE WORKFLOW")
    print(f"===================================================={Colors.RESET}\n")

    env_state: Dict[str, Any] = {"patched": False}
    idx = 0
    while idx < len(plan):
        step = plan[idx]
        print(f"{Colors.BOLD}[Step {step.step_id}]: {step.description}{Colors.RESET}")
        res = mock_environment(step.tool, env_state)

        if res["status"] == "success":
            print(f"  {Colors.GREEN}✔ Passed: {res}{Colors.RESET}\n")
            idx += 1
        else:
            print(f"  {Colors.RED}✖ Failed: {res['error']}{Colors.RESET}")
            remediation_desc = generate_remediation_description(llm, res["error"])
            print(f"  {Colors.YELLOW}⚡ [Dynamic Replanning]: {remediation_desc}{Colors.RESET}\n")
            # Inject a corrective step AND a re-run of the failed step, so the
            # patch is actually verified before the plan proceeds to deploy.
            plan.insert(idx + 1, PlanStep(f"{step.step_id}.1", remediation_desc, "auto_patch"))
            plan.insert(idx + 2, PlanStep(f"{step.step_id}.2", "Re-verify: Rerun test suite after patch", "run_tests"))
            # Point to the newly injected remediation step
            idx += 1

    print(f"{Colors.BOLD}{Colors.GREEN}★ All Planned and Replanned Steps Executed Successfully!{Colors.RESET}\n")

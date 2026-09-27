"""
=============================================================================
20_reflection_and_self_correction.py: Reflexion & Evaluator-Optimizer
=============================================================================
Theory:
In the Reflexion architecture (Shinn et al., NeurIPS 2023):
1. Actor / Generator generates a solution draft.
2. Evaluator / Critic tests the output against automated rubrics/unit tests.
3. Self-Reflection produces verbal feedback ("Where did I go wrong?").
4. Generator incorporates verbal critique to fix bugs until tests pass 100%.

Run directly:
  python3 20_reflection_and_self_correction.py
=============================================================================
"""

import re
from typing import Dict, Any, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

sandbox_module = import_module("19_sandboxed_execution_and_least_privilege")
vet_code = sandbox_module.vet_code
UnsafeCodeError = sandbox_module.UnsafeCodeError


def extract_code(raw_text: str) -> str:
    """A live model — especially an agentic coding model that defaults to
    narrating and emitting tool calls instead of plain code — often wraps
    (or buries) the function in markdown fences or prose. Pull the fenced
    code block out if present; otherwise assume the whole reply is code.
    """
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    return match.group(1) if match else raw_text.strip()


class CodeEvaluator:
    @staticmethod
    def evaluate(code_str: str, test_suite: List[Tuple[str, Any]]) -> Tuple[bool, int, str]:
        # Static AST vetting (module 19) BEFORE anything is executed — a
        # generator that hallucinates an `import os` or a sandbox-escape
        # gets a critical-failure critique fed straight back into the
        # reflection loop, exactly like a failing test would.
        try:
            vet_code(code_str)
        except UnsafeCodeError as e:
            return False, 0, f"Rejected by sandbox vetting (never executed): {e}"

        local_scope = {}
        try:
            # Note: an empty globals dict does NOT strip builtins — Python
            # auto-injects the real `__builtins__` whenever it's absent, so
            # this in-process exec still has full builtin access. The AST
            # vetting above is the actual defense; module 19's subprocess
            # sandbox is what you'd swap in for a hard OS-level boundary.
            exec(code_str, {}, local_scope)
        except Exception as e:
            return False, 0, f"Compilation Error: {e}"

        passed = 0
        critiques = []
        for expr, expected in test_suite:
            try:
                actual = eval(expr, local_scope)
                if actual == expected:
                    passed += 1
                else:
                    critiques.append(f"Failed '{expr}': got {actual}, expected {expected}")
            except Exception as e:
                critiques.append(f"Exception on '{expr}': {e}")

        score = int((passed / len(test_suite)) * 100)
        return (passed == len(test_suite)), score, "\n".join(critiques)


if __name__ == "__main__":
    task = "Write `parse_version(v: str) -> tuple` splitting '1.2.3' to (1, 2, 3), stripping 'v' prefixes and padding missing patch with 0."

    tests = [
        ("parse_version('1.2.3')", (1, 2, 3)),
        ("parse_version('v2.4.1')", (2, 4, 1)),
        ("parse_version('v1.5')", (1, 5, 0))
    ]

    # Each candidate the Generator would produce, keyed by what it's meant to
    # fix relative to the previous attempt's critique (used as `simulated_response`
    # below, so simulation mode still routes through the same LLM call as live mode).
    scripted_generations = [
        # Attempt 1 (no prior critique): naive split (crashes on 'v')
        """def parse_version(v: str):\n    return tuple(int(x) for x in v.split('.'))""",
        # Attempt 2 (incorporates critique about the "v" crash): fixes 'v' but misses padding
        """def parse_version(v: str):\n    return tuple(int(x) for x in v.lstrip('v').split('.'))""",
        # Attempt 3 (incorporates critique about missing patch padding): perfect self-repaired code
        """def parse_version(v: str):\n    parts = [int(x) for x in v.lstrip('v').split('.')]\n    while len(parts) < 3: parts.append(0)\n    return tuple(parts[:3])"""
    ]

    llm = LLMClient()
    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("REFLEXION & SELF-CORRECTION LOOP")
    print(f"===================================================={Colors.RESET}\n")

    critique = ""
    code = None
    for i in range(1, len(scripted_generations) + 1):
        print(f"{Colors.BOLD}{Colors.CYAN}--- Attempt {i} ---{Colors.RESET}")

        # Generator step: the previous critique is fed back into the prompt,
        # so a live LLM would see exactly what to fix. Simulation mode is
        # pinned to the scripted next-attempt for a repeatable demo.
        gen_prompt = (
            f"Task: {task}\n"
            "Respond with ONLY a single ```python fenced code block defining the function. "
            "No narration, no tool calls, no explanation — just the function definition.\n"
        )
        if critique:
            gen_prompt += f"Previous attempt failed with:\n{critique}\nFix these issues.\n"
        raw_response = llm.chat(
            [{"role": "user", "content": gen_prompt}],
            simulated_response=scripted_generations[i - 1]
        )
        code = extract_code(raw_response)
        print(f"Code:\n{code}\n")

        passed, score, critique = CodeEvaluator.evaluate(code, tests)
        print(f"Score: {Colors.BOLD}{score}/100{Colors.RESET}")

        if passed:
            print(f"{Colors.GREEN}★ SUCCESS! All tests passed on attempt {i}.{Colors.RESET}\n")
            break
        else:
            print(f"{Colors.RED}Test Failures:{Colors.RESET}\n{critique}")
            print(f"{Colors.MAGENTA}↳ Verbal Self-Reflection Saved for Next Iteration.{Colors.RESET}\n")

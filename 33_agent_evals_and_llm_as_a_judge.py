"""
=============================================================================
33_agent_evals_and_llm_as_a_judge.py: CI/CD Evals & Benchmarks
=============================================================================
Senior GenAI Engineer Context:
How do you know if a prompt change or model fine-tune improved or degraded
your agent in production?

Core Evaluation Metrics:
1. Trajectory Accuracy: Did the agent take the expected sequence of tool calls?
2. Step Efficiency: Ratio of optimal steps to actual steps (penalizing wandering).
3. LLM-as-a-Judge: Automated evaluation with explicit rubric scores (0-15).

Run directly:
  python3 33_agent_evals_and_llm_as_a_judge.py
=============================================================================
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
JsonOutputParser = import_module("11_langchain_output_parsers_and_schemas").JsonOutputParser


@dataclass
class EvalResult:
    test_id: str
    expected_tools: List[str]
    actual_tools: List[str]
    step_efficiency: float
    judge_score: int
    feedback: str


def trajectory_accuracy(expected_tools: List[str], actual_tools: List[str]) -> float:
    """Fraction of the expected tool sequence that was actually invoked, in
    order — a deterministic metric computed from the trajectory itself, not
    an LLM opinion."""
    if not expected_tools:
        return 1.0
    matched = sum(1 for e, a in zip(expected_tools, actual_tools) if e == a)
    return matched / len(expected_tools)


def llm_judge(
    llm: LLMClient,
    json_parser: JsonOutputParser,
    task: str,
    agent_final_answer: str,
    wasted_steps: int = 0,
) -> Dict[str, Any]:
    """LLM-as-a-Judge: scores the agent's final answer against an explicit
    rubric (0-15) and returns structured, parseable feedback rather than a
    single opaque number.
    """
    rubric_prompt = (
        f"You are grading an AI agent's final answer.\n"
        f"Task given to the agent: {task}\n"
        f"Agent's final answer: {agent_final_answer}\n"
        f"The agent used {wasted_steps} extraneous tool call(s) beyond the minimal solution.\n\n"
        "Score it 0-15 across correctness (0-8), efficiency (0-4, penalize extraneous calls), "
        "and clarity (0-3). Return JSON with keys: correctness, efficiency, clarity, total, "
        "feedback (one sentence).\n"
        f"{json_parser.get_format_instructions()}"
    )
    sim_efficiency = max(0, 4 - 2 * wasted_steps)
    raw = llm.chat(
        [{"role": "user", "content": rubric_prompt}],
        simulated_response=json.dumps({
            "correctness": 8, "efficiency": sim_efficiency, "clarity": 3,
            "total": 8 + sim_efficiency + 3,
            "feedback": (
                "Optimal trajectory, zero extraneous tools called." if wasted_steps == 0
                else f"Correct final output, but wasted {wasted_steps} step(s) on an unnecessary tool call."
            ),
        }),
    )
    try:
        return json_parser.invoke(raw)
    except ValueError:
        return {"correctness": 0, "efficiency": 0, "clarity": 0, "total": 0, "feedback": f"Judge returned unparseable output: {raw[:80]}"}


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("AGENT EVALUATION HARNESS & LLM-AS-A-JUDGE")
    print(f"===================================================={Colors.RESET}\n")

    llm = LLMClient()
    json_parser = JsonOutputParser()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    eval_cases = [
        {
            "test_id": "EVAL-01",
            "task": "Convert NVIDIA's stock price to EUR.",
            "expected_tools": ["stock_quote", "fx_converter"],
            "actual_tools": ["stock_quote", "fx_converter"],
            "agent_final_answer": "NVIDIA is trading at $131.20, which is approximately €120.75.",
        },
        {
            "test_id": "EVAL-02",
            "task": "Calculate the variance of this week's latency samples.",
            "expected_tools": ["calculate_variance"],
            "actual_tools": ["web_search", "calculate_variance"],
            "agent_final_answer": "The variance of the latency samples is 42.7 ms².",
        },
    ]

    for case in eval_cases:
        step_efficiency = len(case["expected_tools"]) / len(case["actual_tools"])
        wasted_steps = len(case["actual_tools"]) - len(case["expected_tools"])
        traj_acc = trajectory_accuracy(case["expected_tools"], case["actual_tools"])
        verdict = llm_judge(llm, json_parser, case["task"], case["agent_final_answer"], wasted_steps)

        result = EvalResult(
            test_id=case["test_id"],
            expected_tools=case["expected_tools"],
            actual_tools=case["actual_tools"],
            step_efficiency=step_efficiency,
            judge_score=verdict.get("total", 0),
            feedback=verdict.get("feedback", ""),
        )

        print(f"{Colors.BOLD}[{result.test_id}]{Colors.RESET}")
        print(f"  • Tools Expected:     {result.expected_tools}")
        print(f"  • Tools Executed:     {result.actual_tools}")
        print(f"  • Trajectory Accuracy:{Colors.CYAN} {traj_acc * 100:.0f}%{Colors.RESET}")
        color = Colors.GREEN if result.step_efficiency == 1.0 else Colors.YELLOW
        print(f"  • Step Efficiency:    {color}{result.step_efficiency * 100:.0f}%{Colors.RESET}")
        print(f"  • Judge Score:        {result.judge_score}/15 ({result.feedback})\n")

"""
=============================================================================
14_react_agent_reasoning_and_acting.py: The ReAct Pattern (Reasoning + Acting)
=============================================================================
ReAct (Yao et al., 2022) interleaves reasoning and tool use in a loop:
  Thought -> Action -> Action Input -> Observation -> ... -> Final Answer

The details that make it work in practice, all implemented here:
1. REAL TOOLS: `search` retrieves from an internal knowledge base with
   module 12's keyword retriever; `calculator` evaluates arithmetic with a
   whitelisting AST walker — never `eval()`, which would execute arbitrary
   code the model writes.
2. STOP SEQUENCES: the model is stopped at "Observation:" so it can't invent
   tool results; the runtime inserts the REAL observation instead.
3. FORMAT REPAIR: if the model drifts from the format, one corrective
   re-prompt before giving up.
4. STEP LIMIT: a hard cap so a confused agent can't loop forever.
5. TRAJECTORY RECORDING: every (action, input, observation) is kept, so the
   run can be audited and scored — module 33 evaluates these trajectories.
6. VERIFICATION: the final answer is checked against ground truth computed
   directly from the knowledge base.

Run directly:
  python3 14_react_agent_reasoning_and_acting.py
=============================================================================
"""

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
SimpleRetriever = import_module("12_langchain_retrievers_and_chains").SimpleRetriever


# =====================================================================
# 1. TOOLS
# =====================================================================
SERVICE_KNOWLEDGE_BASE = [
    "payments-api service metrics: p99 latency is 420 ms; error rate is 0.8%; owner is Team Ledger.",
    "auth-api service metrics: p99 latency is 180 ms; error rate is 0.1%; owner is Team Identity.",
    "search-api service metrics: p99 latency is 350 ms; error rate is 0.3%; owner is Team Discovery.",
    "Platform SLO policy: every customer-facing service must keep p99 latency at or below 300 ms.",
    "Incident policy: a service breaching its SLO for 3 consecutive days must open a reliability ticket.",
]
_retriever = SimpleRetriever(SERVICE_KNOWLEDGE_BASE)


def search(query: str) -> str:
    """Keyword retrieval over the internal knowledge base (top-2 documents)."""
    return " | ".join(_retriever.invoke(query))


def list_services(_: str = "") -> str:
    """Enumerates EVERY service in the knowledge base. For "which of all X"
    questions, completeness must not depend on search recall — a top-k
    search can silently miss one, and the agent would never know."""
    return ", ".join(re.match(r"(\S+) service metrics", d).group(1) for d in SERVICE_KNOWLEDGE_BASE if " service metrics" in d)


_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
                   ast.Pow: operator.pow, ast.Mod: operator.mod}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression: str) -> str:
    """Evaluates arithmetic by walking the parsed AST and allowing ONLY
    number literals and + - * / ** % — function calls, names, attribute
    access and everything else are rejected before anything runs."""
    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError(f"disallowed expression element: {type(node).__name__}")
    try:
        result = evaluate(ast.parse(expression.strip().replace("^", "**"), mode="eval"))
        return str(round(result, 10))
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"


TOOLS: Dict[str, Callable[[str], str]] = {"list_services": list_services, "search": search, "calculator": calculator}

REACT_SYSTEM_PROMPT = f"""You are a reasoning agent that answers questions using tools.
Tools:
- list_services: lists every service name in the knowledge base. Action Input: none
- search: searches the internal service knowledge base. Action Input: a search query.
- calculator: evaluates an arithmetic expression like '420 - 300'. Action Input: the expression.

Use EXACTLY this format, one step per reply:
Thought: <your reasoning>
Action: <one of: {', '.join(TOOLS)}>
Action Input: <the input>

After you receive an Observation, continue. When you know the answer, reply:
Thought: I now know the final answer
Final Answer: <the answer>"""


# =====================================================================
# 2. REACT AGENT
# =====================================================================
@dataclass
class AgentRun:
    question: str
    final_answer: Optional[str] = None
    trajectory: List[Dict[str, str]] = field(default_factory=list)  # [{"action", "input", "observation"}]
    steps: int = 0


class ReActAgent:
    def __init__(self, llm: LLMClient, max_steps: int = 6, verbose: bool = True):
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _next_step(self, messages, sim_resp):
        return self.llm.chat(messages, simulated_response=sim_resp, temperature=0, stop=["\nObservation", "Observation:"]).strip()

    def run(self, question: str, scripted_steps: Optional[List[str]] = None) -> AgentRun:
        run = AgentRun(question=question)
        scratchpad = f"Question: {question}\n"
        scripted = list(scripted_steps or [])

        while run.steps < self.max_steps:
            run.steps += 1
            self._log(f"{Colors.DIM}--- Step {run.steps} ---{Colors.RESET}")
            messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}, {"role": "user", "content": scratchpad}]
            sim_resp = scripted.pop(0) if scripted else None
            response = self._next_step(messages, sim_resp)
            self._log(f"{Colors.CYAN}{response}{Colors.RESET}")

            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", response)
            input_match = re.search(r"Action Input:[ \t]*([^\n]*)", response)
            if "Final Answer:" in response and not action_match:
                run.final_answer = response.split("Final Answer:", 1)[1].strip()
                return run

            if not action_match or not input_match:
                self._log(f"{Colors.YELLOW}⚠ Format drift — re-prompting once for the strict ReAct format...{Colors.RESET}")
                messages += [{"role": "assistant", "content": response},
                             {"role": "user", "content": "That didn't follow the format. Reply with exactly 'Thought:', "
                                                         "'Action:' and 'Action Input:' lines, or 'Final Answer:'."}]
                response = self._next_step(messages, sim_resp)
                action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", response)
                input_match = re.search(r"Action Input:[ \t]*([^\n]*)", response)
                if "Final Answer:" in response and not action_match:
                    run.final_answer = response.split("Final Answer:", 1)[1].strip()
                    return run
                if not action_match or not input_match:
                    self._log(f"{Colors.RED}Format still invalid after repair — halting.{Colors.RESET}")
                    return run

            action, action_input = action_match.group(1).strip(), input_match.group(1).strip().strip("'\"")
            observation = TOOLS[action](action_input) if action in TOOLS else f"Error: unknown tool '{action}'"
            run.trajectory.append({"action": action, "input": action_input, "observation": observation})
            self._log(f"{Colors.YELLOW}⚡ {action}({action_input!r}){Colors.RESET}\n{Colors.BLUE}Observation: {observation}{Colors.RESET}")
            scratchpad += f"{response}\nObservation: {observation}\n"

        self._log(f"{Colors.RED}Step limit ({self.max_steps}) reached without a final answer.{Colors.RESET}")
        return run


def expected_slo_breach() -> Dict[str, int]:
    """Ground truth computed straight from the knowledge base, independently of the agent."""
    slo = int(re.search(r"at or below (\d+) ms", SERVICE_KNOWLEDGE_BASE[3]).group(1))
    latencies = {m.group(1): int(m.group(2)) for doc in SERVICE_KNOWLEDGE_BASE
                 for m in [re.match(r"(\S+) service metrics: p99 latency is (\d+) ms", doc)] if m}
    return {svc: lat - slo for svc, lat in latencies.items() if lat > slo}


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 14: ReAct Agent ==={Colors.RESET}")
    client = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{client.mode.upper()}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}Tool sanity checks{Colors.RESET}")
    print(f"  calculator('420 - 300')        = {calculator('420 - 300')}")
    malicious = "__import__('os').system('ls')"
    print(f"  calculator({malicious!r}) = {calculator(malicious)}  <- rejected, never executed\n")

    question = "Which services breach the p99 latency SLO, and by how many milliseconds each?"
    print(f"{Colors.BOLD}Question:{Colors.RESET} {question}\n")
    scripted = [
        "Thought: First I need the full list of services.\nAction: list_services\nAction Input: none",
        "Thought: Now the SLO threshold.\nAction: search\nAction Input: p99 latency SLO policy",
        "Thought: The SLO is 300 ms. Now the p99 latency of each service.\nAction: search\nAction Input: payments-api search-api p99 latency",
        "Thought: payments-api is 420 ms.\nAction: calculator\nAction Input: 420 - 300",
        "Thought: search-api is 350 ms.\nAction: calculator\nAction Input: 350 - 300",
        "Thought: I now know the final answer\nFinal Answer: payments-api breaches by 120 ms and search-api by 50 ms; auth-api (180 ms) is within the SLO.",
    ]
    result = ReActAgent(client, max_steps=8).run(question, scripted_steps=scripted)

    print(f"\n{Colors.BOLD}Final answer:{Colors.RESET} {result.final_answer}")
    print(f"Trajectory ({len(result.trajectory)} tool calls): {[t['action'] for t in result.trajectory]}")
    truth = expected_slo_breach()
    answer = result.final_answer or ""
    checks = {svc: (svc in answer and str(ms) in answer) for svc, ms in truth.items()}
    print(f"Ground truth from the KB: {truth}")
    for svc, ok in checks.items():
        print(f"  {svc}: {Colors.GREEN + '✔ correct' if ok else Colors.RED + '✖ missing/wrong'}{Colors.RESET}")
    print(f"\n{Colors.GREEN if all(checks.values()) else Colors.YELLOW}"
          f"{'✔ Agent answer verified against the knowledge base.' if all(checks.values()) else '⚠ Agent answer did not fully match ground truth.'}{Colors.RESET}")

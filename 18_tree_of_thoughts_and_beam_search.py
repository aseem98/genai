"""
=============================================================================
18_tree_of_thoughts_and_beam_search.py: Combinatorial Planning & Beam Search
=============================================================================
Senior GenAI Engineer Context:
Linear agents fail when complex multi-step choices lead to dead-ends.
Tree of Thoughts (ToT) uses heuristic state evaluation to explore branches,
prune weak paths with beam search, and backtrack out of dead ends.

Run directly:
  python3 18_tree_of_thoughts_and_beam_search.py
=============================================================================
"""

import json
from typing import List, Tuple
from dataclasses import dataclass
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
JsonOutputParser = import_module("11_langchain_output_parsers_and_schemas").JsonOutputParser


@dataclass
class PlanNode:
    name: str
    score: float
    cost: float


BUDGET_CAP_USD = 1500.0

GOAL = "Design a disaster-recovery strategy for a payments database with an RPO under 5 minutes."

# Cost estimates stay fixed (real infra pricing, not something an LLM should
# invent) — only the heuristic `score` for each branch is LLM-generated below.
DEPTH1_CANDIDATES = [
    ("Active-Active DynamoDB Global Tables", 1200.0),
    ("Active-Passive Warm Standby S3", 350.0),
    ("Single AZ Failover", 150.0),
]

# Depth-2 refinement options, keyed by which depth-1 branch they extend.
DEPTH2_CANDIDATES = {
    "Active-Active DynamoDB Global Tables": [
        ("Cross-region KMS re-encryption", 900.0),
        ("Manual conflict resolution Lambda", 700.0),
    ],
    "Active-Passive Warm Standby S3": [
        ("Automated Route53 health-check failover", 200.0),
        ("Manual DNS cutover runbook", 100.0),
    ],
}


def llm_score_branch(llm: LLMClient, parser: JsonOutputParser, goal: str, branch_name: str, fallback_score: float) -> float:
    """The 'heuristic state evaluation' step of Tree of Thoughts: asks the
    LLM to rate how well this branch serves the goal, on a 0-1 scale — this
    is the actual value function driving the beam search below, not a
    hand-picked constant.
    """
    prompt = (
        f"Goal: {goal}\n"
        f"Candidate approach: \"{branch_name}\"\n"
        "Rate how well this approach achieves the goal, from 0.0 (terrible) to 1.0 (excellent). "
        'Return JSON: {"score": <float>}'
    )
    raw = llm.chat([{"role": "user", "content": prompt}], simulated_response=json.dumps({"score": fallback_score}))
    try:
        return float(parser.invoke(raw)["score"])
    except (ValueError, KeyError, TypeError):
        return fallback_score


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("TREE OF THOUGHTS BEAM SEARCH (BEAM WIDTH = 2)")
    print(f"===================================================={Colors.RESET}\n")

    llm = LLMClient()
    json_parser = JsonOutputParser()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}")
    print(f"Goal: {GOAL}\n")

    # Depth 1 candidate branches — LLM-scored heuristic evaluation
    fallback_scores_d1 = {"Active-Active DynamoDB Global Tables": 0.90, "Active-Passive Warm Standby S3": 0.55, "Single AZ Failover": 0.20}
    d1 = [
        PlanNode(name, llm_score_branch(llm, json_parser, GOAL, name, fallback_scores_d1[name]), cost)
        for name, cost in DEPTH1_CANDIDATES
    ]

    print(f"{Colors.CYAN}--- Depth 1 Branches ---{Colors.RESET}")
    for n in d1:
        print(f"  • {n.name:<40} (Score: {n.score:.2f}, Cost: ${n.cost})")

    # Beam pruning (keep top 2 by heuristic score)
    d1_survivors = sorted(d1, key=lambda x: x.score, reverse=True)[:2]
    print(f"\n{Colors.MAGENTA}↳ Beam Pruning Retained Top 2 Paths:{Colors.RESET}")
    for s in d1_survivors:
        print(f"   ✔ {s.name}")

    # Expand each surviving beam into depth-2 children and evaluate the
    # heuristic against a hard constraint (budget cap). A branch that looked
    # best at depth 1 can still turn out to be a dead end at depth 2 — that's
    # exactly the case beam search + backtracking is meant to handle.
    print(f"\n{Colors.CYAN}--- Depth 2 Expansion (evaluating against ${BUDGET_CAP_USD} budget cap) ---{Colors.RESET}")
    viable_trajectories: List[Tuple[float, str, PlanNode, PlanNode]] = []
    dead_ends: List[str] = []

    fallback_scores_d2 = {
        "Cross-region KMS re-encryption": 0.92, "Manual conflict resolution Lambda": 0.40,
        "Automated Route53 health-check failover": 0.85, "Manual DNS cutover runbook": 0.60,
    }
    for parent in d1_survivors:
        children = [
            PlanNode(name, llm_score_branch(llm, json_parser, GOAL, name, fallback_scores_d2[name]), cost)
            for name, cost in DEPTH2_CANDIDATES[parent.name]
        ]
        best_child = None
        for child in children:
            total_cost = parent.cost + child.cost
            fits_budget = total_cost <= BUDGET_CAP_USD
            print(f"   {parent.name} -> {child.name:<38} (Score: {child.score:.2f}, "
                  f"Total Cost: ${total_cost:.0f}) {'✔ fits budget' if fits_budget else '✖ exceeds budget'}")
            if fits_budget and (best_child is None or child.score > best_child.score):
                best_child = child
        if best_child is None:
            dead_ends.append(parent.name)
        else:
            combined_score = (parent.score + best_child.score) / 2
            viable_trajectories.append((combined_score, parent.name, parent, best_child))

    if dead_ends:
        print(f"\n{Colors.RED}✖ Dead end — every depth-2 refinement of these branches exceeds budget: "
              f"{', '.join(dead_ends)}{Colors.RESET}")
        print(f"{Colors.YELLOW}↳ Backtracking to the next-best surviving beam...{Colors.RESET}")

    if not viable_trajectories:
        print(f"\n{Colors.RED}✖ No viable trajectory found within the budget cap.{Colors.RESET}\n")
    else:
        viable_trajectories.sort(key=lambda t: t[0], reverse=True)
        best_score, best_parent_name, best_parent, best_child = viable_trajectories[0]
        print(f"\n{Colors.GREEN}★ Optimal trajectory after backtracking: "
              f"{best_parent_name} -> {best_child.name} (Combined Score: {best_score:.2f}){Colors.RESET}\n")

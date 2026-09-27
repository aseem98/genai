"""
=============================================================================
23_langgraph_subgraphs_and_hierarchical_teams.py: LangGraph Subgraphs & Teams
=============================================================================
Senior GenAI Engineer Context:
When building enterprise agents, putting 40 nodes into one monolithic graph
creates unmaintainable spaghetti logic.

The **LangGraph Subgraph Pattern**:
1. Encapsulation: A compiled graph can be added as a Node inside a parent graph!
2. Hierarchical Teams:
   - Root Supervisor evaluates user request and routes to specialized team subgraphs.
   - `ResearchTeam` Subgraph: (SearchNode -> SummarizerNode)
   - `CodingTeam` Subgraph: (CodeWriterNode -> SecurityLinterNode)
3. State Isolation: Subgraphs maintain internal state schemas and return only
   their synthesized outputs to the parent graph.

Run directly:
  python3 23_langgraph_subgraphs_and_hierarchical_teams.py
=============================================================================
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


# =====================================================================
# 1. SUBGRAPH A: RESEARCH TEAM
# =====================================================================
class ResearchTeamSubgraph:
    """Independent 2-node graph: Searcher -> Summarizer."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def invoke(self, topic: str) -> str:
        print(f"    {Colors.BLUE}▶ [Research Subgraph] Node 1 (Searcher): Querying papers on {topic}...{Colors.RESET}")
        # Searcher: a real retrieval index is out of scope for this module
        # (see module 26 for a real BM25 + dense hybrid retriever) — these
        # notes stand in for what that retriever would have returned.
        raw_notes = "Found 3 benchmarks showing Vector Search + BM25 outperforms dense-only search by 18%."

        print(f"    {Colors.BLUE}▶ [Research Subgraph] Node 2 (Summarizer): Distilling key findings...{Colors.RESET}")
        summary = self.llm.chat(
            [{"role": "user", "content": f"In one sentence, summarize this research finding on '{topic}': {raw_notes}"}],
            simulated_response=f"{raw_notes} (summarized)",
        )
        return f"Research Synthesis: {summary}"


# =====================================================================
# 2. SUBGRAPH B: CODING TEAM
# =====================================================================
class CodingTeamSubgraph:
    """Independent 2-node graph: Coder -> Security Linter."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def invoke(self, spec: str) -> str:
        print(f"    {Colors.YELLOW}▶ [Coding Subgraph] Node 1 (Coder): Writing Python implementation...{Colors.RESET}")
        code = self.llm.chat(
            [{"role": "user", "content": f"Write a minimal Python class stub (no implementation body needed) for: {spec}"}],
            simulated_response="class HybridSearcher:\n    def search(self): pass",
        )
        print(f"    {Colors.YELLOW}▶ [Coding Subgraph] Node 2 (Linter): Running AST & bandit security scan...{Colors.RESET}")
        # Linter: deliberately NOT an LLM call — a real static-analysis tool
        # (AST parsing, bandit) is deterministic and shouldn't be delegated
        # to a model; that's the actual "specialized subgraph" lesson here.
        return f"Verified Code Artifact:\n{code}\n# Linter: 0 vulnerabilities found."


# =====================================================================
# 3. PARENT SUPERVISOR GRAPH
# =====================================================================
class EnterpriseSupervisorGraph:
    """Parent Graph that orchestrates subgraphs as modular node units."""

    def __init__(self, llm: LLMClient):
        self.research_team = ResearchTeamSubgraph(llm)
        self.coding_team = CodingTeamSubgraph(llm)

    def invoke(self, user_request: str) -> Dict[str, Any]:
        print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
        print(f"ROOT SUPERVISOR RECEIVED: {user_request}")
        print(f"===================================================={Colors.RESET}\n")

        print(f"{Colors.CYAN}[Supervisor Router Node]: Decomposing request into Research + Coding phases...{Colors.RESET}\n")

        # Phase 1: Call Research Subgraph
        print(f"{Colors.BOLD}1. Delegating to Research Team Subgraph:{Colors.RESET}")
        research_output = self.research_team.invoke("Hybrid Search RAG")
        print(f"  {Colors.GREEN}✔ Subgraph Returned:{Colors.RESET} {research_output}\n")

        # Phase 2: Call Coding Subgraph
        print(f"{Colors.BOLD}2. Delegating to Coding Team Subgraph:{Colors.RESET}")
        coding_output = self.coding_team.invoke(research_output)
        print(f"  {Colors.GREEN}✔ Subgraph Returned:{Colors.RESET} {coding_output}\n")

        return {
            "research": research_output,
            "code": coding_output,
            "status": "COMPLETED"
        }


if __name__ == "__main__":
    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}")

    supervisor_app = EnterpriseSupervisorGraph(llm)
    goal = "Research hybrid search best practices and build a secure Python prototype."
    result = supervisor_app.invoke(goal)

    print(f"{Colors.BOLD}{Colors.GREEN}★ All Subgraphs Executed Successfully within Parent Graph.{Colors.RESET}")

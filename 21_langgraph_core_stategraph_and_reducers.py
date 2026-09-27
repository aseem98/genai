"""
=============================================================================
21_langgraph_core_stategraph_and_reducers.py: LangGraph StateGraph & Reducers
=============================================================================
Senior GenAI Engineer Context:
Why did LangGraph replace linear agent loops across the industry?
1. Cyclical graphs: Real agents need cycles (draft -> review -> fix -> review).
2. Centralized Typed State: All nodes read from and write to a shared schema.
3. State Reducers (e.g. `add_messages`): Instead of nodes overwriting the entire
   state, reducers specify how state deltas are merged (e.g. appending messages,
   summing counters, or overriding specific keys).

This module implements the exact LangGraph core architecture from scratch:
- `StateGraph(StateSchema)`
- `START` and `END` sentinels
- Reducer semantics (`add_messages` appending list reducer)
- `add_node()`, `add_edge()`, `add_conditional_edges()`
- `compile()` -> Runnable Graph

Run directly:
  python3 21_langgraph_core_stategraph_and_reducers.py
=============================================================================
"""

import copy
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

# Graph Sentinels
START = "__START__"
END = "__END__"


# =====================================================================
# 1. STATE REDUCER FUNCTIONS (THE LANGGRAPH REDUCER PATTERN)
# =====================================================================
def add_messages(existing: List[Dict[str, str]], new_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Reducer that appends new messages rather than overwriting the conversation history."""
    return existing + new_messages


@dataclass
class AgentState:
    query: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    review_score: int = 0
    iteration: int = 0


# =====================================================================
# 2. STATEGRAPH COMPILER ENGINE
# =====================================================================
class StateGraph:
    """Replicates the LangGraph StateGraph engine in pure Python."""

    def __init__(self, state_schema=AgentState):
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[Any], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable[[Any], str]] = {}
        self.reducers: Dict[str, Callable] = {"messages": add_messages}

    def add_node(self, name: str, action: Callable[[Any], Dict[str, Any]]):
        self.nodes[name] = action

    def add_edge(self, start_key: str, end_key: str):
        self.edges[start_key] = end_key

    def add_conditional_edges(self, source: str, path: Callable[[Any], str]):
        self.conditional_edges[source] = path

    def compile(self) -> "CompiledGraph":
        assert START in self.edges, "Graph must have an edge originating from START."
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, builder: StateGraph):
        self.builder = builder

    def invoke(self, initial_state: AgentState) -> AgentState:
        state = copy.deepcopy(initial_state)
        current_node = self.builder.edges[START]
        step = 0

        print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
        print(f"▶ EXECUTING LANGGRAPH WORKFLOW: Starting at [{current_node}]")
        print(f"===================================================={Colors.RESET}\n")

        while current_node != END and step < 10:
            step += 1
            print(f"{Colors.CYAN}{Colors.BOLD}Step {step}: Node [{current_node}]{Colors.RESET}")
            node_fn = self.builder.nodes[current_node]

            # Execute node logic returning state delta
            delta = node_fn(state)

            # Apply state delta using reducers if specified
            for key, val in delta.items():
                if key in self.builder.reducers and hasattr(state, key):
                    current_val = getattr(state, key)
                    reducer_fn = self.builder.reducers[key]
                    setattr(state, key, reducer_fn(current_val, val))
                else:
                    setattr(state, key, val)

            # Generic checkpoint print: any state schema works here, not just
            # this module's own AgentState — real LangGraph checkpoints the
            # whole state without assuming specific field names either.
            checkpoint_fields = ", ".join(f"{k}={v!r}" for k, v in vars(state).items() if not isinstance(v, (list, dict)) or len(str(v)) < 60)
            print(f"  {Colors.DIM}State Checkpoint: {checkpoint_fields}{Colors.RESET}")

            # Route to next node via conditional or direct edge
            if current_node in self.builder.conditional_edges:
                next_node = self.builder.conditional_edges[current_node](state)
                print(f"  {Colors.YELLOW}↳ Conditional Edge: [{current_node}] -> [{next_node}]{Colors.RESET}\n")
            elif current_node in self.builder.edges:
                next_node = self.builder.edges[current_node]
                print(f"  {Colors.BLUE}↳ Direct Edge: [{current_node}] -> [{next_node}]{Colors.RESET}\n")
            else:
                next_node = END

            current_node = next_node

        print(f"{Colors.GREEN}✔ Graph Execution Reached END sentinel.{Colors.RESET}\n")
        return state


# =====================================================================
# 3. NODE IMPLEMENTATIONS (GENERATOR & CRITIC) — each node is a real LLM call
# =====================================================================
llm = LLMClient()

# Node functions are plain closures over `llm` here, mirroring how a real
# LangGraph node closes over injected dependencies (an LLM client, a retriever).
DRAFT_SIMULATIONS = [
    "Draft 1: Agents use tools to do things.",
    "Draft 2: Agents combine LLM reasoning with cyclical graphs, tool execution, and durable checkpoints.",
]


def draft_node(state: AgentState) -> Dict[str, Any]:
    it = state.iteration + 1
    prior_feedback = f"\nPrevious reviewer feedback: score {state.review_score}/10. Improve on it." if it > 1 else ""
    prompt = f"Write a one-sentence explanation of '{state.query}' for iteration {it}.{prior_feedback}"
    content = llm.chat(
        [{"role": "user", "content": prompt}],
        simulated_response=DRAFT_SIMULATIONS[min(it, len(DRAFT_SIMULATIONS)) - 1],
    )
    return {
        "messages": [{"role": "assistant", "content": content}],  # list delta; the reducer appends it!
        "iteration": it,
    }


def review_node(state: AgentState) -> Dict[str, Any]:
    latest_msg = state.messages[-1]["content"]
    review_prompt = (
        f"Rate this draft from 0-10 on technical depth. Reply with ONLY the integer.\n"
        f"Draft: {latest_msg}"
    )
    raw_score = llm.chat(
        [{"role": "user", "content": review_prompt}],
        simulated_response="10" if "durable checkpoints" in latest_msg else "4",
    )
    digits = "".join(c for c in raw_score if c.isdigit())
    score = min(int(digits), 10) if digits else 0
    return {"review_score": score}


def should_continue(state: AgentState) -> str:
    """Conditional Edge function."""
    if state.review_score >= 8 or state.iteration >= 2:
        return END
    return "drafter"


# =====================================================================
# 4. RUNNABLE TESTS & GRAPH COMPOSITION
# =====================================================================
if __name__ == "__main__":
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}")
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("drafter", draft_node)
    workflow.add_node("reviewer", review_node)

    # Connect Edges
    workflow.add_edge(START, "drafter")
    workflow.add_edge("drafter", "reviewer")
    workflow.add_conditional_edges("reviewer", should_continue)

    # Compile Graph
    app = workflow.compile()

    init = AgentState(query="Explain LangGraph Architecture")
    final_state = app.invoke(init)

    print(f"{Colors.BOLD}{Colors.GREEN}Final Conversation History Accumulated by Reducer ({len(final_state.messages)} msgs):{Colors.RESET}")
    for m in final_state.messages:
        print(f"  • {m['role']}: {m['content']}")

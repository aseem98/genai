"""
=============================================================================
22_langgraph_human_in_the_loop_and_checkpoints.py: LangGraph HITL & MemorySaver
=============================================================================
Senior GenAI Engineer Context:
How does LangGraph implement Human-in-the-Loop without hacky `while` loops?

LangGraph HITL Architecture:
1. `checkpointer = MemorySaver()`: Records every state transition per `thread_id`.
2. Breakpoints: `interrupt_before=["tool_execution_node"]` halts the graph
   just before a sensitive node executes.
3. State Inspection: The host application calls `graph.get_state(config)` to render
   the pending action in a UI modal.
4. State Mutation: The operator calls `graph.update_state(config, {"approved": True})`.
5. Seamless Resumption: Calling `graph.invoke(None, config)` resumes execution
   from the exact interrupted checkpoint!

Run directly:
  python3 22_langgraph_human_in_the_loop_and_checkpoints.py
=============================================================================
"""

import copy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


# =====================================================================
# 1. STATE & MEMORYSAVER CHECKPOINTER
# =====================================================================
@dataclass
class DeploymentState:
    thread_id: str
    target_cluster: str
    image_tag: str
    risk_level: str
    approved: bool = False
    status: str = "PENDING"


class MemorySaver:
    """Stores snapshots of graph states keyed by thread_id and step."""

    def __init__(self):
        self.storage: Dict[str, List[DeploymentState]] = {}

    def put(self, thread_id: str, state: DeploymentState):
        if thread_id not in self.storage:
            self.storage[thread_id] = []
        self.storage[thread_id].append(copy.deepcopy(state))

    def get(self, thread_id: str) -> Optional[DeploymentState]:
        history = self.storage.get(thread_id, [])
        return copy.deepcopy(history[-1]) if history else None


# =====================================================================
# 2. LANGGRAPH WITH BREAKPOINTS & RESUMPTION
# =====================================================================
class CheckpointedGraph:
    def __init__(self, checkpointer: MemorySaver, llm: LLMClient, interrupt_before: List[str] = None):
        self.checkpointer = checkpointer
        self.llm = llm
        self.interrupt_before = interrupt_before or []

    def invoke(self, state_input: Optional[DeploymentState], config: Dict[str, Any]) -> DeploymentState:
        thread_id = config["configurable"]["thread_id"]

        # If state_input is None, we are RESUMING from the latest checkpoint!
        if state_input is None:
            state = self.checkpointer.get(thread_id)
            print(f"  {Colors.CYAN}▶ Resuming Thread '{thread_id}' from checkpoint...{Colors.RESET}")
        else:
            state = state_input

        # Node 1: Prepare & Risk Assess — a real classification call, whose
        # verdict is what actually decides whether the breakpoint below fires.
        if state.status == "PENDING":
            print(f"  [Node: assess_risk] Evaluating deployment image {state.image_tag} to {state.target_cluster}...")
            risk_prompt = (
                f"Classify the risk of deploying image '{state.image_tag}' to cluster "
                f"'{state.target_cluster}'. Reply with EXACTLY one word: LOW_STAKES or HIGH_STAKES. "
                f"Treat any 'prod' cluster name as HIGH_STAKES."
            )
            raw_risk = self.llm.chat(
                [{"role": "user", "content": risk_prompt}],
                simulated_response="HIGH_STAKES",
            )
            state.risk_level = "HIGH_STAKES" if "HIGH" in raw_risk.upper() else "LOW_STAKES"
            state.status = "ASSESSED"
            self.checkpointer.put(thread_id, state)
            print(f"  {Colors.YELLOW}↳ Risk classified as: {state.risk_level}{Colors.RESET}")

        # Breakpoint Check: interrupt_before
        if "deploy_node" in self.interrupt_before and not state.approved:
            print(f"\n{Colors.RED}{Colors.BOLD}🚨 [LANGGRAPH BREAKPOINT HIT]: Interrupted before node 'deploy_node'.{Colors.RESET}")
            print(f"   Execution suspended. State persisted to MemorySaver.\n")
            return state

        # Node 2: Deploy
        print(f"  {Colors.GREEN}[Node: deploy_node] Executing Kubernetes deployment rollout...{Colors.RESET}")
        state.status = "DEPLOYED_SUCCESSFULLY"
        self.checkpointer.put(thread_id, state)
        return state

    def update_state(self, config: Dict[str, Any], updates: Dict[str, Any]):
        """Mutates the checkpointed state for this thread (e.g. human approval)."""
        thread_id = config["configurable"]["thread_id"]
        state = self.checkpointer.get(thread_id)
        for k, v in updates.items():
            setattr(state, k, v)
        self.checkpointer.put(thread_id, state)
        print(f"  {Colors.YELLOW}✔ [update_state]: Updated thread '{thread_id}' with: {updates}{Colors.RESET}")


# =====================================================================
# 3. RUNNABLE TESTS & INTERACTION
# =====================================================================
if __name__ == "__main__":
    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    checkpointer = MemorySaver()
    app = CheckpointedGraph(checkpointer, llm, interrupt_before=["deploy_node"])

    thread_config = {"configurable": {"thread_id": "deploy-session-42"}}

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("1. INITIAL GRAPH INVOCATION (WILL HIT BREAKPOINT)")
    print(f"===================================================={Colors.RESET}\n")

    initial_state = DeploymentState(
        thread_id="deploy-session-42",
        target_cluster="prod-eu-west-1",
        image_tag="payments-service:v3.1.0",
        risk_level="UNKNOWN"
    )

    suspended_state = app.invoke(initial_state, config=thread_config)
    print(f"Current State after pause: Status='{suspended_state.status}', Approved={suspended_state.approved}\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("2. HUMAN OPERATOR INSPECTION & APPROVAL MUTATION")
    print(f"===================================================={Colors.RESET}\n")

    # Operator approves via UI
    app.update_state(thread_config, {"approved": True})

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("3. RESUMING GRAPH EXECUTION (INVOKE WITH NONE)")
    print(f"===================================================={Colors.RESET}\n")

    # Resuming with None tells LangGraph to pull checkpoint and continue!
    final_state = app.invoke(None, config=thread_config)
    print(f"\n{Colors.BOLD}{Colors.GREEN}Final Output State:{Colors.RESET}")
    print(f"  • Cluster: {final_state.target_cluster}")
    print(f"  • Status:  {final_state.status}")
    print(f"  • Approved: {final_state.approved}\n")

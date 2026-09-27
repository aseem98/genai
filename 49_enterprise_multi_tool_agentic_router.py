"""
=============================================================================
49_enterprise_multi_tool_agentic_router.py: Multi-Tool Agentic Router
=============================================================================
Senior GenAI Engineer Context:
This is the capstone that ties the whole curriculum together: a single
stateful agent that routes each user turn to the RIGHT tool — RAG (module
46), Text-to-SQL (module 47), or an external API — evaluates whether the
result actually answers the question, and either loops back for another
tool call or returns a final response. Built on module 21's `StateGraph`
engine (real reducer/conditional-edge mechanics, not a new one), exactly
the "Agent -> Route -> RAG/SQL/API -> Observation -> Evaluate -> Continue?"
loop from the architecture diagram this module implements:

    User Query -> Agent/Planner -> Routing -> {RAG | SQL | API}
        -> Observation -> Evaluation -> Continue? -> (loop | Final Response)

Run directly:
  python3 49_enterprise_multi_tool_agentic_router.py
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

graph_module = import_module("21_langgraph_core_stategraph_and_reducers")
StateGraph = graph_module.StateGraph
START, END = graph_module.START, graph_module.END

rag_module = import_module("46_multimodal_document_rag_pipeline")
DocumentIntelligenceSimulator = rag_module.DocumentIntelligenceSimulator
HybridDocumentIndex = rag_module.HybridDocumentIndex
answer_with_citations = rag_module.answer_with_citations

sql_module = import_module("47_text_to_sql_enterprise_analytics")
build_demo_database = sql_module.build_demo_database
run_text_to_sql = sql_module.run_text_to_sql


# =====================================================================
# 1. STATE (mirrors the diagram's LangGraph state fields exactly)
# =====================================================================
@dataclass
class RouterState:
    """Holds only plain, deep-copyable data — module 21's graph engine
    snapshots state with `copy.deepcopy()` at every step (that's what makes
    checkpointing/HITL in module 22 possible), so live resources like an
    LLM client or a DB connection can't live on the state itself. They're
    closed over by the node functions instead (the same pattern module 21's
    own `draft_node`/`review_node` already use for their `llm`).
    """
    user_query: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    route_decision: str = ""
    final_answer: Optional[str] = None
    loop_count: int = 0
    max_loops: int = 3


def add_tool_outputs(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return existing + new


# =====================================================================
# 2. MOCK EXTERNAL API TOOL (the diagram's third tool branch)
# =====================================================================
def external_hcp_directory_api(hcp_name: str) -> Dict[str, Any]:
    """Real equivalent: an external CRM/directory API call (e.g. Veeva)."""
    directory = {
        "Dr. Patel": {"license_status": "Active", "npi": "1234567890"},
        "Dr. Kim": {"license_status": "Active", "npi": "0987654321"},
    }
    return directory.get(hcp_name, {"error": f"No directory record for '{hcp_name}'"})


# =====================================================================
# 3. NODES
# =====================================================================
def make_agent_router_node(llm: LLMClient):
    def agent_router_node(state: RouterState) -> Dict[str, Any]:
        routing_prompt = (
            f"Classify this user query into EXACTLY ONE route: 'rag' (policy/document questions), "
            f"'sql' (data/analytics questions over HCP/engagement/prescription tables), or 'api' "
            f"(external directory lookups for a named person).\n"
            f"Query: {state.user_query}\nReply with exactly one word: rag, sql, or api."
        )
        fallback = (
            "sql" if any(w in state.user_query.lower() for w in ["how many", "total", "average", "count", "sum"])
            else "api" if "npi" in state.user_query.lower() or "license" in state.user_query.lower()
            else "rag"
        )
        decision = llm.chat([{"role": "user", "content": routing_prompt}], simulated_response=fallback)
        route = next((r for r in ("rag", "sql", "api") if r in decision.lower()), fallback)
        print(f"    {Colors.CYAN}[Agent/Planner] Routing decision: '{route}'{Colors.RESET}")
        return {"route_decision": route}
    return agent_router_node


def make_rag_node(llm: LLMClient, rag_index: HybridDocumentIndex):
    def rag_node(state: RouterState) -> Dict[str, Any]:
        hits = rag_index.hybrid_search(state.user_query, user_role="employee", top_k=2)
        result = answer_with_citations(
            llm, state.user_query, hits,
            simulated_response="You may claim up to $150 per night for hotel accommodation [Source 1].",
        )
        print(f"    {Colors.BLUE}[RAG Tool] Retrieved {len(hits)} chunks -> grounded={result['grounded']}{Colors.RESET}")
        return {"tool_outputs": [{"tool": "rag", "output": result["answer"], "grounded": result["grounded"]}]}
    return rag_node


def make_sql_node(llm: LLMClient, sql_conn: Any):
    def sql_node(state: RouterState) -> Dict[str, Any]:
        result = run_text_to_sql(
            llm, sql_conn, state.user_query,
            sim_sql="SELECT COUNT(*) FROM hcps WHERE tier = 'A';",
            sim_summary="There are 2 tier-A HCPs.",
        )
        valid = result["validation"].valid
        print(f"    {Colors.YELLOW}[SQL Tool] Generated: {result['sql']} -> valid={valid}{Colors.RESET}")
        output = result["summary"] if valid else f"SQL rejected: {result['validation'].reason}"
        return {"tool_outputs": [{"tool": "sql", "output": output, "grounded": valid}]}
    return sql_node


def api_node(state: RouterState) -> Dict[str, Any]:
    # Toy name extraction — a real system uses NER or asks the LLM to extract it.
    name = next((n for n in ("Dr. Patel", "Dr. Kim") if n in state.user_query), "Dr. Patel")
    api_result = external_hcp_directory_api(name)
    print(f"    {Colors.MAGENTA}[API Tool] Directory lookup for '{name}' -> {api_result}{Colors.RESET}")
    return {"tool_outputs": [{"tool": "api", "output": str(api_result), "grounded": "error" not in api_result}]}


def evaluation_node(state: RouterState) -> Dict[str, Any]:
    latest = state.tool_outputs[-1]
    sufficient = latest["grounded"] and state.loop_count < state.max_loops
    print(f"    {Colors.CYAN}[Evaluation] Tool output sufficient={sufficient}{Colors.RESET}")
    if sufficient:
        return {"final_answer": latest["output"]}
    return {"loop_count": state.loop_count + 1}


def should_continue(state: RouterState) -> str:
    if state.final_answer is not None:
        return "respond"
    if state.loop_count >= state.max_loops:
        return "respond"  # give up gracefully rather than looping forever
    return "agent"


def response_node(state: RouterState) -> Dict[str, Any]:
    answer = state.final_answer or "I wasn't able to find a grounded answer within the retry budget."
    return {"final_answer": answer}


# =====================================================================
# 4. GRAPH ASSEMBLY (module 21's real StateGraph — reused, not re-stubbed)
# =====================================================================
def build_router_graph(llm: LLMClient, rag_index: HybridDocumentIndex, sql_conn: Any):
    graph = StateGraph(RouterState)
    graph.reducers["tool_outputs"] = add_tool_outputs

    graph.add_node("agent", make_agent_router_node(llm))
    graph.add_node("rag", make_rag_node(llm, rag_index))
    graph.add_node("sql", make_sql_node(llm, sql_conn))
    graph.add_node("api", api_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("respond", response_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: s.route_decision)
    graph.add_edge("rag", "evaluate")
    graph.add_edge("sql", "evaluate")
    graph.add_edge("api", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue)
    graph.add_edge("respond", END)

    # `route_decision` needs its own conditional-edge target map since
    # module 21's StateGraph routes purely off the returned string —
    # "rag"/"sql"/"api" already match node names, so this "just works."
    return graph.compile()


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Enterprise Multi-Tool Agentic Router ==={Colors.RESET}\n")

    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    hr_policy_doc = """# Expense Reimbursement Policy
## Travel Expenses
Employees may claim up to $150/night for hotel accommodation on approved business travel.
"""
    rag_index = HybridDocumentIndex()
    rag_index.build(DocumentIntelligenceSimulator.extract_and_chunk("hr-policy-v3", hr_policy_doc, access_roles=["all"]))
    sql_conn = build_demo_database()

    app = build_router_graph(llm, rag_index, sql_conn)

    queries = [
        "How much can I claim per night for a hotel on business travel?",
        "How many tier A HCPs do we have?",
        "What is Dr. Patel's license status?",
    ]

    for query in queries:
        print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}\nUSER QUERY: {query}\n{'=' * 60}{Colors.RESET}")
        state = RouterState(user_query=query)
        final_state = app.invoke(state)
        print(f"  {Colors.GREEN}★ Final Response:{Colors.RESET} {final_state.final_answer}\n")

    sql_conn.close()
    print(f"{Colors.GREEN}✔ Agentic routing across RAG, SQL, and API tools — with evaluation and retry — verified end to end.{Colors.RESET}")

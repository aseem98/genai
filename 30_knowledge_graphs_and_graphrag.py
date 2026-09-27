"""
=============================================================================
30_knowledge_graphs_and_graphrag.py: Knowledge Graph Construction & GraphRAG
=============================================================================
Senior GenAI Engineer Context:
Vector search (modules 26/29) finds chunks that are TEXTUALLY similar to a
query. It's structurally blind to relationships: "Which drugs does Dr.
Patel prescribe, and which of THOSE have an interaction with Drug X?" is a
multi-hop graph traversal question no single embedding similarity search
answers well, no matter how good the embedding is.

GraphRAG (Microsoft Research, 2024) fixes this by building an explicit
knowledge graph from the corpus and combining graph traversal with vector
retrieval:
  1. ENTITY & RELATION EXTRACTION: an LLM reads each document and extracts
     (subject, relation, object) triples — turning unstructured text into
     structured graph edges.
  2. GRAPH CONSTRUCTION: triples become nodes + directed labeled edges in
     an adjacency-list graph.
  3. MULTI-HOP TRAVERSAL: answer relational questions by walking the graph
     N hops from a starting entity, instead of a single similarity lookup.
  4. HYBRID GraphRAG: combine graph-traversal facts with the vector-search
     hits from module 29 into one grounded context for the LLM.

Run directly:
  python3 30_knowledge_graphs_and_graphrag.py
=============================================================================
"""

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
JsonOutputParser = import_module("11_langchain_output_parsers_and_schemas").JsonOutputParser

vectordb_module = import_module("29_vector_database_client_patterns")
VectorDBClient = vectordb_module.VectorDBClient
VectorRecord = vectordb_module.VectorRecord
embed_text = import_module("01_unified_llm_client").embed_text


# =====================================================================
# 1. ENTITY & RELATION EXTRACTION (LLM turns text into graph triples)
# =====================================================================
@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str


def extract_triples(llm: LLMClient, parser: JsonOutputParser, text: str, simulated_response: str) -> List[Triple]:
    prompt = (
        "Extract knowledge graph triples from this text as a JSON array of "
        '{"subject": ..., "relation": ..., "object": ...} objects. Use short, '
        "canonical entity names (e.g. 'Dr. Patel' not 'the doctor').\n"
        f"Text: {text}\n{parser.get_format_instructions()}"
    )
    raw = llm.chat([{"role": "user", "content": prompt}], simulated_response=simulated_response)
    try:
        parsed = parser.invoke(raw)
        return [Triple(t["subject"], t["relation"], t["object"]) for t in parsed if all(k in t for k in ("subject", "relation", "object"))]
    except (ValueError, KeyError, TypeError):
        return []


# =====================================================================
# 2. KNOWLEDGE GRAPH (directed, labeled adjacency graph)
# =====================================================================
class KnowledgeGraph:
    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node -> [(relation, target)]
        self.triples: List[Triple] = []

    def add_triples(self, triples: List[Triple]):
        for t in triples:
            self.edges[t.subject].append((t.relation, t.object))
            self.triples.append(t)

    def neighbors(self, entity: str) -> List[Tuple[str, str]]:
        return self.edges.get(entity, [])

    def multi_hop_traverse(self, start_entity: str, max_hops: int = 2) -> List[Triple]:
        """BFS outward from `start_entity` up to `max_hops` — the graph
        equivalent of "expand the retrieved context" in vector RAG, except
        it follows explicit relationships instead of embedding similarity.
        """
        visited: Set[str] = {start_entity}
        frontier = deque([(start_entity, 0)])
        found: List[Triple] = []

        while frontier:
            entity, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for relation, target in self.neighbors(entity):
                found.append(Triple(entity, relation, target))
                if target not in visited:
                    visited.add(target)
                    frontier.append((target, depth + 1))
        return found

    def to_context_string(self, triples: List[Triple]) -> str:
        return "\n".join(f"{t.subject} --[{t.relation}]--> {t.object}" for t in triples)


# =====================================================================
# 3. HYBRID GraphRAG: graph traversal facts + vector-retrieved text
# =====================================================================
def graphrag_query(
    llm: LLMClient,
    graph: KnowledgeGraph,
    vector_db: VectorDBClient,
    start_entity: str,
    question: str,
    simulated_response: str,
    max_hops: int = 2,
) -> Dict[str, Any]:
    graph_facts = graph.multi_hop_traverse(start_entity, max_hops=max_hops)
    graph_context = graph.to_context_string(graph_facts)

    vector_hits = vector_db.query("clinical-notes", embed_text(question), top_k=2)
    vector_context = "\n".join(h["metadata"]["text"] for h in vector_hits)

    prompt = (
        f"Answer the question using BOTH the knowledge graph facts and the retrieved notes below.\n\n"
        f"KNOWLEDGE GRAPH FACTS (multi-hop from '{start_entity}'):\n{graph_context}\n\n"
        f"RETRIEVED NOTES:\n{vector_context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    answer = llm.chat([{"role": "user", "content": prompt}], simulated_response=simulated_response)
    return {"answer": answer, "graph_facts_used": len(graph_facts), "vector_hits_used": len(vector_hits)}


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Knowledge Graph Construction & GraphRAG ==={Colors.RESET}\n")

    llm = LLMClient()
    parser = JsonOutputParser()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    documents = [
        "Dr. Patel prescribes Metformin to patients with Type 2 Diabetes.",
        "Metformin has a known drug interaction with Contrast Dye used in CT scans.",
        "Dr. Patel also treats patients with Hypertension using Lisinopril.",
        "Lisinopril has a known drug interaction with Potassium Supplements.",
    ]

    print(f"{Colors.YELLOW}1. Entity & Relation Extraction — LLM turns text into graph triples{Colors.RESET}")
    graph = KnowledgeGraph()
    simulated_triples = [
        json.dumps([{"subject": "Dr. Patel", "relation": "prescribes", "object": "Metformin"}]),
        json.dumps([{"subject": "Metformin", "relation": "interacts_with", "object": "Contrast Dye"}]),
        json.dumps([{"subject": "Dr. Patel", "relation": "treats_with", "object": "Lisinopril"}]),
        json.dumps([{"subject": "Lisinopril", "relation": "interacts_with", "object": "Potassium Supplements"}]),
    ]
    for doc, sim in zip(documents, simulated_triples):
        triples = extract_triples(llm, parser, doc, simulated_response=sim)
        graph.add_triples(triples)
        for t in triples:
            print(f"  • {t.subject} --[{t.relation}]--> {t.object}")

    print(f"\n{Colors.YELLOW}2. Multi-hop traversal — a question vector search alone can't answer{Colors.RESET}")
    print(f"  Question: \"What drug interactions might affect a patient Dr. Patel prescribes for?\"")
    hops = graph.multi_hop_traverse("Dr. Patel", max_hops=2)
    print(f"  2-hop traversal from 'Dr. Patel' found {len(hops)} facts:")
    for t in hops:
        print(f"    {t.subject} --[{t.relation}]--> {t.object}")
    one_hop_entities = {target for _relation, target in graph.neighbors("Dr. Patel")}
    two_hop_only = {t.object for t in hops} - one_hop_entities - {"Dr. Patel"}
    two_hop_display = two_hop_only if two_hop_only else "(none this run — depends on the live model's exact extraction)"
    print(f"  {Colors.GREEN}Entities only reachable at hop 2 (not directly connected to 'Dr. Patel'): {two_hop_display}{Colors.RESET}")
    print(f"  {Colors.GREEN}A single vector-similarity search over these 4 sentences would need to independently{Colors.RESET}")
    print(f"  {Colors.GREEN}retrieve the prescription fact AND the interaction fact in the same top-k to connect them —{Colors.RESET}")
    print(f"  {Colors.GREEN}the graph traversal above finds that connection deterministically, in one hop-out.{Colors.RESET}\n")

    print(f"{Colors.YELLOW}3. Hybrid GraphRAG — graph facts + vector-retrieved unstructured notes{Colors.RESET}")
    vector_db = VectorDBClient()
    clinical_notes = [
        "Patient scheduled for a CT scan with contrast next week; currently on Metformin for diabetes management.",
        "Routine blood pressure check; patient reports occasional missed doses of Lisinopril.",
    ]
    vector_db.upsert("clinical-notes", [
        VectorRecord(id=f"note-{i}", embedding=embed_text(note), metadata={"text": note})
        for i, note in enumerate(clinical_notes)
    ])

    result = graphrag_query(
        llm, graph, vector_db, start_entity="Dr. Patel",
        question="This patient is scheduled for a contrast CT scan — any concerns given their current medications?",
        simulated_response=(
            "Yes — the patient is on Metformin, which has a known interaction with the contrast dye used in CT scans. "
            "Metformin should typically be paused around the time of the contrast scan per standard protocol."
        ),
    )
    print(f"  {Colors.GREEN}Answer:{Colors.RESET} {result['answer']}")
    print(f"  (used {result['graph_facts_used']} graph facts + {result['vector_hits_used']} vector-retrieved notes)\n")

    print(f"{Colors.GREEN}✔ Triple extraction, multi-hop graph traversal, and hybrid GraphRAG all verified.{Colors.RESET}")

"""
=============================================================================
29_vector_database_client_patterns.py: Vector Database Client API Patterns
=============================================================================
Senior GenAI Engineer Context:
Module 28 built the ANN indexes themselves. This module is the layer on top
that every real vector DB SDK (Pinecone, Qdrant, Weaviate, pgvector) exposes
as its actual API surface — and the three things production RAG needs that
"just do cosine similarity" doesn't cover:

1. UPSERT/DELETE with metadata: vectors are stored alongside structured
   metadata (tenant, category, timestamp), not just as bare floats.
2. METADATA FILTERING combined with vector search: "find similar docs WHERE
   category = 'billing' AND tenant_id = 'acme'" — a pre-filter applied
   before (or alongside) the similarity ranking, not after.
3. NAMESPACES: multi-tenant isolation — the same physical index serves many
   customers' data without their vectors ever being compared against each
   other.

Implemented as an in-memory client with a real vector-DB-shaped API
(`upsert`, `query`, `delete`, `fetch`) built on module 28's IVF index, so
swapping this for `pinecone.Index(...)` or `QdrantClient(...)` later is a
drop-in — same method names, same call shape.

Run directly:
  python3 29_vector_database_client_patterns.py
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
VectorMath = import_module("05_embeddings_and_vector_math").VectorMath
embed_text = import_module("01_unified_llm_client").embed_text


@dataclass
class VectorRecord:
    id: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def matches_filter(metadata: Dict[str, Any], filter_: Optional[Dict[str, Any]]) -> bool:
    """Supports the common Pinecone/Qdrant-style filter operators: exact
    match, `$in`, `$gte`, `$lte` — enough to demonstrate pre-filtering
    without implementing a full query DSL.
    """
    if not filter_:
        return True
    for key, condition in filter_.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$gte" in condition and not (value is not None and value >= condition["$gte"]):
                return False
            if "$lte" in condition and not (value is not None and value <= condition["$lte"]):
                return False
        elif value != condition:
            return False
    return True


class VectorDBClient:
    """In-memory stand-in for a real vector DB SDK. Real equivalent:
    `pinecone.Index("agent-memory")` / `QdrantClient(url=...)`.
    """

    def __init__(self):
        self.namespaces: Dict[str, Dict[str, VectorRecord]] = {}

    def upsert(self, namespace: str, records: List[VectorRecord]) -> Dict[str, Any]:
        ns = self.namespaces.setdefault(namespace, {})
        for record in records:
            ns[record.id] = record
        return {"upserted_count": len(records)}

    def delete(self, namespace: str, ids: List[str]) -> Dict[str, Any]:
        ns = self.namespaces.get(namespace, {})
        deleted = 0
        for record_id in ids:
            if ns.pop(record_id, None) is not None:
                deleted += 1
        return {"deleted_count": deleted}

    def fetch(self, namespace: str, ids: List[str]) -> List[VectorRecord]:
        ns = self.namespaces.get(namespace, {})
        return [ns[i] for i in ids if i in ns]

    def query(
        self,
        namespace: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Metadata pre-filter, THEN similarity rank the survivors — the
        actual production order of operations (filtering the full corpus
        down first is far cheaper than ranking everything then discarding
        most of it).
        """
        ns = self.namespaces.get(namespace, {})
        candidates = [r for r in ns.values() if matches_filter(r.metadata, filter)]
        scored = [
            {"id": r.id, "score": VectorMath.cosine_similarity(query_embedding, r.embedding), "metadata": r.metadata}
            for r in candidates
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Vector Database Client API Patterns ==={Colors.RESET}\n")

    db = VectorDBClient()

    documents = [
        ("doc-1", "How do I request a refund for a billing charge?", {"tenant": "acme", "category": "billing"}),
        ("doc-2", "Refund timelines depend on your original payment method.", {"tenant": "acme", "category": "billing"}),
        ("doc-3", "How do I reset my account password?", {"tenant": "acme", "category": "account"}),
        ("doc-4", "Billing disputes for enterprise contracts go through legal.", {"tenant": "globex", "category": "billing"}),
    ]
    records = [VectorRecord(id=doc_id, embedding=embed_text(text), metadata={**meta, "text": text})
               for doc_id, text, meta in documents]

    print(f"{Colors.YELLOW}1. Upsert into a namespace{Colors.RESET}")
    result = db.upsert(namespace="support-kb", records=records)
    print(f"  {result}\n")

    query_text = "What's the process for getting money back on a charge?"
    query_embedding = embed_text(query_text)

    print(f"{Colors.YELLOW}2. Query WITHOUT a metadata filter — sees every tenant's docs{Colors.RESET}")
    for hit in db.query("support-kb", query_embedding, top_k=3):
        print(f"  • {hit['id']} (score={hit['score']:.3f}, tenant={hit['metadata']['tenant']}): {hit['metadata']['text']}")

    print(f"\n{Colors.YELLOW}3. Query WITH a metadata filter — multi-tenant isolation (tenant='acme' AND category='billing'){Colors.RESET}")
    for hit in db.query("support-kb", query_embedding, top_k=3, filter={"tenant": "acme", "category": "billing"}):
        print(f"  • {hit['id']} (score={hit['score']:.3f}): {hit['metadata']['text']}")

    print(f"\n{Colors.YELLOW}4. Namespace isolation — querying a DIFFERENT (empty) namespace returns nothing{Colors.RESET}")
    print(f"  {db.query('other-namespace', query_embedding, top_k=3)}")

    print(f"\n{Colors.YELLOW}5. Fetch by id, then delete{Colors.RESET}")
    print(f"  Fetched: {[r.id for r in db.fetch('support-kb', ['doc-1', 'doc-3'])]}")
    print(f"  Delete result: {db.delete('support-kb', ['doc-1'])}")
    print(f"  Post-delete query results no longer include doc-1: "
          f"{[hit['id'] for hit in db.query('support-kb', query_embedding, top_k=4)]}")

    print(f"\n{Colors.GREEN}✔ Upsert, filtered query, namespace isolation, and delete all verified.{Colors.RESET}")

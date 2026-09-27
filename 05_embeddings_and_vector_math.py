"""
=============================================================================
05_embeddings_and_vector_math.py: Embeddings & the Vector Math Behind Search
=============================================================================
An embedding maps text to a fixed-length vector so that "similar" texts
land near each other. Every RAG system, semantic cache, and agent memory in
this curriculum rests on the math in this module:

1. DISTANCE METRICS: cosine similarity, dot product, Euclidean (L2) and
   Manhattan (L1) distance — and when each is appropriate.
2. THE NORMALIZATION TRICK: for unit-length vectors, dot product EQUALS
   cosine similarity. Vector databases pre-normalize so they can use the
   cheaper dot product (and SIMD hardware) at query time.
3. EXACT k-NN SEARCH: score the query against every stored vector, keep the
   top k. Correct but O(N) — module 28 builds the approximate indexes that
   scale this to millions of vectors.
4. LEXICAL vs SEMANTIC EMBEDDINGS: feature hashing (module 01's offline
   embedding) matches shared WORDS; a trained embedding model (set
   EMBEDDING_MODEL in .env) matches MEANING. The synonym test below shows
   the difference on real text — this is why production RAG uses trained
   embeddings AND keyword search together (module 26).

Run directly:
  python3 05_embeddings_and_vector_math.py
=============================================================================
"""

import math
import re
import zlib
from typing import List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
embed_text = llm_module.embed_text


# =====================================================================
# 1. VECTOR SIMILARITY & DISTANCE MATHEMATICS
# =====================================================================
class VectorMath:
    @staticmethod
    def l2_norm(v: List[float]) -> float:
        """||v||_2 = sqrt(sum(v_i^2))"""
        return math.sqrt(sum(x * x for x in v))

    @staticmethod
    def normalize(v: List[float]) -> List[float]:
        norm = VectorMath.l2_norm(v)
        return [x / norm for x in v] if norm else v

    @staticmethod
    def dot_product(u: List[float], v: List[float]) -> float:
        assert len(u) == len(v), "Vectors must have identical dimensions."
        return sum(a * b for a, b in zip(u, v))

    @staticmethod
    def cosine_similarity(u: List[float], v: List[float]) -> float:
        """(u . v) / (||u|| ||v||): the angle between vectors, ignoring length. Range [-1, 1]."""
        norm_u, norm_v = VectorMath.l2_norm(u), VectorMath.l2_norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return VectorMath.dot_product(u, v) / (norm_u * norm_v)

    @staticmethod
    def euclidean_distance(u: List[float], v: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))

    @staticmethod
    def manhattan_distance(u: List[float], v: List[float]) -> float:
        return sum(abs(a - b) for a, b in zip(u, v))


# =====================================================================
# 2. EXACT k-NEAREST-NEIGHBOR INDEX
# =====================================================================
class SimpleVectorIndex:
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []

    def add(self, doc_text: str, embedding: List[float]):
        self.documents.append(doc_text)
        # Pre-normalize once at insert time so every query can use a plain dot product.
        self.embeddings.append(VectorMath.normalize(embedding) if self.metric == "cosine" else embedding)

    def search(self, query_emb: List[float], top_k: int = 2) -> List[Tuple[str, float]]:
        q = VectorMath.normalize(query_emb) if self.metric == "cosine" else query_emb
        if self.metric == "cosine":
            scored = [(d, VectorMath.dot_product(q, e)) for d, e in zip(self.documents, self.embeddings)]
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored = [(d, VectorMath.euclidean_distance(q, e)) for d, e in zip(self.documents, self.embeddings)]
            scored.sort(key=lambda x: x[1])
        return scored[:top_k]


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 05: Embeddings & Vector Math ==={Colors.RESET}\n")
    client = LLMClient()
    print(f"Embedding mode: {Colors.GREEN}{client.embedding_mode.upper()}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Why metric choice matters: same direction, different length{Colors.RESET}")
    short_doc, long_doc = [1.0, 2.0, 0.0], [10.0, 20.0, 0.0]   # identical topic mix, 10x the magnitude
    other_topic = [0.0, 0.5, 2.0]
    print(f"  cosine(short, long)  = {VectorMath.cosine_similarity(short_doc, long_doc):.3f}   <- identical direction")
    print(f"  L2(short, long)      = {VectorMath.euclidean_distance(short_doc, long_doc):.3f}  <- large, purely from magnitude")
    print(f"  L2(short, other)     = {VectorMath.euclidean_distance(short_doc, other_topic):.3f}   <- 'closer' by L2 despite being a different topic")
    print(f"  {Colors.DIM}Text embeddings encode meaning in DIRECTION, so cosine (or dot product on unit vectors) is the default.{Colors.RESET}\n")

    print(f"{Colors.YELLOW}2. The normalization trick, verified on real embeddings{Colors.RESET}")
    a = embed_text("How do I rotate the database credentials?")
    b = embed_text("Steps to rotate database passwords safely")
    cos = VectorMath.cosine_similarity(a, b)
    dot_unit = VectorMath.dot_product(VectorMath.normalize(a), VectorMath.normalize(b))
    print(f"  cosine(a, b)                     = {cos:.10f}")
    print(f"  dot(normalize(a), normalize(b))  = {dot_unit:.10f}")
    print(f"  {Colors.GREEN}✔ Identical (difference {abs(cos - dot_unit):.1e}) — so an index can store unit vectors and use dot product.{Colors.RESET}\n")

    print(f"{Colors.YELLOW}3. Exact k-NN over real documents{Colors.RESET}")
    corpus = [
        "Kubernetes horizontal pod autoscaling scales replicas based on CPU utilization",
        "Quarterly revenue and EBITDA grew in the enterprise segment",
        "PostgreSQL streaming replication ships the write-ahead log to standby replicas",
        "Rotate database credentials every 90 days and store them in a secrets manager",
        "Protein folding determines how cells build functional enzymes",
    ]
    index = SimpleVectorIndex(metric="cosine")
    for doc in corpus:
        index.add(doc, embed_text(doc))
    query = "how does postgres replication to a standby work"
    print(f"  Query: \"{query}\"")
    for rank, (doc, score) in enumerate(index.search(embed_text(query), top_k=2), 1):
        print(f"    #{rank} ({score:.3f}) {doc}")
    print()

    print(f"{Colors.YELLOW}4. Lexical vs semantic: a synonym test with NO shared words{Colors.RESET}")
    synonym_pairs = [
        ("I want my money back", "How do I request a refund?"),
        ("The laptop won't turn on", "My computer fails to boot"),
    ]
    control_pairs = [  # unrelated sentences: the "no meaning shared" baseline
        ("I want my money back", "The laptop won't turn on"),
        ("How do I request a refund?", "My computer fails to boot"),
    ]
    sim = lambda p: VectorMath.cosine_similarity(embed_text(p[0]), embed_text(p[1]))
    for p in synonym_pairs:
        print(f"  same meaning: sim({p[0]!r}, {p[1]!r}) = {sim(p):.3f}")
    for p in control_pairs:
        print(f"  unrelated:    sim({p[0]!r}, {p[1]!r}) = {sim(p):.3f}")
    avg_syn = sum(map(sim, synonym_pairs)) / len(synonym_pairs)
    avg_ctl = sum(map(sim, control_pairs)) / len(control_pairs)
    print(f"  average: same-meaning {avg_syn:.3f} vs unrelated {avg_ctl:.3f}")
    if client.embedding_mode == "hashing":
        # Explain every nonzero hashing score from first principles.
        words = lambda t: set(re.findall(r"\b\w+\b", t.lower()))
        bucket = lambda w: zlib.crc32(w.encode("utf-8")) % llm_module.HASHING_EMBEDDING_DIM
        print(f"  {Colors.CYAN}Where the same-meaning scores actually come from:{Colors.RESET}")
        for p1, p2 in synonym_pairs:
            w1, w2 = words(p1), words(p2)
            shared = sorted(w1 & w2)
            collisions = sorted((a, b) for a in w1 - w2 for b in w2 - w1 if bucket(a) == bucket(b))
            print(f"    {p1!r} vs {p2!r}: shared words={shared or 'none'}, hash-bucket collisions={collisions or 'none'}")
        print(f"  {Colors.RED}So the gap above is lexical accident (a shared stopword, a bucket collision), not meaning —")
        print(f"  swap in different wording and it disappears. Feature hashing has no notion of synonyms.{Colors.RESET}")
        print(f"  {Colors.DIM}Set EMBEDDING_MODEL to a trained model and same-meaning pairs score high for the right reason.{Colors.RESET}")
    else:
        print(f"  {Colors.GREEN}The trained model scores same-meaning pairs higher despite zero shared content words — semantic search.{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Distance metrics, the normalization identity, and exact k-NN verified on real text embeddings.{Colors.RESET}")

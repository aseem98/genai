"""
=============================================================================
26_advanced_rag_hybrid_and_reranking.py: Production RAG, BM25 & Rerankers
=============================================================================
Senior GenAI Engineer Context:
Why dense vector search alone fails in production:
1. Missing exact keywords (SKUs, UUIDs, error codes like HTTP 503).
2. Lost in the Middle syndrome: LLMs forget context placed in the center of long prompts.

Production Solution (all implemented below, in this order):
- Okapi BM25 Sparse Search (exact keyword matching).
- Dense Semantic Search (hashing-trick pseudo-embeddings + cosine similarity —
  a stand-in for a real embedding model, but a genuine dense vector search).
- Reciprocal Rank Fusion (RRF) to merge the two rankings into one shortlist.
- Cross-Encoder Re-ranking heuristic (bigram overlap) applied only to that
  shortlist, mirroring the real "cheap retrieve, expensive rerank" pattern.
- Lost-in-the-Middle context reordering of the final shortlist.

Run directly:
  python3 26_advanced_rag_hybrid_and_reranking.py
=============================================================================
"""

import math
import re
import zlib
from typing import List, Dict, Any, Tuple
from collections import Counter
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
vecmath_module = import_module("05_embeddings_and_vector_math")
VectorMath = vecmath_module.VectorMath


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str]):
        self.corpus = [re.findall(r"\b\w+\b", d.lower()) for d in documents]
        self.doc_lens = [len(c) for c in self.corpus]
        self.avg_dl = sum(self.doc_lens) / (len(documents) or 1)
        self.doc_freqs = [Counter(c) for c in self.corpus]

        df = {}
        for c in self.corpus:
            for term in set(c):
                df[term] = df.get(term, 0) + 1

        for term, freq in df.items():
            self.idf[term] = math.log((len(documents) - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str) -> List[Tuple[int, float]]:
        q_tokens = re.findall(r"\b\w+\b", query.lower())
        scores = []
        for idx in range(len(self.corpus)):
            score = 0.0
            doc_len = self.doc_lens[idx]
            freqs = self.doc_freqs[idx]
            for term in q_tokens:
                if term in freqs:
                    tf = freqs[term]
                    idf = self.idf.get(term, 0.0)
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl))
                    score += idf * (num / den)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# =====================================================================
# 2. DENSE SEMANTIC SEARCH (hashing-trick pseudo-embeddings)
# =====================================================================
def simulate_embedding(text: str, dim: int = 64) -> List[float]:
    """
    Stands in for a real embedding model: hashes each word into a fixed-size
    vector (the classic "hashing trick"). It's not semantically meaningful
    like a trained model, but it IS a genuine dense fixed-length vector, so
    cosine similarity over it behaves like real dense search (catches
    synonyms/word-order differences that exact BM25 term-matching misses),
    which is the property this demo needs.
    """
    vec = [0.0] * dim
    for word in re.findall(r"\b\w+\b", text.lower()):
        # zlib.crc32 (not the builtin hash()) so results are stable across
        # runs/processes — builtin string hashing is randomized per-process.
        vec[zlib.crc32(word.encode("utf-8")) % dim] += 1.0
    return VectorMath.normalize(vec)


def dense_search(docs: List[str], query: str) -> List[Tuple[int, float]]:
    doc_embeds = [simulate_embedding(d) for d in docs]
    q_embed = simulate_embedding(query)
    scores = [(i, VectorMath.cosine_similarity(q_embed, e)) for i, e in enumerate(doc_embeds)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# =====================================================================
# 3. RECIPROCAL RANK FUSION (merges sparse + dense rankings)
# =====================================================================
def reciprocal_rank_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """RRF(d) = sum over rankers of 1 / (k + rank(d)). Rank-based (not score-based)
    fusion sidesteps BM25 and cosine similarity living on incompatible scales."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_idx, _score) in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# =====================================================================
# 4. CROSS-ENCODER RERANKING (heuristic)
# =====================================================================
def cross_encoder_rerank(docs: List[str], query: str, candidate_idxs: List[int]) -> List[Tuple[int, float]]:
    """
    A real cross-encoder jointly attends over (query, doc) pairs instead of
    scoring them independently, so it's far more precise but too expensive to
    run over the whole corpus. Simulated here as an exact-phrase/bigram
    overlap score, applied only to the small shortlist RRF already produced
    — that "cheap retrieve, expensive rerank on a shortlist" division of
    labor is the actual production pattern being taught.
    """
    q_tokens = re.findall(r"\b\w+\b", query.lower())
    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    scored = []
    for idx in candidate_idxs:
        d_tokens = re.findall(r"\b\w+\b", docs[idx].lower())
        d_bigrams = set(zip(d_tokens, d_tokens[1:]))
        overlap = len(q_bigrams & d_bigrams)
        scored.append((idx, float(overlap)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# =====================================================================
# 5. LOST-IN-THE-MIDDLE CONTEXT REORDERING
# =====================================================================
def reorder_lost_in_the_middle(ranked_idxs: List[int]) -> List[int]:
    """
    LLMs attend most reliably to the start and end of a long context and
    weakest to the middle. Given docs already sorted best-to-worst, place the
    best one first, the second-best last, the third-best second, etc., so the
    most relevant documents land at the two ends instead of the middle.
    """
    ordered = [None] * len(ranked_idxs)
    left, right = 0, len(ranked_idxs) - 1
    for i, doc_idx in enumerate(ranked_idxs):
        if i % 2 == 0:
            ordered[left] = doc_idx
            left += 1
        else:
            ordered[right] = doc_idx
            right -= 1
    return ordered


if __name__ == "__main__":
    docs = [
        "Kubernetes pods auto-scale horizontally using CPU/memory HPA metrics.",
        "Error HTTP 503 occurs during traffic spikes when upstream connection pool is exhausted.",
        "Postgres connection pooling is handled by PgBouncer.",
        "When HTTP 503 errors occur, the circuit breaker trips and initiates exponential backoff."
    ]

    bm25 = BM25()
    bm25.fit(docs)
    query = "How to handle HTTP 503 errors and connection pool exhaustion?"

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print(f"QUERY: {query}")
    print(f"===================================================={Colors.RESET}\n")

    bm25_results = bm25.search(query)
    print(f"{Colors.CYAN}1. Sparse Search (Okapi BM25) — exact keyword matching:{Colors.RESET}")
    for idx, score in bm25_results[:3]:
        print(f"  • Doc {idx} (Score: {score:.3f}): {docs[idx]}")

    dense_results = dense_search(docs, query)
    print(f"\n{Colors.CYAN}2. Dense Semantic Search — hashing-trick pseudo-embeddings + cosine sim:{Colors.RESET}")
    for idx, score in dense_results[:3]:
        print(f"  • Doc {idx} (Score: {score:.3f}): {docs[idx]}")

    fused = reciprocal_rank_fusion([bm25_results, dense_results])
    print(f"\n{Colors.YELLOW}3. Reciprocal Rank Fusion — merges both rankings into one shortlist:{Colors.RESET}")
    for idx, score in fused:
        print(f"  • Doc {idx} (RRF Score: {score:.4f}): {docs[idx]}")

    shortlist = [idx for idx, _ in fused[:3]]
    reranked = cross_encoder_rerank(docs, query, shortlist)
    print(f"\n{Colors.MAGENTA}4. Cross-Encoder Rerank (heuristic bigram overlap on the shortlist):{Colors.RESET}")
    for idx, score in reranked:
        print(f"  • Doc {idx} (Rerank Score: {score:.1f}): {docs[idx]}")

    final_order = reorder_lost_in_the_middle([idx for idx, _ in reranked])
    print(f"\n{Colors.GREEN}5. Lost-in-the-Middle Reordering — best docs pushed to context start/end:{Colors.RESET}")
    for pos, idx in enumerate(final_order, 1):
        print(f"  • Position {pos}: Doc {idx}: {docs[idx]}")

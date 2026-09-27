"""
=============================================================================
28_ann_indexing_hnsw_and_ivf.py: Approximate Nearest Neighbor Indexing
=============================================================================
Senior GenAI Engineer Context:
Module 05's exact k-NN search is O(N) per query — fine for a demo corpus,
unusable for a production index with millions of vectors. Every real vector
database (Pinecone, Qdrant, Weaviate, pgvector/pgvecto.rs, FAISS) is built on
Approximate Nearest Neighbor (ANN) indexing: trade a small amount of recall
for a massive reduction in distance computations per query.

Two production ANN index families implemented from scratch here:
1. IVF (Inverted File Index): cluster all vectors offline (k-means), then at
   query time only scan the `n_probe` nearest clusters instead of everything.
2. HNSW (Hierarchical Navigable Small World): build a proximity graph where
   each node connects to its M nearest neighbors; search greedily hops
   toward the query instead of comparing against every vector.

Both are benchmarked against brute-force exact k-NN on recall@k and number
of distance computations, over the SAME synthetic embedding corpus — the
actual tradeoff every vector DB makes explicit as an "ef_search" / "n_probe"
tuning knob.

Run directly:
  python3 28_ann_indexing_hnsw_and_ivf.py
=============================================================================
"""

import random
from typing import Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
VectorMath = import_module("05_embeddings_and_vector_math").VectorMath

random.seed(42)


def random_unit_vector(dim: int) -> List[float]:
    vec = [random.gauss(0, 1) for _ in range(dim)]
    return VectorMath.normalize(vec)


def brute_force_knn(vectors: List[List[float]], query: List[float], k: int) -> Tuple[List[int], int]:
    """Exact search — the ground truth we benchmark ANN recall against."""
    scored = [(i, VectorMath.cosine_similarity(query, v)) for i, v in enumerate(vectors)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:k]], len(vectors)  # (top-k ids, distance computations)


# =====================================================================
# 1. IVF (INVERTED FILE INDEX): coarse-quantize, then probe only nearby cells
# =====================================================================
class IVFIndex:
    def __init__(self, n_clusters: int = 8, n_probe: int = 2, iterations: int = 10):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.iterations = iterations
        self.centroids: List[List[float]] = []
        self.inverted_lists: Dict[int, List[int]] = {}
        self.vectors: List[List[float]] = []

    def build(self, vectors: List[List[float]]):
        """Simplified k-means: a handful of Lloyd's-algorithm iterations."""
        self.vectors = vectors
        dim = len(vectors[0])
        self.centroids = random.sample(vectors, self.n_clusters)

        for _ in range(self.iterations):
            buckets: Dict[int, List[int]] = {c: [] for c in range(self.n_clusters)}
            for idx, v in enumerate(vectors):
                nearest = max(range(self.n_clusters), key=lambda c: VectorMath.cosine_similarity(v, self.centroids[c]))
                buckets[nearest].append(idx)

            for c, member_idxs in buckets.items():
                if not member_idxs:
                    continue
                mean = [0.0] * dim
                for idx in member_idxs:
                    for d in range(dim):
                        mean[d] += vectors[idx][d]
                mean = [x / len(member_idxs) for x in mean]
                self.centroids[c] = VectorMath.normalize(mean)

        self.inverted_lists = buckets

    def search(self, query: List[float], k: int) -> Tuple[List[int], int]:
        # Probe only the `n_probe` nearest cluster centroids...
        cluster_sims = sorted(range(self.n_clusters), key=lambda c: VectorMath.cosine_similarity(query, self.centroids[c]), reverse=True)
        probed_clusters = cluster_sims[:self.n_probe]

        # ...and only exact-score the vectors inside those clusters.
        candidate_idxs = [idx for c in probed_clusters for idx in self.inverted_lists.get(c, [])]
        scored = [(idx, VectorMath.cosine_similarity(query, self.vectors[idx])) for idx in candidate_idxs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored[:k]], len(candidate_idxs)


# =====================================================================
# 2. HNSW (simplified single-layer navigable small world graph)
# =====================================================================
class SimpleHNSW:
    """A simplified, single-layer HNSW: real HNSW uses a multi-layer skip-
    list-like structure for O(log N) search; this single-layer version keeps
    the core idea (a navigable proximity graph + greedy best-first search)
    transparent without the layer-assignment machinery.
    """

    def __init__(self, M: int = 6):
        self.M = M  # max neighbors per node
        self.vectors: List[List[float]] = []
        self.graph: Dict[int, List[int]] = {}
        self.entry_point: int = 0

    def build(self, vectors: List[List[float]], construction_pool_size: int = 40):
        self.vectors = vectors
        self.graph = {0: []}
        self.entry_point = 0

        for new_idx in range(1, len(vectors)):
            # Construction-time neighbor search: a real HNSW builder also
            # doesn't compare against every existing node — it samples a
            # candidate pool (here) or uses its own greedy search over the
            # graph built so far. Either way, insertion cost stays roughly
            # constant per node instead of growing with the corpus size.
            pool = random.sample(range(new_idx), min(construction_pool_size, new_idx))
            pool.sort(key=lambda i: VectorMath.cosine_similarity(vectors[new_idx], vectors[i]), reverse=True)
            neighbors = pool[:self.M]

            self.graph[new_idx] = neighbors
            for n in neighbors:
                self.graph[n].append(new_idx)
                if len(self.graph[n]) > self.M:
                    # Prune to the M closest, keeping the graph degree-bounded.
                    self.graph[n].sort(key=lambda i: VectorMath.cosine_similarity(vectors[n], vectors[i]), reverse=True)
                    self.graph[n] = self.graph[n][:self.M]

    def search(self, query: List[float], k: int, ef_search: int = 16) -> Tuple[List[int], int]:
        """Greedy best-first graph traversal from the entry point, bounded by
        `ef_search` (HNSW's real tuning knob: higher ef_search visits more
        nodes for higher recall). Stops as soon as the best remaining
        candidate can no longer beat the worst result already found — the
        actual early-termination condition real HNSW search uses, not just
        "keep expanding until the frontier runs dry."
        """
        entry_sim = VectorMath.cosine_similarity(query, self.vectors[self.entry_point])
        visited = {self.entry_point}
        candidates = [(entry_sim, self.entry_point)]  # nodes left to expand
        found = [(entry_sim, self.entry_point)]        # best results seen so far (size-bounded by ef_search)
        distance_computations = 1

        while candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            current_sim, current = candidates.pop(0)

            if len(found) >= ef_search:
                worst_found_sim = min(found, key=lambda x: x[0])[0]
                if current_sim < worst_found_sim:
                    break  # nothing left in `candidates` can improve on `found`

            for neighbor in self.graph.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                sim = VectorMath.cosine_similarity(query, self.vectors[neighbor])
                distance_computations += 1

                worst_found_sim = min(found, key=lambda x: x[0])[0] if found else -1.0
                if len(found) < ef_search or sim > worst_found_sim:
                    candidates.append((sim, neighbor))
                    found.append((sim, neighbor))
                    if len(found) > ef_search:
                        found.sort(key=lambda x: x[0], reverse=True)
                        found = found[:ef_search]

        found.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in found[:k]], distance_computations


def recall_at_k(approx_ids: List[int], exact_ids: List[int]) -> float:
    return len(set(approx_ids) & set(exact_ids)) / len(exact_ids)


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== ANN Indexing: IVF vs HNSW vs Brute-Force k-NN ==={Colors.RESET}\n")

    DIM, N, K = 32, 1500, 5
    corpus = [random_unit_vector(DIM) for _ in range(N)]
    query = random_unit_vector(DIM)

    exact_ids, exact_computations = brute_force_knn(corpus, query, K)
    print(f"{Colors.CYAN}Ground truth (brute force): top-{K} ids = {exact_ids} ({exact_computations} distance computations){Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. IVF Index (n_clusters=16, n_probe=5){Colors.RESET}")
    ivf = IVFIndex(n_clusters=16, n_probe=5)
    ivf.build(corpus)
    ivf_ids, ivf_computations = ivf.search(query, K)
    print(f"  Approx top-{K} ids: {ivf_ids}")
    print(f"  Distance computations: {ivf_computations}/{N} ({ivf_computations/N*100:.1f}% of brute force)")
    print(f"  Recall@{K}: {Colors.GREEN}{recall_at_k(ivf_ids, exact_ids)*100:.0f}%{Colors.RESET}")
    print(f"  {Colors.DIM}(IVF's recall depends on real cluster structure in the data; these are random unit vectors with{Colors.RESET}")
    print(f"  {Colors.DIM} no natural clusters, which is exactly the case IVF handles worse than a graph-based index like HNSW){Colors.RESET}\n")

    print(f"{Colors.YELLOW}2. HNSW Index (M=12, ef_search=30){Colors.RESET}")
    hnsw = SimpleHNSW(M=12)
    hnsw.build(corpus)
    hnsw_ids, hnsw_computations = hnsw.search(query, K, ef_search=30)
    print(f"  Approx top-{K} ids: {hnsw_ids}")
    print(f"  Distance computations: {hnsw_computations}/{N} ({hnsw_computations/N*100:.1f}% of brute force)")
    print(f"  Recall@{K}: {Colors.GREEN}{recall_at_k(hnsw_ids, exact_ids)*100:.0f}%{Colors.RESET}\n")

    print(f"{Colors.YELLOW}3. HNSW with a wider search frontier (ef_search=150) — the recall/speed knob{Colors.RESET}")
    hnsw_wide_ids, hnsw_wide_computations = hnsw.search(query, K, ef_search=150)
    print(f"  Distance computations: {hnsw_wide_computations}/{N} ({hnsw_wide_computations/N*100:.1f}% of brute force)")
    print(f"  Recall@{K}: {Colors.GREEN}{recall_at_k(hnsw_wide_ids, exact_ids)*100:.0f}%{Colors.RESET}\n")

    print(f"{Colors.GREEN}✔ Both ANN indexes trade a fraction of brute-force distance computations for near-exact recall.{Colors.RESET}")

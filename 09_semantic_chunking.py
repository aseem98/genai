"""
=============================================================================
09_semantic_chunking.py: Semantic Chunking (Embedding-Driven Chunk Boundaries)
=============================================================================
Senior GenAI Engineer Context:
Module 08's recursive character splitter and module 46's hierarchical
chunker both cut text at STRUCTURAL boundaries (separators, headings) or a
fixed size. Neither looks at MEANING. Semantic chunking instead:
  1. Splits the document into sentences.
  2. Embeds every sentence.
  3. Walks the sequence of consecutive-sentence similarities and cuts a new
     chunk boundary wherever similarity drops sharply — a "topic shift" —
     instead of every N characters.

This directly fixes a real failure mode of fixed-size chunking: a hard cut
at char 500 can slice a single idea in half, hurting both retrieval (the
embedding is now a diluted mix of two topics) and the LLM's ability to
reason over a complete thought. Semantic chunking keeps each chunk
topically coherent, at the cost of variable chunk sizes.

Run directly:
  python3 09_semantic_chunking.py
=============================================================================
"""

import re
from typing import List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
VectorMath = import_module("05_embeddings_and_vector_math").VectorMath
embed_text = import_module("01_unified_llm_client").embed_text


def split_into_sentences(text: str) -> List[str]:
    # A real system uses a proper sentence tokenizer (spaCy, nltk); this
    # regex-based split is the same simplification module 08 already uses
    # for word/paragraph boundaries.
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def semantic_chunk(text: str, breakpoint_percentile_threshold: float = 40.0, min_chunk_sentences: int = 1) -> List[str]:
    """The real technique (used by LlamaIndex's SemanticSplitterNodeParser
    and similar): compute the similarity between each pair of consecutive
    sentences, then cut wherever that similarity falls in the bottom
    `breakpoint_percentile_threshold`% of ALL observed similarities in this
    document — an adaptive threshold, not a fixed magic number, since
    "how similar is similar enough" varies document to document.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = [embed_text(s) for s in sentences]
    consecutive_similarities = [
        VectorMath.cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # A LOW similarity is a candidate breakpoint, so the cutoff is the
    # threshold-th percentile of the similarity distribution — anything
    # below it is treated as a genuine topic shift.
    cutoff = _percentile(consecutive_similarities, breakpoint_percentile_threshold)

    chunks: List[List[str]] = [[sentences[0]]]
    for i, sim in enumerate(consecutive_similarities):
        if sim < cutoff and len(chunks[-1]) >= min_chunk_sentences:
            chunks.append([])
        chunks[-1].append(sentences[i + 1])

    return [" ".join(c) for c in chunks]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def fixed_size_chunk(text: str, size: int = 120) -> List[str]:
    """The naive baseline this module is contrasted against."""
    words = text.split()
    chunks, current = [], []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_topic_purity(document: str, chunks: List[str], sentences: List[str], topics: List[str]) -> Tuple[float, List[List[str]]]:
    """Measures, instead of asserting, how many chunks contain exactly one
    topic: locate each chunk's character span in the document, then collect
    the ground-truth topic of every sentence that span overlaps."""
    sentence_spans, pos = [], 0
    for s in sentences:
        start = document.find(s, pos)
        sentence_spans.append((start, start + len(s)))
        pos = start + len(s)
    chunk_topics, pos = [], 0
    for c in chunks:
        start = document.find(c, pos)
        end = start + len(c)
        pos = end
        chunk_topics.append(sorted({t for (s0, s1), t in zip(sentence_spans, topics) if s0 < end and s1 > start}))
    pure = sum(1 for t in chunk_topics if len(t) == 1)
    return pure / len(chunks), chunk_topics


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 09: Semantic Chunking ==={Colors.RESET}\n")

    labeled_sentences = [  # (sentence, ground-truth topic) — the labels are only used for scoring
        ("PgBouncer pools Postgres connections to reduce Postgres connection overhead.", "pooling"),
        ("PgBouncer supports three connection pooling modes: session, transaction, and statement pooling.", "pooling"),
        ("PgBouncer transaction pooling mode is recommended for high-concurrency Postgres workloads.", "pooling"),
        ("Switching topics entirely, our company quarterly revenue grew 18 percent year over year.", "revenue"),
        ("Quarterly revenue growth was driven primarily by expansion in the enterprise revenue segment.", "revenue"),
        ("Enterprise segment customers now represent 60 percent of total company revenue.", "revenue"),
        ("Returning to infrastructure topics, Redis eviction policies control Redis behavior under memory pressure.", "redis"),
        ("The Redis allkeys-lru eviction policy evicts the least recently used Redis keys regardless of TTL.", "redis"),
    ]
    sentences = [s for s, _ in labeled_sentences]
    topics = [t for _, t in labeled_sentences]
    document = " ".join(sentences)

    print(f"{Colors.YELLOW}1. Naive fixed-size chunking (120 chars) — cuts mid-topic{Colors.RESET}")
    fixed_chunks = fixed_size_chunk(document, size=120)
    for i, c in enumerate(fixed_chunks, 1):
        print(f"  Chunk {i}: {c}")

    print(f"\n{Colors.YELLOW}2. Semantic chunking — boundaries fall at actual topic shifts{Colors.RESET}")
    semantic_chunks = semantic_chunk(document, breakpoint_percentile_threshold=40.0)
    for i, c in enumerate(semantic_chunks, 1):
        print(f"  Chunk {i}: {c}")

    print(f"\n{Colors.YELLOW}3. Measured topic purity (share of chunks that contain exactly ONE topic){Colors.RESET}")
    for label, chunks in (("Fixed-size", fixed_chunks), ("Semantic", semantic_chunks)):
        purity, chunk_topics = chunk_topic_purity(document, chunks, sentences, topics)
        color = Colors.GREEN if purity == 1.0 else Colors.YELLOW
        print(f"  {label:<11} {color}{purity * 100:5.1f}% pure{Colors.RESET}  sizes={[len(c) for c in chunks]}  topics per chunk={chunk_topics}")
    print(f"  {Colors.DIM}A mixed-topic chunk embeds as a blurred average of both topics and retrieves worse for either.{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Semantic and fixed-size chunking compared on a measured purity score.{Colors.RESET}")

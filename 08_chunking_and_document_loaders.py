"""
=============================================================================
08_chunking_and_document_loaders.py: Document Ingestion & Chunking Strategies
=============================================================================
Senior GenAI Engineer Context:
"Garbage in, garbage out" is the #1 failure mode of RAG pipelines.
If you chunk documents poorly:
1. Sentences are truncated in the middle, breaking semantic meaning.
2. Context across paragraphs is lost if overlap is zero.
3. Metadata (headers, source URLs, page numbers) is lost.

Strategies implemented here (mirroring LangChain DocumentLoaders & TextSplitters):
1. Fixed-Size Character Chunking (with boundary artifacts).
2. Recursive Character Text Splitting (hierarchically splitting by ["\\n\\n", "\\n", " ", ""]).
3. Markdown Header-Aware Chunking (extracting H1/H2 metadata into chunk metadata).
4. Chunk Overlap analysis.

Run directly:
  python3 08_chunking_and_document_loaders.py
=============================================================================
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from importlib import import_module

Colors = import_module("01_unified_llm_client").Colors
bpe_module = import_module("02_tokenization_bpe_from_scratch")
BPETokenizer = bpe_module.BPETokenizer
load_training_corpus = bpe_module.load_training_corpus


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 1. RECURSIVE CHARACTER TEXT SPLITTER (LANGCHAIN PATTERN)
# =====================================================================
class RecursiveCharacterTextSplitter:
    """
    Splits text recursively by looking at the largest semantic separator first:
    ["\n\n" (paragraphs), "\n" (lines), " " (words), "" (characters)]
    """

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40, length_function=len):
        """`length_function` decides what "size" means: `len` counts
        characters; passing a tokenizer's token counter makes chunk_size a
        TOKEN budget (LangChain's `from_tiktoken_encoder` does exactly this)."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        splits = self._split_recursive(text, self.separators)
        size = lambda pieces: self.length_function("".join(pieces))  # measure the JOINED text, not a sum

        current_chunk: List[str] = []
        for s in splits:
            # Flush repeatedly (not just once) until `s` actually fits, since
            # the retained overlap can itself still exceed chunk_size.
            while current_chunk and size(current_chunk + [s]) > self.chunk_size:
                final_chunks.append("".join(current_chunk))
                overlap_pieces: List[str] = []
                for piece in reversed(current_chunk):
                    if size([piece] + overlap_pieces) <= self.chunk_overlap:
                        overlap_pieces.insert(0, piece)
                    else:
                        break
                if len(overlap_pieces) == len(current_chunk):
                    # Overlap couldn't shrink the carry — drop it so the loop
                    # always makes progress instead of spinning forever.
                    overlap_pieces = []
                current_chunk = overlap_pieces
            current_chunk.append(s)

        if current_chunk:
            final_chunks.append("".join(current_chunk))
        return final_chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        result = []
        for i, s in enumerate(splits):
            piece = s + (sep if i < len(splits) - 1 else "")
            if self.length_function(piece) > self.chunk_size and remaining_seps:
                result.extend(self._split_recursive(piece, remaining_seps))
            else:
                result.append(piece)
        return result


# =====================================================================
# 2. MARKDOWN HEADER-AWARE SPLITTER
# =====================================================================
class MarkdownHeaderTextSplitter:
    """Splits markdown based on headers and tags metadata with section titles."""

    def split_markdown(self, markdown_text: str) -> List[Document]:
        lines = markdown_text.split("\n")
        documents = []
        current_header = "Introduction"
        current_lines = []

        for line in lines:
            if line.startswith("# ") or line.startswith("## "):
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        documents.append(Document(page_content=content, metadata={"section": current_header}))
                    current_lines = []
                current_header = line.lstrip("#").strip()
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                documents.append(Document(page_content=content, metadata={"section": current_header}))

        return documents


# =====================================================================
# 3. TOKEN-BASED SPLITTER (sizes in tokens, like tiktoken's TokenTextSplitter)
# =====================================================================
class TokenTextSplitter:
    """Context windows and embedding-model limits are measured in TOKENS,
    not characters, so production splitters count tokens. Uses module 02's
    real BPE tokenizer: encode once, slice the id list into windows of
    `chunk_tokens` with `overlap_tokens` shared between neighbors, decode."""

    def __init__(self, tokenizer, chunk_tokens: int = 60, overlap_tokens: int = 10):
        assert overlap_tokens < chunk_tokens
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    def split_text(self, text: str) -> List[str]:
        ids = self.tokenizer.encode(text)
        step = self.chunk_tokens - self.overlap_tokens
        return [self.tokenizer.decode(ids[i:i + self.chunk_tokens]) for i in range(0, len(ids), step)
                if ids[i:i + self.chunk_tokens]]


# =====================================================================
# 4. RUNNABLE TESTS & COMPARISON
# =====================================================================
if __name__ == "__main__":
    sample_doc = """# Database Architecture Guide

PostgreSQL uses Multi-Version Concurrency Control (MVCC) to handle high concurrency.
Each transaction sees a snapshot of data rather than locking rows directly.

## Replication and High Availability
Streaming replication sends Write-Ahead Logs (WAL) continuously from Primary to Standby.
In synchronous mode, a commit is not acknowledged until written to at least one standby replica.
This guarantees zero data loss (RPO = 0) at the expense of slight commit latency.

## Vacuuming and Maintenance
Dead tuples generated by UPDATE and DELETE operations are reclaimed by the autovacuum daemon.
Failure to tune autovacuum leads to index bloating and severe table scan degradation."""

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("1. MARKDOWN HEADER-AWARE SPLITTING (METADATA PRESERVED)")
    print(f"===================================================={Colors.RESET}\n")

    md_splitter = MarkdownHeaderTextSplitter()
    sections = md_splitter.split_markdown(sample_doc)

    for i, doc in enumerate(sections, 1):
        print(f"{Colors.CYAN}[Section {i}]: Section='{doc.metadata['section']}'{Colors.RESET}")
        print(f"{doc.page_content}\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("2. RECURSIVE CHARACTER TEXT SPLITTING (SIZE=180, OVERLAP=40)")
    print(f"===================================================={Colors.RESET}\n")

    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=40)
    chunks = rec_splitter.split_text(sample_doc)

    for idx, c in enumerate(chunks, 1):
        print(f"{Colors.YELLOW}[Chunk #{idx} ({len(c)} chars)]:{Colors.RESET}")
        print(f"\"{c.strip()}\"\n")

    print(f"{Colors.GREEN}✔ Notice how the recursive separator hierarchy avoids clipping words mid-token, unlike a naive fixed-width slice.{Colors.RESET}\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("3. TOKEN-BASED SPLITTING (60 TOKENS, 10-TOKEN OVERLAP) WITH MODULE 02's BPE")
    print(f"===================================================={Colors.RESET}\n")
    tokenizer = BPETokenizer()
    tokenizer.train(load_training_corpus(), vocab_size=512)
    token_chunks = TokenTextSplitter(tokenizer, chunk_tokens=60, overlap_tokens=10).split_text(sample_doc)
    for idx, c in enumerate(token_chunks, 1):
        print(f"{Colors.YELLOW}[Token chunk #{idx}: {len(tokenizer.encode(c))} tokens, {len(c)} chars]{Colors.RESET} \"{c.strip()[:90]}...\"")
    print(f"\n  Every chunk is <= 60 tokens by construction (lossless round trip: "
          f"{tokenizer.decode(tokenizer.encode(sample_doc)) == sample_doc}) — but notice the mid-word cuts.\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("4. BEST OF BOTH: RECURSIVE SEPARATORS, SIZE MEASURED IN TOKENS")
    print(f"===================================================={Colors.RESET}\n")
    count_tokens = lambda t: len(tokenizer.encode(t))
    hybrid_chunks = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10, length_function=count_tokens).split_text(sample_doc)
    for idx, c in enumerate(hybrid_chunks, 1):
        print(f"{Colors.YELLOW}[Chunk #{idx}: {count_tokens(c)} tokens]{Colors.RESET} \"{c.strip()[:90]}...\"")
    worst = max(count_tokens(c) for c in hybrid_chunks)
    print(f"\n  Largest chunk = {worst} tokens (budget 60), and cuts land on paragraph/line/word boundaries.")
    print(f"{Colors.GREEN}✔ This is the production default: structure-aware splitting with a token-denominated budget.{Colors.RESET}")

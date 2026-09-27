"""
=============================================================================
12_langchain_retrievers_and_chains.py: LangChain Retrievers & RAG Chains
=============================================================================
Senior GenAI Engineer Context:
A basic RAG chain takes a query and retrieves chunks directly.
In production, users phrase questions with ambiguous or imprecise terms.

Advanced Retriever Patterns implemented here:
1. Standard LCEL RAG Chain:
   `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser`
2. MultiQueryRetriever: Uses an LLM to generate multiple alternative perspectives
   of the user's prompt to overcome semantic vocabulary mismatches.
3. Contextual Compression: Extracts only relevant sentences from retrieved docs,
   reducing prompt token costs and noise.

Run directly:
  python3 12_langchain_retrievers_and_chains.py
=============================================================================
"""


import re
from typing import List
from importlib import import_module

lcel_module = import_module("10_langchain_lcel_runnables_and_pipes")
Runnable = lcel_module.Runnable
RunnablePassthrough = lcel_module.RunnablePassthrough
PromptTemplate = lcel_module.PromptTemplate
StrOutputParser = lcel_module.StrOutputParser
Colors = lcel_module.Colors
LLMClient = import_module("01_unified_llm_client").LLMClient


# =====================================================================
# 1. VECTOR RETRIEVER BASE
# =====================================================================
class SimpleRetriever(Runnable):
    def __init__(self, documents: List[str]):
        self.documents = documents

    def invoke(self, query: str) -> List[str]:
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        scored = []
        for doc in self.documents:
            d_tokens = set(re.findall(r"\b\w+\b", doc.lower()))
            overlap = len(q_tokens.intersection(d_tokens))
            scored.append((doc, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:2]]


def format_docs(docs: List[str]) -> str:
    return "\n\n".join([f"Document {i+1}:\n{d}" for i, d in enumerate(docs)])


# =====================================================================
# 2. ADVANCED MULTI-QUERY RETRIEVER
# =====================================================================
class MultiQueryRetriever(Runnable):
    """
    Expands a single user question into multiple synthetic query variants
    to bridge the vocabulary gap between user input and indexed documents.
    """

    def __init__(self, base_retriever: SimpleRetriever, llm: LLMClient):
        self.retriever = base_retriever
        self.llm = llm

    def invoke(self, original_query: str) -> List[str]:
        print(f"  {Colors.DIM}[MultiQuery] Original Query: \"{original_query}\"{Colors.RESET}")

        # Ask the LLM to rewrite the query from different angles. In live
        # mode this is a real model call; in simulation mode we still route
        # through `self.llm.chat` (rather than hardcoding the variants here)
        # so both modes exercise the same code path.
        expansion_prompt = (
            f"Generate 2 alternative phrasings of this search query, one per line, "
            f"to broaden document recall: \"{original_query}\""
        )
        raw_variations = self.llm.chat(
            [{"role": "user", "content": expansion_prompt}],
            simulated_response=(
                "Postgres database high latency connections connection pooling\n"
                "PgBouncer max connection limit configuration"
            )
        )
        variations = [original_query] + [line.strip() for line in raw_variations.splitlines() if line.strip()]

        print(f"  {Colors.CYAN}[MultiQuery] Generated {len(variations)} query variations for broader recall:{Colors.RESET}")
        for v in variations:
            print(f"    • {v}")

        # Retrieve docs for each variation and deduplicate while preserving
        # order (a plain `set()` would shuffle output nondeterministically
        # across runs since Python randomizes string hashing per process).
        unique_docs = dict.fromkeys(
            d for q in variations for d in self.retriever.invoke(q)
        )

        return list(unique_docs)


# =====================================================================
# 3. CONTEXTUAL COMPRESSION RETRIEVER
# =====================================================================
class ContextualCompressionRetriever(Runnable):
    """Retrieved documents are usually mostly irrelevant to the specific
    question — a paragraph about Postgres might have one useful sentence.
    This retriever asks the LLM to extract ONLY the relevant sentences from
    each retrieved doc (LangChain's LLMChainExtractor) and drops docs with
    nothing relevant, shrinking the context the final answer is built on."""

    def __init__(self, base_retriever: Runnable, llm: LLMClient):
        self.base_retriever = base_retriever
        self.llm = llm
        self.last_stats = {}

    @staticmethod
    def _keyword_extract(query: str, doc: str) -> str:
        """Offline fallback: keep sentences sharing a content word with the query."""
        q = {w for w in re.findall(r"\b\w{4,}\b", query.lower())}
        keep = [s for s in re.split(r"(?<=[.!?])\s+", doc) if q & set(re.findall(r"\b\w{4,}\b", s.lower()))]
        return " ".join(keep) or "NONE"

    def invoke(self, query: str) -> List[str]:
        docs = self.base_retriever.invoke(query)
        compressed = []
        for doc in docs:
            extract = self.llm.chat([{"role": "user", "content": (
                f"Question: {query}\n\nDocument:\n{doc}\n\n"
                "Copy out ONLY the sentences from the document that help answer the question, verbatim. "
                "If none are relevant, reply exactly NONE."
            )}], simulated_response=self._keyword_extract(query, doc), temperature=0).strip()
            if extract and extract.upper() != "NONE":
                compressed.append(extract)
        self.last_stats = {"docs_in": len(docs), "docs_out": len(compressed),
                           "chars_in": sum(map(len, docs)), "chars_out": sum(map(len, compressed))}
        return compressed


# =====================================================================
# 4. RUNNABLE TESTS & MODERN LCEL RAG CHAIN
# =====================================================================
if __name__ == "__main__":
    knowledge_base = [
        "PgBouncer acts as a lightweight connection pooler for PostgreSQL, buffering incoming client spikes. "
        "It was originally developed at Skype. The project is written in C and licensed under the ISC license. "
        "Transaction pooling mode lets thousands of clients share a few dozen server connections.",
        "When database connection count exceeds max_connections, Postgres throws FATAL: remaining connection slots are reserved. "
        "The default max_connections is 100. Each connection is a separate backend process that consumes several megabytes of RAM. "
        "Postgres 16 added improvements to logical replication.",
        "Kafka consumer lag indicates downstream processing workers cannot keep pace with producer write throughput. "
        "Lag is measured per partition.",
        "Redis cluster slot resharding can be performed live without taking nodes offline. "
        "Redis was created by Salvatore Sanfilippo."
    ]

    client = LLMClient()
    base_retriever = SimpleRetriever(knowledge_base)
    multi_retriever = MultiQueryRetriever(base_retriever, client)

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("1. LCEL RAG CHAIN WITH QUERY EXPANSION (MULTI-QUERY RETRIEVER)")
    print(f"===================================================={Colors.RESET}\n")

    rag_prompt = PromptTemplate(
        "You are an SRE assistant. Answer the user question based strictly on this context:\n"
        "--- CONTEXT ---\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    class RAGModel(Runnable):
        """Generator step of the RAG chain: forwards the retrieval-grounded
        prompt to the real LLM. In simulation mode it falls back to a canned
        answer that's still grounded in the same retrieved context, so both
        modes exercise the identical `invoke()` code path.
        """

        def __init__(self, llm: LLMClient):
            self.llm = llm

        def invoke(self, formatted_prompt: str) -> str:
            return self.llm.chat(
                [{"role": "user", "content": formatted_prompt}],
                simulated_response=(
                    "Based on the provided documentation, Postgres rejects new clients with 'FATAL: remaining connection slots' "
                    "when the connection count exceeds max_connections. You should deploy PgBouncer to pool connections and buffer client spikes."
                ),
            )

    # Modern LCEL RAG Composition:
    # 1. RunnableParallel dict formats context via retriever and passes raw question
    # 2. Injected into prompt template
    # 3. Forwarded to LLM
    # 4. Parsed by StrOutputParser
    rag_chain = (
        {
            "context": multi_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | RAGModel(client)
        | StrOutputParser()
    )

    user_q = "How do we prevent our Postgres DB from running out of connections?"
    print(f"{Colors.BOLD}User Question:{Colors.RESET} {user_q}\n")

    answer = rag_chain.invoke(user_q)
    print(f"\n{Colors.BOLD}{Colors.GREEN}Final RAG Synthesized Answer:{Colors.RESET}\n{answer}\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("2. CONTEXTUAL COMPRESSION — KEEP ONLY THE RELEVANT SENTENCES")
    print(f"===================================================={Colors.RESET}\n")
    compressor = ContextualCompressionRetriever(base_retriever, client)
    compressed_docs = compressor.invoke(user_q)
    stats = compressor.last_stats
    for d in compressed_docs:
        print(f"  • {d}")
    print(f"\n  docs: {stats['docs_in']} -> {stats['docs_out']} | context chars: {stats['chars_in']} -> {stats['chars_out']} "
          f"({(1 - stats['chars_out'] / max(stats['chars_in'], 1)) * 100:.0f}% smaller)")
    print(f"  {Colors.DIM}Trivia like the license or the project's origin is dropped before it can distract the answer or cost tokens.{Colors.RESET}\n")

    compressed_chain = (
        {"context": compressor | format_docs, "question": RunnablePassthrough()}
        | rag_prompt | RAGModel(client) | StrOutputParser()
    )
    print(f"{Colors.BOLD}{Colors.GREEN}Answer from compressed context:{Colors.RESET}\n{compressed_chain.invoke(user_q)}")

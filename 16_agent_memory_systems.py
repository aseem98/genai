"""
=============================================================================
16_agent_memory_systems.py: The 4 Types of Agent Memory
=============================================================================
An agent's "memory" is not one thing — production systems separate it into
four tiers with different storage, retrieval, and lifetime characteristics:

1. WORKING MEMORY: the live conversation buffer. Bounded by the context
   window, so once it's near-full, older turns are SUMMARIZED (not just
   dropped) into a running summary — an LLM call, not a data structure.
2. EPISODIC MEMORY: specific past interactions and the feedback/preferences
   they produced, recalled by relevance to the current situation.
3. SEMANTIC MEMORY: general factual knowledge, retrieved by real embedding
   similarity (module 05) rather than keyword overlap.
4. PROCEDURAL MEMORY: standing instructions — SOPs, tone, constraints —
   injected every turn as the system prompt, not retrieved.

The demo builds up working memory until it needs to compress, and shows all
four tiers assembled into one final prompt for a real agent response.

Run directly:
  python3 16_agent_memory_systems.py
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
embed_text = llm_module.embed_text
vecmath_module = import_module("05_embeddings_and_vector_math")
VectorMath = vecmath_module.VectorMath
SimpleVectorIndex = vecmath_module.SimpleVectorIndex


# =====================================================================
# 1. SEMANTIC MEMORY — real embedding similarity, not keyword overlap
# =====================================================================
class SemanticMemory:
    def __init__(self):
        self.index = SimpleVectorIndex(metric="cosine")

    def add(self, fact: str):
        self.index.add(fact, embed_text(fact))

    def search(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        return self.index.search(embed_text(query), top_k=top_k)


# =====================================================================
# 2. EPISODIC MEMORY — past interactions, recalled by relevance
# =====================================================================
@dataclass
class Episode:
    task: str
    feedback: str
    embedding: List[float] = field(default_factory=list)


class EpisodicMemory:
    def __init__(self):
        self.episodes: List[Episode] = []

    def record(self, task: str, feedback: str):
        self.episodes.append(Episode(task, feedback, embed_text(f"{task} {feedback}")))

    def recall(self, query: str, top_k: int = 1) -> List[Episode]:
        if not self.episodes:
            return []
        q = embed_text(query)
        ranked = sorted(self.episodes, key=lambda e: VectorMath.cosine_similarity(q, e.embedding), reverse=True)
        return ranked[:top_k]


# =====================================================================
# 3. WORKING MEMORY — sliding window + LLM-summarized older turns
# =====================================================================
class WorkingMemory:
    def __init__(self, llm: LLMClient, max_recent_turns: int = 4):
        self.llm = llm
        self.max_recent_turns = max_recent_turns
        self.recent: List[Dict[str, str]] = []
        self.running_summary: str = ""

    def add_turn(self, role: str, content: str):
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > self.max_recent_turns:
            self._compress_oldest()

    def _compress_oldest(self):
        to_summarize = self.recent[:-self.max_recent_turns] if len(self.recent) > self.max_recent_turns else []
        if not to_summarize:
            return
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in to_summarize)
        prompt = (
            (f"Existing summary of earlier conversation:\n{self.running_summary}\n\n" if self.running_summary else "")
            + f"New turns to fold in:\n{transcript}\n\nWrite one updated, concise summary covering everything above."
        )
        self.running_summary = self.llm.chat(
            [{"role": "user", "content": prompt}],
            simulated_response="User is configuring a Postgres deployment and prefers SQLAlchemy ORM over raw SQL.",
            temperature=0,
        )
        self.recent = self.recent[-self.max_recent_turns:]

    def context_messages(self) -> List[Dict[str, str]]:
        prefix = [{"role": "system", "content": f"Summary of earlier conversation: {self.running_summary}"}] if self.running_summary else []
        return prefix + self.recent


PROCEDURAL_MEMORY = (
    "Standing instructions: be concise, always state which environment (dev/staging/prod) a config "
    "applies to, and never suggest storing credentials in plaintext."
)


if __name__ == "__main__":
    llm = LLMClient()
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 16: The 4 Types of Agent Memory ==={Colors.RESET}")
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Working memory compresses once it exceeds its window{Colors.RESET}")
    working = WorkingMemory(llm, max_recent_turns=4)
    conversation = [
        ("user", "I need to set up our Postgres connection for the staging environment."),
        ("assistant", "Sure — what connection pool size do you want?"),
        ("user", "Let's start with 20 connections and a 15 second timeout."),
        ("assistant", "Got it: pool_size=20, timeout=15s for staging."),
        ("user", "Also, please use SQLAlchemy ORM models instead of raw SQL strings — that's a hard preference."),
        ("assistant", "Understood, SQLAlchemy ORM it is."),
        ("user", "Now, what should the timeout be for our Postgres connection again?"),
    ]
    for role, content in conversation:
        before = len(working.recent)
        working.add_turn(role, content)
        if len(working.recent) < before + 1:
            print(f"  {Colors.MAGENTA}[compressed {before + 1 - len(working.recent)} old turn(s) into summary]{Colors.RESET} "
                  f"-> \"{working.running_summary[:90]}...\"")
    print(f"  Working memory now holds {len(working.recent)} raw turns + 1 running summary (not the full history)\n")

    print(f"{Colors.YELLOW}2. Semantic memory — recall via embedding similarity{Colors.RESET}")
    semantic = SemanticMemory()
    facts = [
        "PostgreSQL database connection timeout is configured to 15 seconds in staging.",
        "Redis eviction policy is allkeys-lru with a 2GB memory cap.",
        "The billing service deploys to us-east-1 and eu-west-1.",
    ]
    for fact in facts:
        semantic.add(fact)
    query = "what is our postgres connection timeout value?"
    hits = semantic.search(query, top_k=1)
    print(f"  Query: {query!r}")
    print(f"  Best match (sim={hits[0][1]:.3f}): {hits[0][0]}")
    if llm.embedding_mode == "hashing":
        paraphrase = "how long until a stale postgres connection is dropped?"  # no shared words with the intended fact
        wrong = semantic.search(paraphrase, top_k=1)[0]
        print(f"  {Colors.YELLOW}Feature hashing needs shared WORDS: the paraphrase {paraphrase!r} retrieves")
        print(f"  the wrong fact instead ({wrong[0][:45]}...) — set EMBEDDING_MODEL for real semantic recall.{Colors.RESET}\n")
    else:
        print()

    print(f"{Colors.YELLOW}3. Episodic memory — recall a specific past interaction, by relevance{Colors.RESET}")
    episodic = EpisodicMemory()
    episodic.record("Configure Postgres connection script", "User explicitly prefers SQLAlchemy ORM models instead of raw SQL strings.")
    episodic.record("Set up CI pipeline", "User wants GitHub Actions, not Jenkins.")
    best_episode = episodic.recall("How should I write the database access layer?", top_k=1)[0]
    print(f"  Recalled preference: {best_episode.feedback}\n")

    print(f"{Colors.YELLOW}4. Procedural memory (standing instructions, injected every turn){Colors.RESET}")
    print(f"  {PROCEDURAL_MEMORY}\n")

    print(f"{Colors.YELLOW}5. All four tiers assembled into one real agent turn{Colors.RESET}")
    user_question = working.recent[-1]["content"]
    messages = (
        [{"role": "system", "content": PROCEDURAL_MEMORY}]
        + working.context_messages()
        + [{"role": "system", "content": (
            f"[Semantic memory] {hits[0][0]}\n"
            f"[Episodic memory] Past preference: {best_episode.feedback}"
        )}]
    )
    response = llm.chat(messages, simulated_response=(
        "The Postgres connection timeout for staging is 15 seconds. As a reminder, please implement this via "
        "SQLAlchemy ORM per your earlier preference, and don't hardcode credentials in the config."
    ))
    print(f"  {Colors.GREEN}Agent:{Colors.RESET} {response}")
    mentions_timeout, mentions_orm = "15" in response, "sqlalchemy" in response.lower() or "orm" in response.lower()
    print(f"\n  Uses semantic fact (15s): {Colors.GREEN + '✔' if mentions_timeout else Colors.RED + '✖'}{Colors.RESET}  "
          f"Honors episodic preference (ORM): {Colors.GREEN + '✔' if mentions_orm else Colors.RED + '✖'}{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Working, semantic, episodic and procedural memory all exercised in one live turn.{Colors.RESET}")

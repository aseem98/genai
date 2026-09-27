"""
=============================================================================
10_langchain_lcel_runnables_and_pipes.py: LangChain Expression Language (LCEL)
=============================================================================
Senior GenAI Engineer Context:
Legacy LangChain relied on bloated classes like `LLMChain` and `SequentialChain`.
Modern LangChain (v0.2/0.3+) is built entirely on LCEL (LangChain Expression Language)
and the `Runnable` protocol.

Core LCEL Architecture implemented here from scratch:
1. The `Runnable` Protocol: Base class defining `invoke()`, `batch()`, and `__or__` (`|`).
2. The Pipe Operator (`|`): Enables declarative Unix-style chaining:
   `chain = prompt | model | parser`
3. `RunnablePassthrough`: Passes raw inputs forward without modification.
4. `RunnableParallel` (or dict): Dispatches multiple runnables concurrently.
5. `RunnableLambda`: Wraps arbitrary Python callables into runnables.

Run directly:
  python3 10_langchain_lcel_runnables_and_pipes.py
=============================================================================
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Callable, Sequence, Type, Union
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


# =====================================================================
# 1. RUNNABLE BASE PROTOCOL & PIPE OPERATOR (__or__)
# =====================================================================
class Runnable:
    """Base class for all LCEL components supporting Unix pipe syntax (`|`)."""

    def invoke(self, input_data: Any) -> Any:
        raise NotImplementedError

    def batch(self, inputs: List[Any], max_concurrency: int = 4) -> List[Any]:
        """Like LangChain's `batch`: inputs run concurrently (I/O-bound LLM
        calls overlap in threads), results come back in input order."""
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            return list(pool.map(self.invoke, inputs))

    def with_retry(self, attempts: int = 3, retry_on: Sequence[Type[Exception]] = (Exception,)) -> "RunnableRetry":
        return RunnableRetry(self, attempts, tuple(retry_on))

    def with_fallbacks(self, fallbacks: List["Runnable"]) -> "RunnableWithFallbacks":
        return RunnableWithFallbacks(self, fallbacks)

    def __or__(self, other: Union["Runnable", Callable, Dict[str, Any]]) -> "RunnableSequence":
        """Overloads the `|` operator to create a chain."""
        if isinstance(other, dict):
            other = RunnableParallel(other)
        elif callable(other) and not isinstance(other, Runnable):
            other = RunnableLambda(other)
        return RunnableSequence(self, other)

    def __ror__(self, other: Union[Callable, Dict[str, Any]]) -> "RunnableSequence":
        """Allows `dict | runnable` syntax."""
        if isinstance(other, dict):
            other = RunnableParallel(other)
        elif callable(other) and not isinstance(other, Runnable):
            other = RunnableLambda(other)
        return RunnableSequence(other, self)


class RunnableSequence(Runnable):
    """Executes runnables in sequential pipe order: A | B | C."""

    def __init__(self, first: Runnable, second: Runnable):
        self.first = first
        self.second = second

    def invoke(self, input_data: Any) -> Any:
        intermediate = self.first.invoke(input_data)
        return self.second.invoke(intermediate)


class RunnableLambda(Runnable):
    """Wraps an arbitrary Python function into an LCEL Runnable."""

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)


class RunnablePassthrough(Runnable):
    """Passes through the input data unchanged (useful in RAG pipelines)."""

    def invoke(self, input_data: Any) -> Any:
        return input_data


class RunnableParallel(Runnable):
    """Runs every branch over the same input CONCURRENTLY (a thread per
    branch) and returns {branch_name: result} — the same semantics as
    LangChain's RunnableParallel, where branches are independent LLM calls."""

    def __init__(self, steps: Dict[str, Runnable]):
        self.steps = {
            k: (RunnableLambda(v) if callable(v) and not isinstance(v, Runnable) else v)
            for k, v in steps.items()
        }

    def invoke(self, input_data: Any) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=len(self.steps)) as pool:
            futures = {key: pool.submit(r.invoke, input_data) for key, r in self.steps.items()}
            return {key: f.result() for key, f in futures.items()}


class Timed(Runnable):
    """Wraps a runnable and records how long its last invoke took."""

    def __init__(self, bound: Runnable):
        self.bound, self.last_seconds = bound, 0.0

    def invoke(self, input_data: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self.bound.invoke(input_data)
        finally:
            self.last_seconds = time.perf_counter() - t0


class RunnableRetry(Runnable):
    """`.with_retry()`: re-invoke on failure with exponential backoff."""

    def __init__(self, bound: Runnable, attempts: int, retry_on: tuple):
        self.bound, self.attempts, self.retry_on = bound, attempts, retry_on

    def invoke(self, input_data: Any) -> Any:
        for attempt in range(1, self.attempts + 1):
            try:
                return self.bound.invoke(input_data)
            except self.retry_on:
                if attempt == self.attempts:
                    raise
                time.sleep(0.1 * 2 ** (attempt - 1))


class RunnableWithFallbacks(Runnable):
    """`.with_fallbacks()`: if the primary runnable raises, try each fallback in order."""

    def __init__(self, primary: Runnable, fallbacks: List[Runnable]):
        self.runnables = [primary] + fallbacks

    def invoke(self, input_data: Any) -> Any:
        last_error: Exception = RuntimeError("no runnables")
        for r in self.runnables:
            try:
                return r.invoke(input_data)
            except Exception as e:
                last_error = e
        raise last_error


# =====================================================================
# 2. PROMPT TEMPLATES, CHAT MODELS & OUTPUT PARSERS
# =====================================================================
class PromptTemplate(Runnable):
    """Formats a template string with provided keyword variables."""

    def __init__(self, template: str):
        self.template = template

    def invoke(self, input_data: Dict[str, Any]) -> str:
        return self.template.format(**input_data)


class ChatModelRunnable(Runnable):
    """Wraps LLMClient into the LCEL Runnable pipeline."""

    def __init__(self, client: LLMClient, simulated_response: str = None):
        self.client = client
        self.simulated_response = simulated_response

    def invoke(self, formatted_prompt: str) -> str:
        messages = [
            {"role": "user", "content": formatted_prompt}
        ]
        return self.client.chat(messages, simulated_response=self.simulated_response)


class StrOutputParser(Runnable):
    """Strips whitespace and parses raw LLM generation into a clean string."""

    def invoke(self, llm_output: str) -> str:
        return llm_output.strip()


# =====================================================================
# 3. RUNNABLE TESTS & LCEL PIPELINES
# =====================================================================
if __name__ == "__main__":
    client = LLMClient()

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("1. CLASSIC LCEL PIPE: PromptTemplate | ChatModel | StrOutputParser")
    print(f"===================================================={Colors.RESET}\n")

    prompt = PromptTemplate("Explain the concept of {concept} to a {audience} in 2 sentences.")
    model = ChatModelRunnable(
        client,
        simulated_response=(
            "A microservice is like a specialist chef in a kitchen who only makes pizzas. "
            "Instead of one cook doing everything, each chef does one job really fast and well."
        )
    )
    parser = StrOutputParser()

    # The Iconic LCEL Pipe Syntax!
    chain = prompt | model | parser

    result = chain.invoke({"concept": "Microservices", "audience": "10-year old"})
    print(f"{Colors.GREEN}Chain Result:{Colors.RESET}\n{result}\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("2. PARALLEL LCEL BRANCHING: RunnableParallel")
    print(f"===================================================={Colors.RESET}\n")

    # Construct a pipeline that splits execution into 2 parallel tracks:
    # Track A: Summarizer
    # Track B: Sentiment Analyzer
    summarizer = (
        PromptTemplate("Summarize this bug report: {text}")
        | ChatModelRunnable(client, simulated_response="NullPointerException on user login endpoint during peak load.")
        | parser
    )

    sentiment = (
        PromptTemplate("Classify urgency of this report: {text}")
        | ChatModelRunnable(client, simulated_response="URGENCY: P0 CRITICAL (Customer facing outage)")
        | parser
    )

    timed_branches = {"summary": Timed(summarizer), "urgency_rating": Timed(sentiment)}
    parallel_chain = RunnableParallel({**timed_branches, "raw_length": lambda x: len(x["text"])})

    test_input = {"text": "URGENT: All users in EU-Central are getting 500 errors when clicking the login submit button!"}
    t0 = time.perf_counter()
    parallel_output = parallel_chain.invoke(test_input)
    wall_clock = time.perf_counter() - t0

    print(f"{Colors.CYAN}Parallel Execution Output Dictionary:{Colors.RESET}")
    for k, v in parallel_output.items():
        print(f"  • {k:<15}: {v}")
    branch_times = [b.last_seconds for b in timed_branches.values()]
    print(f"\n  Branch latencies: {', '.join(f'{k}={b.last_seconds:.2f}s' for k, b in timed_branches.items())}")
    print(f"  Sequential would cost the SUM = {sum(branch_times):.2f}s | RunnableParallel wall-clock = {wall_clock:.2f}s "
          f"| slowest branch = {max(branch_times):.2f}s")
    print(f"  {Colors.DIM}Concurrency means wall-clock tracks the slowest branch, not the total — measured in one run,"
          f" so per-call latency variance can't distort the comparison.{Colors.RESET}\n")

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("3. .batch() — MANY INPUTS THROUGH ONE CHAIN, CONCURRENTLY")
    print(f"===================================================={Colors.RESET}\n")
    reports = [{"text": t} for t in ["Checkout page shows a blank screen", "Typo on the About page", "Password reset emails not sending"]]
    t0 = time.perf_counter()
    urgencies = sentiment.batch(reports, max_concurrency=3)
    print(f"  {len(reports)} reports classified in {time.perf_counter() - t0:.2f}s:")
    for r, u in zip(reports, urgencies):
        print(f"  • {r['text']:<38} -> {u[:70]}")
    print()

    print(f"{Colors.BOLD}{Colors.HEADER}====================================================")
    print("4. .with_retry() vs .with_fallbacks() — AGAINST A REAL FAILING BACKEND")
    print(f"===================================================={Colors.RESET}\n")
    if client.mode == "live":
        broken_model = ChatModelRunnable(LLMClient(model="unconfigured-provider/some-model", max_retries=1))
        broken_chain = prompt | broken_model | parser
        t0 = time.perf_counter()
        try:
            broken_chain.with_retry(attempts=3).invoke({"concept": "retries", "audience": "engineer"})
        except RuntimeError as e:
            print(f"  with_retry(3): still failed after 3 attempts in {time.perf_counter() - t0:.2f}s -> {str(e)[:70]}...")
            print(f"  {Colors.DIM}A 401 'no credentials' error is permanent — retrying it only adds latency.{Colors.RESET}")
        resilient = broken_chain.with_fallbacks([chain])
        print(f"  with_fallbacks([working chain]): {resilient.invoke({'concept': 'fallbacks', 'audience': 'engineer'})[:110]}...")
        print(f"  {Colors.GREEN}Rule of thumb: retry TRANSIENT errors (429/5xx/timeouts); fall back on PERMANENT ones.{Colors.RESET}")
    else:
        print("  (needs live mode — the failing backend is a real endpoint returning a real 401)")

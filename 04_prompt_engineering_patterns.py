"""
=============================================================================
04_prompt_engineering_patterns.py: Prompting Techniques, Measured
=============================================================================
Prompt engineering is an empirical discipline: a technique is only "better"
if it measurably improves accuracy on YOUR task at a cost you accept. This
module runs a small benchmark of word problems with known answers through
four techniques against the live model, and scores each on accuracy AND
tokens spent:

1. ZERO-SHOT (direct answer): just the question, forced to answer with a
   number and no working.
2. FEW-SHOT (in-context learning): worked examples before the question
   teach the expected format and approach.
3. ZERO-SHOT CHAIN-OF-THOUGHT (Kojima et al., 2022): "Let's think step by
   step" — the model writes intermediate reasoning before the answer.
4. SELF-CONSISTENCY (Wang et al., 2022): sample several CoT answers at a
   non-zero temperature and take the majority vote.

Also shown: prompt structure that makes answers machine-parseable (a system
role, delimiters, and a strict final-line format), and concurrent requests
with a thread pool so a benchmark doesn't run one call at a time.

Run directly:
  python3 04_prompt_engineering_patterns.py
=============================================================================
"""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


@dataclass
class Problem:
    question: str
    answer: float
    intuitive_wrong_answer: float  # what a hasty, no-reasoning answer tends to be


BENCHMARK = [
    Problem("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many dollars does the ball cost?", 0.05, 0.10),
    Problem("If 5 machines take 5 minutes to make 5 widgets, how many minutes would 100 machines take to make 100 widgets?", 5, 100),
    Problem("A lily pad patch doubles in size every day. It covers the whole lake in 48 days. On which day does it cover half the lake?", 47, 24),
    Problem("A shirt costs $80. It is discounted by 20%, then 10% sales tax is added to the discounted price. What is the final price in dollars?", 70.4, 72),
    Problem("A server handles 1,200 requests per minute. How many requests does it handle in 3.5 hours?", 252000, 4200),
    Problem("A store has 40 apples, sells 25% of them, then receives 15 more. How many apples does it have now?", 45, 55),
]

SYSTEM = "You are a careful math assistant. Always end your reply with a final line of the form 'Answer: <number>'."

FEW_SHOT_EXAMPLES = [
    ("A pen costs $2 more than a pencil. Together they cost $3. How many dollars does the pencil cost?",
     "Let the pencil cost p. Then the pen costs p + 2, so p + (p + 2) = 3, giving 2p = 1 and p = 0.5.\nAnswer: 0.5"),
    ("A tank drains 10% of its water every hour. It starts with 200 liters. How many liters remain after 2 hours?",
     "After 1 hour: 200 * 0.9 = 180. After 2 hours: 180 * 0.9 = 162.\nAnswer: 162"),
]


def build_messages(technique: str, question: str) -> List[Dict[str, str]]:
    if technique == "zero_shot":
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question:\n\"\"\"{question}\"\"\"\nReply with ONLY the final line 'Answer: <number>' and nothing else."}]
    if technique == "few_shot":
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEW_SHOT_EXAMPLES:
            msgs += [{"role": "user", "content": f"Question:\n\"\"\"{q}\"\"\""}, {"role": "assistant", "content": a}]
        return msgs + [{"role": "user", "content": f"Question:\n\"\"\"{question}\"\"\""}]
    # zero-shot CoT (also the per-sample prompt for self-consistency)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question:\n\"\"\"{question}\"\"\"\nLet's think step by step."}]


def extract_answer(text: str) -> Optional[float]:
    match = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not match:
        numbers = re.findall(r"-?[\d,]*\.?\d+", text)
        if not numbers:
            return None
        match = numbers
    try:
        return float(match[-1].replace(",", ""))
    except ValueError:
        return None


def simulated_reply(technique: str, problem: Problem, sample_idx: int) -> str:
    """Offline stand-in: a hasty direct answer falls for the intuitive trap;
    step-by-step reasoning gets it right. (Live mode ignores this.)"""
    if technique == "zero_shot":
        return f"Answer: {problem.intuitive_wrong_answer:g}"
    if technique == "self_consistency" and sample_idx == 2:
        return f"Quick estimate.\nAnswer: {problem.intuitive_wrong_answer:g}"
    return f"Working through it step by step...\nAnswer: {problem.answer:g}"


def run_one(technique: str, problem: Problem, temperature: float, sample_idx: int = 0) -> Tuple[Optional[float], int]:
    client = LLMClient()  # one client per thread, so `last_usage` isn't shared across threads
    reply = client.chat(
        build_messages(technique, problem.question),
        temperature=temperature, max_tokens=700,
        simulated_response=simulated_reply(technique, problem, sample_idx),
    )
    tokens = (client.last_usage or {}).get("completion_tokens", len(reply.split()))
    return extract_answer(reply), tokens


def is_correct(predicted: Optional[float], expected: float) -> bool:
    return predicted is not None and abs(predicted - expected) < 1e-6 * max(1.0, abs(expected))


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 04: Prompt Engineering Patterns, Measured ==={Colors.RESET}\n")
    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET} | benchmark: {len(BENCHMARK)} problems with known answers\n")

    print(f"{Colors.YELLOW}1. What each technique actually sends (few-shot shown){Colors.RESET}")
    for m in build_messages("few_shot", BENCHMARK[0].question):
        print(f"  [{m['role']:<9}] {m['content'][:95].replace(chr(10), ' ')}{'...' if len(m['content']) > 95 else ''}")
    print()

    SELF_CONSISTENCY_SAMPLES = 3
    jobs = []
    for i, p in enumerate(BENCHMARK):
        jobs += [("zero_shot", i, 0.0, 0), ("few_shot", i, 0.0, 0), ("cot", i, 0.0, 0)]
        jobs += [("self_consistency", i, 0.8, s) for s in range(SELF_CONSISTENCY_SAMPLES)]

    print(f"{Colors.YELLOW}2. Running {len(jobs)} calls concurrently (thread pool, 6 workers)...{Colors.RESET}")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {job: pool.submit(run_one, job[0], BENCHMARK[job[1]], job[2], job[3]) for job in jobs}
        results = {job: fut.result() for job, fut in futures.items()}

    print(f"\n{Colors.YELLOW}3. Results{Colors.RESET}")
    header = f"  {'Problem':<45} {'zero-shot':>10} {'few-shot':>10} {'CoT':>10} {'self-cons.':>11}"
    print(header)
    scores = Counter()
    tokens_spent = Counter()
    for i, p in enumerate(BENCHMARK):
        row = []
        for tech in ("zero_shot", "few_shot", "cot"):
            pred, tok = results[(tech, i, 0.0, 0)]
            ok = is_correct(pred, p.answer)
            scores[tech] += ok
            tokens_spent[tech] += tok
            row.append(f"{Colors.GREEN if ok else Colors.RED}{('✔' if ok else '✖')} {(f'{pred:g}' if pred is not None else '—'):<7}{Colors.RESET}")
        votes = [results[("self_consistency", i, 0.8, s)] for s in range(SELF_CONSISTENCY_SAMPLES)]
        tokens_spent["self_consistency"] += sum(t for _, t in votes)
        majority = Counter(round(v, 6) for v, _ in votes if v is not None).most_common(1)
        voted = majority[0][0] if majority else None
        ok = is_correct(voted, p.answer)
        scores["self_consistency"] += ok
        row.append(f"{Colors.GREEN if ok else Colors.RED}{('✔' if ok else '✖')} {(f'{voted:g}' if voted is not None else '—'):<8}{Colors.RESET}")
        print(f"  {p.question[:43] + '..':<45} " + " ".join(row))

    print(f"\n  {'Technique':<18}{'Accuracy':>10}{'Output tokens':>16}")
    for tech in ("zero_shot", "few_shot", "cot", "self_consistency"):
        print(f"  {tech:<18}{scores[tech]:>6}/{len(BENCHMARK)}{tokens_spent[tech]:>16,}")
    print(f"\n  {Colors.DIM}Read the tradeoff, not just the accuracy: reasoning techniques spend many more output tokens.{Colors.RESET}")
    print(f"  {Colors.DIM}Strong models often solve these even zero-shot; the gap is widest on smaller/cheaper models.{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Four prompting techniques benchmarked on accuracy and token cost.{Colors.RESET}")

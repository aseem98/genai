"""
=============================================================================
03_sampling_and_decoding_strategies.py: How an LLM Picks the Next Token
=============================================================================
At every step a language model outputs one LOGIT (unnormalized score) per
vocabulary token. The decoding strategy turns those logits into the text
you see. This module trains a real (small) language model — a word-level
bigram model with add-k smoothing — on this curriculum's docs, so every
logit below is learned from data, then implements each decoding knob you'll
find in an API request:

1. TEMPERATURE: softmax(z / T). T -> 0 approaches greedy argmax; T > 1
   flattens the distribution (more diverse, more errors).
2. TOP-K: keep only the K highest-probability tokens, renormalize.
3. TOP-P (nucleus): keep the smallest set whose cumulative probability
   reaches P — adapts to how confident the model is at each step.
4. FREQUENCY & PRESENCE PENALTIES: subtract from a token's logit based on
   how often (frequency) / whether (presence) it already appeared — exactly
   OpenAI's documented formula.
5. GREEDY vs BEAM SEARCH: deterministic decoding; beam search keeps the B
   best partial sequences by total log-probability instead of one.

Finally the same knobs on the live model: temperature 0 vs 1.3, sampled
several times, measuring how many distinct answers come back.

Run directly:
  python3 03_sampling_and_decoding_strategies.py
=============================================================================
"""

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
load_training_corpus = import_module("02_tokenization_bpe_from_scratch").load_training_corpus


# =====================================================================
# 1. A REAL LANGUAGE MODEL: word bigram with add-k smoothing
# =====================================================================
class BigramLanguageModel:
    """Add-k smoothing gives every unseen (prev, next) pair a pseudo-count
    of k, so nothing has zero probability. Keep k small: the total
    pseudo-mass per context is k * vocab_size, and for a context seen only a
    few times a large k (e.g. 0.1 * 2,000 words = 200 fake counts) swamps the
    real data and makes every rare context look maximally uncertain."""

    def __init__(self, k: float = 0.001):
        self.k = k
        self.counts: Dict[str, Counter] = defaultdict(Counter)
        self.vocab: List[str] = []

    def train(self, text: str):
        text = re.sub(r"\S*[/\\]\S*|\S+\.(py|md|json|sqlite3)\b", " ", text)  # drop paths/filenames
        tokens = re.findall(r"[a-z]+|[.,]", text.lower())
        for prev, nxt in zip(tokens, tokens[1:]):
            self.counts[prev][nxt] += 1
        self.vocab = sorted(set(tokens))

    def logits(self, prev: str) -> Dict[str, float]:
        """log P(next | prev) for every vocab word — the model's raw output."""
        row = self.counts.get(prev, Counter())
        total = sum(row.values()) + self.k * len(self.vocab)
        return {w: math.log((row.get(w, 0) + self.k) / total) for w in self.vocab}


# =====================================================================
# 2. DECODING KNOBS
# =====================================================================
def softmax(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    t = max(temperature, 1e-6)
    scaled = {w: z / t for w, z in logits.items()}
    m = max(scaled.values())
    exps = {w: math.exp(z - m) for w, z in scaled.items()}  # subtract max for numerical stability
    total = sum(exps.values())
    return {w: e / total for w, e in exps.items()}


def top_k_filter(probs: Dict[str, float], k: int) -> Dict[str, float]:
    kept = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k])
    total = sum(kept.values())
    return {w: p / total for w, p in kept.items()}


def top_p_filter(probs: Dict[str, float], p: float) -> Dict[str, float]:
    kept, cumulative = {}, 0.0
    for w, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        kept[w] = prob
        cumulative += prob
        if cumulative >= p:
            break
    total = sum(kept.values())
    return {w: q / total for w, q in kept.items()}


def apply_penalties(logits: Dict[str, float], generated: List[str], frequency_penalty: float, presence_penalty: float) -> Dict[str, float]:
    """OpenAI's formula: logit[j] -= count[j] * frequency_penalty + (count[j] > 0) * presence_penalty"""
    counts = Counter(generated)
    return {w: z - counts[w] * frequency_penalty - (1.0 if counts[w] > 0 else 0.0) * presence_penalty for w, z in logits.items()}


def sample(probs: Dict[str, float], rng: random.Random) -> str:
    r, cumulative = rng.random(), 0.0
    for w, p in probs.items():
        cumulative += p
        if r <= cumulative:
            return w
    return w


def generate(
    model: BigramLanguageModel, start: str, length: int, rng: random.Random,
    temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
    frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
) -> List[str]:
    out = [start]
    for _ in range(length):
        logits = apply_penalties(model.logits(out[-1]), out, frequency_penalty, presence_penalty)
        if temperature == 0:
            out.append(max(logits, key=logits.get))  # greedy
            continue
        probs = softmax(logits, temperature)
        if top_k:
            probs = top_k_filter(probs, top_k)
        if top_p:
            probs = top_p_filter(probs, top_p)
        out.append(sample(probs, rng))
    return out


def beam_search(model: BigramLanguageModel, start: str, length: int, beam_width: int) -> Tuple[List[str], float]:
    beams = [([start], 0.0)]  # (sequence, total log-prob)
    for _ in range(length):
        candidates = []
        for seq, score in beams:
            for w, logp in sorted(model.logits(seq[-1]).items(), key=lambda x: x[1], reverse=True)[:beam_width]:
                candidates.append((seq + [w], score + logp))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0]


def sequence_logprob(model: BigramLanguageModel, seq: List[str]) -> float:
    return sum(model.logits(a)[b] for a, b in zip(seq, seq[1:]))


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 03: Sampling & Decoding Strategies ==={Colors.RESET}\n")

    model = BigramLanguageModel(k=0.001)
    model.train(load_training_corpus())
    print(f"{Colors.YELLOW}1. A real bigram LM trained on the curriculum docs — vocab {len(model.vocab):,} words{Colors.RESET}")
    probs = softmax(model.logits("the"))
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top next-word probabilities after 'the': {[(w, round(p, 3)) for w, p in top5]}\n")

    print(f"{Colors.YELLOW}2. Temperature reshapes the SAME logits{Colors.RESET}")
    for t in (0.3, 1.0, 2.0):
        dist = softmax(model.logits("the"), t)
        top = max(dist.values())
        entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
        print(f"  T={t:<4} top-token prob={top:.3f}  entropy={entropy:5.2f} bits")
    print()

    print(f"{Colors.YELLOW}3. Top-k vs top-p: how many candidate words survive filtering after 'the'{Colors.RESET}")
    base = softmax(model.logits("the"), 1.0)
    print(f"  no filter: {len(base):,} | top_k=20: {len(top_k_filter(base, 20))} | "
          f"top_p=0.5: {len(top_p_filter(base, 0.5))} | top_p=0.9: {len(top_p_filter(base, 0.9))}")
    # Find, from the data, the frequent context word the model is MOST sure about.
    frequent = [w for w, row in model.counts.items() if sum(row.values()) >= 10]
    nucleus_size = {w: len(top_p_filter(softmax(model.logits(w)), 0.9)) for w in frequent}
    confident_word = min(nucleus_size, key=nucleus_size.get)
    print(f"  after '{confident_word}' (the model's most predictable frequent context), top_p=0.9 keeps only "
          f"{nucleus_size[confident_word]} vs {len(top_p_filter(base, 0.9))} after 'the'")
    print(f"  — nucleus sampling adapts its candidate count to the model's confidence; a fixed top_k can't.\n")

    print(f"{Colors.YELLOW}4. Generating text with different settings (seeded, so reproducible){Colors.RESET}")
    settings = [
        ("greedy (T=0)", dict(temperature=0)),
        ("T=0.7, top_p=0.9", dict(temperature=0.7, top_p=0.9)),
        ("T=1.5 (too hot)", dict(temperature=1.5)),
        ("greedy + frequency_penalty=2.0", dict(temperature=0, frequency_penalty=2.0)),
    ]
    for label, kwargs in settings:
        words = generate(model, "the", 14, random.Random(7), **kwargs)
        repeats = len(words) - len(set(words))
        print(f"  {label:<31} repeats={repeats:<2} | {' '.join(words)}")
    print(f"  {Colors.DIM}Greedy decoding loops on high-frequency phrases; the frequency penalty breaks the loop.{Colors.RESET}\n")

    print(f"{Colors.YELLOW}5. Greedy vs beam search — beam finds a higher total-probability sequence{Colors.RESET}")
    greedy_seq = generate(model, "the", 6, random.Random(0), temperature=0)
    beam_seq, beam_score = beam_search(model, "the", 6, beam_width=5)
    print(f"  greedy: {' '.join(greedy_seq):<55} log P = {sequence_logprob(model, greedy_seq):.2f}")
    print(f"  beam=5: {' '.join(beam_seq):<55} log P = {beam_score:.2f}\n")

    llm = LLMClient()
    print(f"{Colors.YELLOW}6. The same knob on the live model (Active Mode: {llm.mode.upper()}) — 4 samples each{Colors.RESET}")
    prompt = [{"role": "user", "content": "Give a two-word name for a new coffee shop. Reply with only the name."}]
    for t in (0.0, 1.3):
        answers = [
            llm.chat(prompt, temperature=t, max_tokens=12,
                     simulated_response=("Bean There" if t == 0 else random.Random(i).choice(["Bean There", "Daily Grind", "Brew Haven", "Roast Rebels"]))).strip()
            for i in range(4)
        ]
        print(f"  T={t}: {len(set(answers))} distinct of 4 -> {answers}")

    print(f"\n{Colors.GREEN}✔ Every decoding strategy implemented and run on a real trained language model.{Colors.RESET}")

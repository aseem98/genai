"""
=============================================================================
02_tokenization_bpe_from_scratch.py: Byte-Pair Encoding, Trained for Real
=============================================================================
An LLM never sees characters or words — it sees integer TOKEN IDs produced
by a tokenizer. GPT-2/3/4, Llama 3 and most modern models use byte-level
Byte-Pair Encoding (BPE). This module trains one from scratch on real text
(this curriculum's own README and module docstrings) and uses it:

1. BASE VOCABULARY = the 256 possible bytes, so ANY string (emoji, Hindi,
   code) is encodable — there is no "unknown token".
2. PRE-TOKENIZATION: a regex splits text into word-ish pieces first, so
   merges never cross word boundaries (the GPT-2 approach).
3. TRAINING: repeatedly find the most frequent adjacent pair of token ids
   and merge it into a brand-new token id. Each merge is recorded in order.
4. ENCODING: apply the learned merges, lowest (earliest) rank first.
5. DECODING: concatenate each token's bytes and UTF-8 decode — the round
   trip `decode(encode(x)) == x` holds for every input.

Then the practical consequences: context-window/cost arithmetic with real
token counts, why models struggle to count letters in a word, and a
comparison against the live API's own reported `prompt_tokens`.

Run directly:
  python3 02_tokenization_bpe_from_scratch.py
=============================================================================
"""

import glob
import os
import re
import time
from collections import Counter
from typing import Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

# GPT-2's pre-tokenization pattern, simplified for Python's `re` (no \p{L}):
# contractions, runs of letters, runs of digits, runs of punctuation, spaces.
PRETOKENIZE_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


class BPETokenizer:
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}      # (id_a, id_b) -> new id, in learned order
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _pair_counts(word_freqs: Dict[Tuple[int, ...], int]) -> Counter:
        counts: Counter = Counter()
        for ids, freq in word_freqs.items():
            for pair in zip(ids, ids[1:]):
                counts[pair] += freq
        return counts

    @staticmethod
    def _merge(ids: Tuple[int, ...], pair: Tuple[int, int], new_id: int) -> Tuple[int, ...]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return tuple(out)

    def train(self, text: str, vocab_size: int) -> List[Tuple[bytes, int]]:
        """Learns `vocab_size - 256` merges. Works on a {pre-token: count}
        table instead of the raw text, so each merge pass touches every
        UNIQUE word once rather than every occurrence."""
        word_freqs: Dict[Tuple[int, ...], int] = Counter(
            tuple(piece.encode("utf-8")) for piece in PRETOKENIZE_PATTERN.findall(text)
        )
        history = []
        for new_id in range(256, vocab_size):
            counts = self._pair_counts(word_freqs)
            if not counts:
                break
            best_pair, best_count = counts.most_common(1)[0]
            if best_count < 2:
                break  # nothing repeats anymore — further merges would just memorize
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            word_freqs = {self._merge(ids, best_pair, new_id): f for ids, f in word_freqs.items()}
            history.append((self.vocab[new_id], best_count))
        return history

    def _encode_piece(self, piece: str) -> List[int]:
        ids = tuple(piece.encode("utf-8"))
        while len(ids) >= 2:
            # Pick the adjacent pair that was learned EARLIEST (lowest rank).
            ranked = [(self.merges.get(p, float("inf")), p) for p in zip(ids, ids[1:])]
            rank, pair = min(ranked)
            if rank == float("inf"):
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode(self, text: str) -> List[int]:
        return [i for piece in PRETOKENIZE_PATTERN.findall(text) for i in self._encode_piece(piece)]

    def decode(self, ids: List[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, text: str) -> List[str]:
        """Human-readable tokens. A token holding only PART of a multi-byte
        UTF-8 character isn't valid text on its own, so it's shown as hex."""
        out = []
        for i in self.encode(text):
            raw = self.vocab[i]
            try:
                out.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("".join(f"<0x{b:02X}>" for b in raw))
        return out


def load_training_corpus() -> str:
    """Real training text: every module docstring in this folder, plus the
    README — with decorative `=====` banner lines stripped, since otherwise
    the tokenizer spends its first merges memorizing runs of '=' (a small
    lesson in why tokenizer training data is cleaned first)."""
    here = os.path.dirname(os.path.abspath(__file__))
    parts = []
    for path in sorted(glob.glob(os.path.join(here, "[0-9][0-9]_*.py"))):
        src = open(path, encoding="utf-8").read()
        match = re.search(r'"""(.*?)"""', src, re.DOTALL)
        if match:
            parts.append(match.group(1))
    readme = os.path.join(here, "README.md")
    if os.path.exists(readme):
        parts.append(open(readme, encoding="utf-8").read())
    text = "\n".join(parts)
    boilerplate = re.compile(r"\s*([=\-]{5,}|Run directly:|python3 \S+\.py)\s*")
    return "\n".join(line for line in text.split("\n") if not boilerplate.fullmatch(line))


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 02: Byte-Pair Encoding, Trained From Scratch ==={Colors.RESET}\n")

    corpus = load_training_corpus()
    print(f"{Colors.YELLOW}1. Training on this curriculum's own docs ({len(corpus):,} chars){Colors.RESET}")
    tokenizer = BPETokenizer()
    t0 = time.time()
    history = tokenizer.train(corpus, vocab_size=512)
    print(f"  Learned {len(history)} merges in {time.time() - t0:.1f}s -> vocab size {len(tokenizer.vocab)}")
    print(f"  First 10 merges (most frequent pairs):   {[tok.decode('utf-8', 'replace') for tok, _ in history[:10]]}")
    print(f"  Last 10 merges (longer, rarer subwords): {[tok.decode('utf-8', 'replace') for tok, _ in history[-10:]]}\n")

    print(f"{Colors.YELLOW}2. Encoding — known words become 1 token, unseen words split into subwords{Colors.RESET}")
    for text in ["The agent retrieves context.", "Pharmacovigilance signal detection", "नमस्ते 🚀"]:
        ids = tokenizer.encode(text)
        print(f"  {text!r}")
        print(f"    tokens ({len(ids)}): {tokenizer.token_strings(text)}")
        assert tokenizer.decode(ids) == text, "round-trip failed"
    print(f"  {Colors.GREEN}✔ decode(encode(x)) == x for all inputs, including Hindi and emoji (byte-level base vocab).{Colors.RESET}\n")

    print(f"{Colors.YELLOW}3. Compression — why a bigger vocabulary means fewer tokens per request{Colors.RESET}")
    sample = corpus[:5000]
    for size in (300, 400, 512):
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=size)
        n = len(tok.encode(sample))
        print(f"  vocab={size:<4} -> {n:>5} tokens for 5,000 chars  ({len(sample.encode()) / n:.2f} bytes/token)")
    print(f"  {Colors.DIM}(GPT-4's ~100k-token vocabulary reaches ~4 bytes/token on English text){Colors.RESET}\n")

    print(f"{Colors.YELLOW}4. Why LLMs miscount letters: the model sees tokens, not characters{Colors.RESET}")
    word = "strawberry"
    print(f"  {word!r} -> {tokenizer.token_strings(word)}")
    print(f"  Counting the 'r's requires knowing the spelling INSIDE each token id — information the model")
    print(f"  only has indirectly, which is why letter-counting questions are a classic failure case.\n")

    print(f"{Colors.YELLOW}5. Context-window & cost arithmetic with real token counts{Colors.RESET}")
    doc = corpus[:20000]
    doc_tokens = len(tokenizer.encode(doc))
    window, price_per_m = 8192, 3.00
    print(f"  A 20,000-char document = {doc_tokens:,} tokens with this tokenizer")
    if doc_tokens <= window:
        print(f"  Fits in one {window:,}-token context window with {window - doc_tokens:,} tokens left for the answer")
    else:
        print(f"  Exceeds an {window:,}-token context window -> needs {-(-doc_tokens // window)} chunks (see module 08)")
    print(f"  Input cost at ${price_per_m}/1M tokens: ${doc_tokens / 1e6 * price_per_m:.5f}\n")

    llm = LLMClient()
    print(f"{Colors.YELLOW}6. Compare against the live model's own tokenizer (Active Mode: {llm.mode.upper()}){Colors.RESET}")
    probe = "Summarize the key risks of deploying an autonomous agent to production."
    llm.chat([{"role": "user", "content": probe}], simulated_response="(offline)", max_tokens=5)
    ours = len(tokenizer.encode(probe))
    if llm.last_usage:
        print(f"  Our tokenizer: {ours} tokens | API-reported prompt_tokens: {llm.last_usage.get('prompt_tokens')}")
        print(f"  {Colors.DIM}(The API count also includes chat-template/role tokens, and its vocabulary is ~200x larger.){Colors.RESET}")
    else:
        print(f"  Our tokenizer: {ours} tokens (API token counts need live mode)")

    print(f"\n{Colors.GREEN}✔ Real BPE tokenizer trained, verified lossless, and used for context/cost math.{Colors.RESET}")

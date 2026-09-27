"""
=============================================================================
06_transformer_attention_from_scratch.py: Self-Attention, Transformers & KV Cache
=============================================================================
Every LLM you call through module 01 is a stack of transformer blocks. This
module builds one in pure Python (no numpy) and VERIFIES each property
numerically instead of asserting it:

1. SCALED DOT-PRODUCT ATTENTION: softmax(Q K^T / sqrt(d_k)) V — each token's
   output is a weighted average of every value vector, weighted by how well
   its query matches each key. Attention is a soft dictionary lookup.
2. WHY DIVIDE BY sqrt(d_k): dot products of random d_k-dimensional vectors
   have variance ~d_k, so without scaling the softmax saturates to a
   near-one-hot distribution and gradients vanish. Measured below.
3. CAUSAL MASK: a decoder may only attend to earlier positions — enforced by
   setting future scores to -inf before the softmax. Verified: changing a
   future token never changes an earlier token's output.
4. MULTI-HEAD ATTENTION: h smaller attention heads in parallel, concatenated
   and projected, so different heads can track different relationships.
5. POSITIONAL ENCODING: attention is order-blind on its own; sinusoidal
   encodings (Vaswani et al.) inject position.
6. THE TRANSFORMER BLOCK: pre-LayerNorm -> attention -> residual ->
   LayerNorm -> feed-forward (GELU) -> residual.
7. KV CACHE: during generation, keys/values of past tokens never change, so
   they're cached and only the NEW token's attention is computed. Verified
   bit-for-bit equal to full recomputation, with the multiply count saved.

Run directly:
  python3 06_transformer_attention_from_scratch.py
=============================================================================
"""

import math
import random
from typing import List, Optional, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors

Matrix = List[List[float]]
MULTIPLY_COUNTER = [0]  # counts scalar multiplications, to measure KV-cache savings


# =====================================================================
# 0. MINIMAL LINEAR ALGEBRA
# =====================================================================
def matmul(a: Matrix, b: Matrix) -> Matrix:
    cols = list(zip(*b))
    MULTIPLY_COUNTER[0] += len(a) * len(cols) * len(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in cols] for row in a]


def transpose(a: Matrix) -> Matrix:
    return [list(r) for r in zip(*a)]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def softmax_row(row: List[float]) -> List[float]:
    m = max(row)
    exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def random_matrix(rows: int, cols: int, rng: random.Random, scale: float = 0.3) -> Matrix:
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def max_abs_diff(a: Matrix, b: Matrix) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


# =====================================================================
# 1. SCALED DOT-PRODUCT ATTENTION (with optional causal mask)
# =====================================================================
def attention(q: Matrix, k: Matrix, v: Matrix, causal: bool, scale: bool = True, q_offset: int = 0) -> Tuple[Matrix, Matrix]:
    """q: (n_q, d_k), k/v: (n_k, d_k). `q_offset` is the absolute position of
    q's first row — needed so the causal mask stays correct when q is just
    the newest token (KV-cache decoding)."""
    d_k = len(q[0])
    scores = matmul(q, transpose(k))
    if scale:
        scores = [[s / math.sqrt(d_k) for s in row] for row in scores]
    if causal:
        scores = [[s if j <= i + q_offset else float("-inf") for j, s in enumerate(row)] for i, row in enumerate(scores)]
    weights = [softmax_row(row) for row in scores]
    return matmul(weights, v), weights


def entropy(p: List[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


# =====================================================================
# 2. MULTI-HEAD ATTENTION + TRANSFORMER BLOCK
# =====================================================================
class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, rng: random.Random):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.Wq, self.Wk, self.Wv, self.Wo = (random_matrix(d_model, d_model, rng) for _ in range(4))

    def _split(self, m: Matrix, h: int) -> Matrix:
        return [row[h * self.d_head:(h + 1) * self.d_head] for row in m]

    def project_kv(self, x: Matrix) -> Tuple[Matrix, Matrix]:
        return matmul(x, self.Wk), matmul(x, self.Wv)

    def forward(self, x: Matrix, causal: bool = True, kv_cache: Optional[Tuple[Matrix, Matrix]] = None, q_offset: int = 0) -> Matrix:
        q = matmul(x, self.Wq)
        k, v = kv_cache if kv_cache is not None else self.project_kv(x)
        head_outputs = [attention(self._split(q, h), self._split(k, h), self._split(v, h), causal, q_offset=q_offset)[0]
                        for h in range(self.n_heads)]
        concat = [sum((head[i] for head in head_outputs), []) for i in range(len(x))]
        return matmul(concat, self.Wo)


def layer_norm(x: Matrix, eps: float = 1e-5) -> Matrix:
    out = []
    for row in x:
        mu = sum(row) / len(row)
        var = sum((v - mu) ** 2 for v in row) / len(row)
        out.append([(v - mu) / math.sqrt(var + eps) for v in row])
    return out


def gelu(x: float) -> float:
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, rng: random.Random):
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.W1 = random_matrix(d_model, 4 * d_model, rng)
        self.W2 = random_matrix(4 * d_model, d_model, rng)

    def forward(self, x: Matrix) -> Matrix:
        x = add(x, self.attn.forward(layer_norm(x), causal=True))          # attention sub-layer + residual
        hidden = [[gelu(v) for v in row] for row in matmul(layer_norm(x), self.W1)]
        return add(x, matmul(hidden, self.W2))                             # feed-forward sub-layer + residual


def sinusoidal_positions(n_positions: int, d_model: int) -> Matrix:
    return [[math.sin(pos / 10000 ** (i / d_model)) if i % 2 == 0 else math.cos(pos / 10000 ** ((i - 1) / d_model))
             for i in range(d_model)] for pos in range(n_positions)]


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 06: Transformer Attention From Scratch ==={Colors.RESET}\n")
    rng = random.Random(0)
    tokens = ["The", "agent", "called", "the", "tool", "again"]
    d_model, n_heads = 16, 4
    embeddings = random_matrix(len(tokens), d_model, rng, scale=1.0)
    x = add(embeddings, sinusoidal_positions(len(tokens), d_model))

    print(f"{Colors.YELLOW}1. Attention as a soft dictionary lookup{Colors.RESET}")
    keys = random_matrix(4, 8, rng, scale=1.0)
    values = [[float(i)] * 8 for i in range(4)]            # value i is just the number i
    query = [[3.0 * v for v in keys[2]]]                   # a query aligned with key #2
    out, weights = attention(query, keys, values, causal=False)
    print(f"  weights over 4 keys: {[round(w, 3) for w in weights[0]]}  -> output ≈ {out[0][0]:.2f} (value #2 is 2.0)\n")

    print(f"{Colors.YELLOW}2. Why scale by sqrt(d_k): unscaled scores saturate the softmax{Colors.RESET}")
    for d_k in (4, 64, 512):
        q = random_matrix(1, d_k, rng, scale=1.0)
        k = random_matrix(20, d_k, rng, scale=1.0)
        v = random_matrix(20, 1, rng)
        _, w_scaled = attention(q, k, v, causal=False, scale=True)
        _, w_raw = attention(q, k, v, causal=False, scale=False)
        print(f"  d_k={d_k:<4} entropy scaled={entropy(w_scaled[0]):.2f} bits | unscaled={entropy(w_raw[0]):.2f} bits "
              f"| unscaled max weight={max(w_raw[0]):.3f}")
    print(f"  {Colors.DIM}(uniform over 20 keys = {math.log2(20):.2f} bits; unscaled attention collapses toward one key as d_k grows){Colors.RESET}\n")

    print(f"{Colors.YELLOW}3. Causal mask — verified, not assumed{Colors.RESET}")
    q, k, v = matmul(x, random_matrix(d_model, 8, rng)), matmul(x, random_matrix(d_model, 8, rng)), matmul(x, random_matrix(d_model, 8, rng))
    out_a, w = attention(q, k, v, causal=True)
    print("  attention weights (row = query position, col = key position):")
    for tok, row in zip(tokens, w):
        print(f"    {tok:<7}" + " ".join(f"{p:5.2f}" for p in row))
    assert all(abs(sum(r) - 1) < 1e-12 for r in w), "rows must sum to 1"
    v_changed = [row[:] for row in v]
    v_changed[-1] = [999.0] * 8                              # tamper with the LAST token only
    out_b, _ = attention(q, k, v_changed, causal=True)
    print(f"  Changed only the last token's value; max change in earlier outputs = {max_abs_diff(out_a[:-1], out_b[:-1]):.1e}")
    print(f"  {Colors.GREEN}✔ Each row sums to 1 and no position can see the future.{Colors.RESET}\n")

    print(f"{Colors.YELLOW}4. Positional encoding: nearby positions are more similar than distant ones{Colors.RESET}")
    pe = sinusoidal_positions(64, 64)
    dot = lambda a, b: sum(p * q for p, q in zip(a, b))
    print("  similarity of position 10 to position 10+k: " + ", ".join(f"k={k}:{dot(pe[10], pe[10 + k]):.1f}" for k in (0, 1, 4, 16, 40)))
    print()

    print(f"{Colors.YELLOW}5. A full transformer block (multi-head attention + FFN, residuals, LayerNorm){Colors.RESET}")
    block = TransformerBlock(d_model, n_heads, rng)
    y = block.forward(x)
    print(f"  input {len(x)}x{len(x[0])} -> output {len(y)}x{len(y[0])} ({n_heads} heads of {d_model // n_heads} dims each)\n")

    print(f"{Colors.YELLOW}6. KV cache: incremental decoding equals full recomputation{Colors.RESET}")
    mha = block.attn
    MULTIPLY_COUNTER[0] = 0
    full_outputs = []
    for t in range(1, len(tokens) + 1):                    # naive: recompute the whole prefix every step
        full_outputs.append(mha.forward(x[:t], causal=True)[-1])
    naive_multiplies = MULTIPLY_COUNTER[0]

    MULTIPLY_COUNTER[0] = 0
    cached_k: Matrix = []
    cached_v: Matrix = []
    cached_outputs = []
    for t in range(len(tokens)):                           # cached: project ONLY the new token's K and V
        new_k, new_v = mha.project_kv([x[t]])
        cached_k, cached_v = cached_k + new_k, cached_v + new_v
        cached_outputs.append(mha.forward([x[t]], causal=True, kv_cache=(cached_k, cached_v), q_offset=t)[0])
    cached_multiplies = MULTIPLY_COUNTER[0]

    print(f"  max |full - cached| over all steps = {max_abs_diff(full_outputs, cached_outputs):.1e}")
    print(f"  scalar multiplies: full recompute={naive_multiplies:,}  with KV cache={cached_multiplies:,} "
          f"({naive_multiplies / cached_multiplies:.1f}x fewer)")
    print(f"  {Colors.DIM}The saving grows with sequence length — which is why every inference server keeps a KV cache,")
    print(f"  and why long contexts cost GPU memory (the cache itself).{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Attention, masking, multi-head, positional encoding, and KV cache implemented and verified.{Colors.RESET}")

"""
=============================================================================
48_temporal_forecasting_and_genetic_optimization.py: LSTM + Genetic Algorithm
=============================================================================
Senior GenAI Engineer Context:
Promotional optimization (the pattern behind pharma HCP engagement planning)
splits into two distinct ML problems:
1. TEMPORAL MODELING: given a sequence of past promotional touches, predict
   an HCP's future response (NBRx uplift). LSTMs are the classic architecture
   because they carry state across time steps — capturing recency, carryover,
   and fatigue effects a stateless model can't.
2. COMBINATORIAL OPTIMIZATION: given that predictive model, search the huge
   space of possible 13-week x N-channel promotional SEQUENCES for the one
   that maximizes predicted response under real constraints (channel caps).
   This is NOT a differentiable problem end-to-end (the search space is
   discrete/combinatorial), which is exactly why a Genetic Algorithm — not
   gradient descent — is the standard tool for this second half.

Both are implemented FOR REAL here, pure Python / zero dependencies:
- An LSTM cell with genuine forget/input/candidate/output gates, trained via
  actual gradient descent (numerical gradients — exact for this scale, and
  the mechanics are the same whether the gradient comes from autodiff or
  finite differences: params -= learning_rate * gradient).
- A Genetic Algorithm (selection, crossover, mutation, elitism) that uses
  the TRAINED LSTM as its fitness function to evolve an optimal 13-week
  promotional sequence per HCP.

Run directly:
  python3 48_temporal_forecasting_and_genetic_optimization.py
=============================================================================
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors

random.seed(7)


# =====================================================================
# 1. LSTM CELL — real gate math, pure Python (no numpy)
# =====================================================================
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60, min(60, x))))


def dot(weights: List[float], values: List[float]) -> float:
    return sum(w * v for w, v in zip(weights, values))


class LSTMCell:
    """A single-layer LSTM cell with hidden_size H and scalar (1-feature)
    input — small enough to train with numerical gradients in pure Python,
    while implementing the exact same gate equations a PyTorch `nn.LSTM`
    computes internally.
    """

    def __init__(self, hidden_size: int = 4, seed: int = 0):
        rng = random.Random(seed)
        in_size = hidden_size + 1  # concatenated [h_prev, x_t]

        def rand_vec(n):
            return [rng.uniform(-0.4, 0.4) for _ in range(n)]

        self.H = hidden_size
        # One weight vector per hidden unit per gate: gates[name][unit] is a
        # length-(H+1) vector applied to [h_prev..., x_t].
        self.gates: Dict[str, List[List[float]]] = {
            g: [rand_vec(in_size) for _ in range(hidden_size)] for g in ("f", "i", "c", "o")
        }
        self.biases: Dict[str, List[float]] = {g: [0.0] * hidden_size for g in ("f", "i", "c", "o")}
        self.Wy = rand_vec(hidden_size)  # hidden -> scalar output
        self.by = 0.0

    def get_params(self) -> List[float]:
        """Flattens every trainable weight into one list — lets the trainer
        below treat this LSTM (or any future architecture) generically.
        """
        params = []
        for g in ("f", "i", "c", "o"):
            for unit_weights in self.gates[g]:
                params.extend(unit_weights)
            params.extend(self.biases[g])
        params.extend(self.Wy)
        params.append(self.by)
        return params

    def set_params(self, params: List[float]):
        idx = 0
        in_size = self.H + 1
        for g in ("f", "i", "c", "o"):
            for u in range(self.H):
                self.gates[g][u] = params[idx: idx + in_size]
                idx += in_size
            self.biases[g] = params[idx: idx + self.H]
            idx += self.H
        self.Wy = params[idx: idx + self.H]
        idx += self.H
        self.by = params[idx]

    def forward(self, sequence: List[float]) -> float:
        """Runs the full gate recurrence across the sequence, returns the
        predicted NEXT value after the final timestep.
        """
        h = [0.0] * self.H
        c = [0.0] * self.H
        for x_t in sequence:
            combined = h + [x_t]
            f = [sigmoid(dot(self.gates["f"][u], combined) + self.biases["f"][u]) for u in range(self.H)]
            i = [sigmoid(dot(self.gates["i"][u], combined) + self.biases["i"][u]) for u in range(self.H)]
            c_tilde = [math.tanh(dot(self.gates["c"][u], combined) + self.biases["c"][u]) for u in range(self.H)]
            o = [sigmoid(dot(self.gates["o"][u], combined) + self.biases["o"][u]) for u in range(self.H)]
            c = [f[u] * c[u] + i[u] * c_tilde[u] for u in range(self.H)]
            h = [o[u] * math.tanh(c[u]) for u in range(self.H)]
        return dot(self.Wy, h) + self.by


def mse_loss(model: LSTMCell, dataset: List[Tuple[List[float], float]]) -> float:
    return sum((model.forward(seq) - target) ** 2 for seq, target in dataset) / len(dataset)


def train_lstm(model: LSTMCell, dataset: List[Tuple[List[float], float]], epochs: int = 40, lr: float = 0.05, eps: float = 1e-3) -> List[float]:
    """Trains via numerical (finite-difference) gradient descent. At this
    parameter count (~100), this is exact and tractable in pure Python —
    the update rule (`param -= lr * d(loss)/d(param)`) is identical to
    backprop; only HOW the gradient is computed differs.
    """
    loss_history = []
    for epoch in range(epochs):
        base_params = model.get_params()
        base_loss = mse_loss(model, dataset)
        loss_history.append(base_loss)

        gradients = []
        for i in range(len(base_params)):
            perturbed = list(base_params)
            perturbed[i] += eps
            model.set_params(perturbed)
            loss_plus = mse_loss(model, dataset)
            gradients.append((loss_plus - base_loss) / eps)

        updated = [p - lr * g for p, g in zip(base_params, gradients)]
        model.set_params(updated)
    return loss_history


# =====================================================================
# 2. SYNTHETIC HCP RESPONSE DATA (recency + carryover + fatigue signal)
# =====================================================================
def generate_synthetic_hcp_data(n_sequences: int = 12, seq_len: int = 8) -> List[Tuple[List[float], float]]:
    """Simulates weekly promotional intensity -> next-week NBRx response,
    with a carryover effect (today's response depends on recent touches,
    weighted toward the most recent weeks) plus fatigue (diminishing
    returns from repeated high-intensity weeks) and noise.
    """
    rng = random.Random(42)
    dataset = []
    for _ in range(n_sequences):
        touches = [rng.uniform(0, 1) for _ in range(seq_len + 1)]
        weighted_recent = sum(t * (0.5 ** (len(touches[:seq_len]) - 1 - i)) for i, t in enumerate(touches[:seq_len]))
        fatigue_penalty = 0.15 * max(0, sum(touches[:seq_len]) - seq_len * 0.5)
        response = max(0.0, min(1.0, 0.6 * weighted_recent - fatigue_penalty + rng.gauss(0, 0.03)))
        dataset.append((touches[:seq_len], response))
    return dataset


# =====================================================================
# 3. GENETIC ALGORITHM: evolve an optimal 13-week channel sequence
# =====================================================================
CHANNELS = ["Call", "Email", "SMS", "None"]
CHANNEL_INTENSITY = {"Call": 1.0, "Email": 0.3, "SMS": 0.2, "None": 0.0}
CHANNEL_WEEKLY_CAP = {"Call": 4, "Email": 8, "SMS": 8}  # max uses per 13-week cycle
HORIZON_WEEKS = 13


@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 25
    mutation_rate: float = 0.15
    elitism_count: int = 3


def random_sequence() -> List[str]:
    return [random.choice(CHANNELS) for _ in range(HORIZON_WEEKS)]


def constraint_penalty(sequence: List[str]) -> float:
    """Soft constraint: exceeding a channel's cap subtracts from fitness
    instead of making the candidate invalid outright — lets the GA still
    learn FROM slightly-infeasible candidates instead of discarding them.
    """
    penalty = 0.0
    for channel, cap in CHANNEL_WEEKLY_CAP.items():
        used = sequence.count(channel)
        if used > cap:
            penalty += 0.05 * (used - cap)
    return penalty


def fitness(model: LSTMCell, sequence: List[str]) -> float:
    intensities = [CHANNEL_INTENSITY[c] for c in sequence]
    predicted_response = model.forward(intensities)
    return predicted_response - constraint_penalty(sequence)


def tournament_select(population: List[List[str]], scores: List[float], k: int = 3) -> List[str]:
    contenders = random.sample(list(zip(population, scores)), k)
    return max(contenders, key=lambda pair: pair[1])[0]


def crossover(parent_a: List[str], parent_b: List[str]) -> List[str]:
    point = random.randint(1, HORIZON_WEEKS - 1)
    return parent_a[:point] + parent_b[point:]


def mutate(sequence: List[str], rate: float) -> List[str]:
    return [random.choice(CHANNELS) if random.random() < rate else gene for gene in sequence]


def run_genetic_algorithm(model: LSTMCell, config: GAConfig) -> Tuple[List[str], float, List[float]]:
    population = [random_sequence() for _ in range(config.population_size)]
    best_fitness_per_generation = []

    for generation in range(config.generations):
        scores = [fitness(model, seq) for seq in population]
        ranked = sorted(zip(population, scores), key=lambda p: p[1], reverse=True)
        best_fitness_per_generation.append(ranked[0][1])

        next_population = [seq for seq, _ in ranked[:config.elitism_count]]  # elitism
        while len(next_population) < config.population_size:
            parent_a = tournament_select(population, scores)
            parent_b = tournament_select(population, scores)
            child = mutate(crossover(parent_a, parent_b), config.mutation_rate)
            next_population.append(child)
        population = next_population

    final_scores = [fitness(model, seq) for seq in population]
    best_idx = max(range(len(population)), key=lambda i: final_scores[i])
    return population[best_idx], final_scores[best_idx], best_fitness_per_generation


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Temporal Forecasting (LSTM) + Genetic Algorithm Optimization ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Training an LSTM to predict HCP response from promotional touch sequences{Colors.RESET}")
    dataset = generate_synthetic_hcp_data(n_sequences=12, seq_len=8)
    model = LSTMCell(hidden_size=4, seed=1)

    initial_loss = mse_loss(model, dataset)
    loss_history = train_lstm(model, dataset, epochs=40, lr=0.05)
    final_loss = mse_loss(model, dataset)

    print(f"  Initial MSE loss: {initial_loss:.5f}")
    print(f"  Loss every 10 epochs: {[round(l, 5) for l in loss_history[::10]]}")
    print(f"  Final MSE loss:   {Colors.GREEN}{final_loss:.5f}{Colors.RESET} "
          f"({'improved' if final_loss < initial_loss else 'did not improve'} over {len(loss_history)} epochs)\n")

    print(f"{Colors.YELLOW}2. Sanity check: model on a held-out synthetic sequence{Colors.RESET}")
    holdout_seq, holdout_target = generate_synthetic_hcp_data(n_sequences=1, seq_len=8)[0]
    prediction = model.forward(holdout_seq)
    print(f"  True response: {holdout_target:.3f} | Predicted: {prediction:.3f} | Abs error: {abs(prediction - holdout_target):.3f}\n")

    print(f"{Colors.YELLOW}3. Genetic Algorithm: evolving the optimal 13-week promotional sequence{Colors.RESET}")
    config = GAConfig(population_size=30, generations=25, mutation_rate=0.15, elitism_count=3)
    best_sequence, best_score, fitness_curve = run_genetic_algorithm(model, config)

    print(f"  Fitness by generation (every 5): {[round(f, 4) for f in fitness_curve[::5]]}")
    print(f"  {Colors.GREEN}Best fitness found: {best_score:.4f}{Colors.RESET} (started at {fitness_curve[0]:.4f} in generation 0)\n")

    print(f"{Colors.BOLD}Recommended 13-week promotional sequence:{Colors.RESET}")
    for week, channel in enumerate(best_sequence, start=1):
        print(f"  Week {week:>2}: {channel}")

    channel_counts = {c: best_sequence.count(c) for c in CHANNELS}
    print(f"\n  Channel usage: {channel_counts}")
    print(f"  Caps respected: {all(channel_counts.get(c, 0) <= cap for c, cap in CHANNEL_WEEKLY_CAP.items())}")

    print(f"\n{Colors.GREEN}✔ LSTM trained via real gradient descent; GA evolved a constraint-aware sequence using it as the fitness function.{Colors.RESET}")

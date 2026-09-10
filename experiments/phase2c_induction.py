"""Phase 2c: induction head with count+argmax readout (LM への一歩).

Task: each sequence is a fresh random pattern of length T/2 (distinct
tokens from a 64-token vocabulary, so the routing target is unambiguous
and the counting oracle sits at 100%) followed by its exact copy.
Predicting the second half requires induction: find the earlier occurrence
of the current token and emit what followed it — nothing is memorizable
across samples, and a static bigram table is useless because the mapping
changes every sample.

Model: CountingGateAttention — the prev-token wiring is fixed; only the
matching circuit (a logic-gate encoder) is learned. The readout is a
per-vocab vote count, replacing softmax with count+argmax (a multi-way
majority vote — the gate-native normalization). Hard mode runs the frozen
boolean circuit with integer votes.

Ceiling: bigram collisions inside a random pattern make some predictions
ambiguous, so we report the exact-token-match counting ORACLE alongside.
A model that matches the oracle routes perfectly.

Controls: a 2-layer transformer (learned positions, so it cannot run on
longer sequences), and a static per-token head (should sit near chance).

Trained at T=32, evaluated at T=32 and T=64.

Usage: python experiments/phase2c_induction.py [--steps 4000] [--seeds 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic.attention import CountingGateAttention
from experiments.phase1_recall import to_bits

RESULTS = Path(__file__).resolve().parent.parent / "results"

VOCAB, TOKEN_BITS = 64, 6
T_TRAIN = 32
EVAL_T = [32, 64]
CODE_BITS, HIDDEN = 12, 96


def make_batch(batch: int, T: int, g: torch.Generator):
    half = T // 2
    pat = torch.argsort(torch.rand(batch, VOCAB, generator=g), dim=-1)[:, :half]
    x = torch.cat([pat, pat], dim=1)                    # (B, T)
    return x, to_bits(x, TOKEN_BITS)


def pred_slice(T: int):
    """Predict x_{t+1} for t in [T/2, T-2] — the copyable region."""
    return slice(T // 2, T - 1), slice(T // 2 + 1, T)


@torch.no_grad()
def oracle_acc(x: torch.Tensor) -> float:
    """Count votes by exact token identity; argmax with the same tie rule."""
    B, T = x.shape
    match = (x.unsqueeze(2) == x.unsqueeze(1))                    # (B,Tq,Tk)
    mask = torch.ones(T, T, dtype=torch.bool).tril(-1)
    v = F.one_hot(x, VOCAB).float()
    val = torch.zeros_like(v); val[:, :-1] = v[:, 1:]
    counts = (match & mask).float() @ val                         # (B,T,V)
    s_in, s_out = pred_slice(T)
    return (counts[:, s_in].argmax(-1) == x[:, s_out]).float().mean().item()


class TinyTransformer(nn.Module):
    def __init__(self, T: int, d: int = 64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=128,
                                           batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d, VOCAB)
        self.T = T

    def forward(self, x, _bits=None):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T)).unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(T)
        return self.head(self.blocks(h, mask=causal))


class StaticHead(nn.Module):
    """Next token from current token only, no attention (chance-level control)."""

    def __init__(self):
        super().__init__()
        self.table = nn.Linear(VOCAB, VOCAB)

    def forward(self, x, _bits=None):
        return self.table(F.one_hot(x, VOCAB).float())


def train(model, is_gate, steps, g, batch=64):
    opt = torch.optim.Adam(model.parameters(), lr=0.03 if is_gate else 1e-3)
    s_in, s_out = pred_slice(T_TRAIN)
    for _ in range(steps):
        x, bits = make_batch(batch, T_TRAIN, g)
        logits = model(bits, x) if is_gate else model(x)
        loss = F.cross_entropy(logits[:, s_in].reshape(-1, VOCAB),
                               x[:, s_out].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def evaluate(model, is_gate, T, g, n_samples=500):
    x, bits = make_batch(n_samples, T, g)
    s_in, s_out = pred_slice(T)
    logits = model(bits, x) if is_gate else model(x)
    soft = (logits[:, s_in].argmax(-1) == x[:, s_out]).float().mean().item()
    hard = None
    if is_gate:
        counts = model.forward_hard(bits.bool(), x)
        hard = (counts[:, s_in].argmax(-1) == x[:, s_out]).float().mean().item()
    return soft, hard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    g0 = torch.Generator().manual_seed(7)
    oracle = {f"T={T}": round(oracle_acc(make_batch(2000, T, g0)[0]), 4) for T in EVAL_T}
    print("oracle (perfect routing ceiling):", oracle)

    results = {"oracle": oracle}
    for name in ["gate-counting", "transformer-2L", "static"]:
        results[name] = []
        for seed in range(args.seeds):
            torch.manual_seed(1000 + seed)
            g = torch.Generator().manual_seed(1000 + seed)
            if name == "gate-counting":
                model, is_gate = CountingGateAttention(
                    TOKEN_BITS, VOCAB, CODE_BITS, HIDDEN,
                    generator=g, residual_init=True), True
            elif name == "transformer-2L":
                model, is_gate = TinyTransformer(T_TRAIN), False
            else:
                model, is_gate = StaticHead(), False
            t0 = time.time()
            train(model, is_gate, args.steps, g)
            entry = {"seed": seed, "train_s": round(time.time() - t0, 1), "acc": {}}
            for T in EVAL_T:
                if name == "transformer-2L" and T != T_TRAIN:
                    continue  # learned positions stop at T_TRAIN
                soft, hard = evaluate(model, is_gate, T, g)
                entry["acc"][f"T={T}"] = {"soft": round(soft, 4),
                                          **({"hard": round(hard, 4)} if hard is not None else {})}
            print(name, entry)
            results[name].append(entry)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase2c.json").write_text(json.dumps(results, indent=2))
    print("wrote results/phase2c.json")

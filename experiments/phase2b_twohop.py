"""Phase 2b: can hard routing be COMPOSED? Two-hop associative recall.

Task: context is N (key, value) pairs where every value is itself one of
the context keys (6-bit each). Query q is a context key; the target is
V[V[q]] — one lookup through the table, then a second lookup with the
first result. A single attention (gate or softmax) cannot express this;
two stacked ones can. This is the minimal test that gate attention layers
stack: the soft output of hop 1 must serve as the query of hop 2 during
training, and survive full discretization at inference.

Models:
  gate-2hop    : two GateAttentionA modules in series (separate encoders)
  gate-1hop    : single GateAttentionA trained on the same target (control:
                 should fail, proving depth is required)
  softmax-2hop : two stacked dot-product attentions (control)
  mlp          : fixed-length MLP (control)

Trained at N=8, evaluated at N=8/16/32, exact match over 6 bits.

Usage: python experiments/phase2b_twohop.py [--steps 12000] [--seeds 3]
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
from gatelogic import GateAttentionA, SoftmaxAttentionBaseline, MLPBaseline
import experiments.phase1_recall as p1
from experiments.phase1_recall import to_bits, KEY_BITS, N_TRAIN, EVAL_LENGTHS

RESULTS = Path(__file__).resolve().parent.parent / "results"
CODE_BITS, HIDDEN = 12, 96


def make_batch(batch: int, n_pairs: int, g: torch.Generator):
    keys_int = torch.argsort(torch.rand(batch, 2 ** KEY_BITS, generator=g), dim=-1)[:, :n_pairs]
    link = torch.randint(0, n_pairs, (batch, n_pairs), generator=g)  # v_i = k_link[i]
    vals_int = keys_int.gather(1, link)
    q_pos = torch.randint(0, n_pairs, (batch,), generator=g)
    hop1_int = vals_int.gather(1, q_pos.unsqueeze(1)).squeeze(1)     # V[q]
    hop1_pos = link.gather(1, q_pos.unsqueeze(1))                    # position of V[q] as a key
    target_int = vals_int.gather(1, hop1_pos).squeeze(1)             # V[V[q]]
    query_int = keys_int.gather(1, q_pos.unsqueeze(1)).squeeze(1)
    return (to_bits(keys_int, KEY_BITS), to_bits(vals_int, KEY_BITS),
            to_bits(query_int, KEY_BITS), to_bits(target_int, KEY_BITS),
            to_bits(hop1_int, KEY_BITS))


class GateTwoHop(nn.Module):
    def __init__(self, g, share_qk: bool = False):
        super().__init__()
        kw = dict(generator=g, residual_init=True, share_qk=share_qk)
        self.hop1 = GateAttentionA(KEY_BITS, KEY_BITS, CODE_BITS, HIDDEN, **kw)
        self.hop2 = GateAttentionA(KEY_BITS, KEY_BITS, CODE_BITS, HIDDEN, **kw)

    def forward(self, k, v, q):
        return self.hop2(k, v, self.hop1(k, v, q))

    @torch.no_grad()
    def forward_hard(self, k, v, q):
        return self.hop2.forward_hard(k, v, self.hop1.forward_hard(k, v, q))


class SoftmaxTwoHop(nn.Module):
    def __init__(self):
        super().__init__()
        self.hop1 = SoftmaxAttentionBaseline(KEY_BITS, KEY_BITS, d_model=32)
        self.hop2 = SoftmaxAttentionBaseline(KEY_BITS, KEY_BITS, d_model=32)

    def forward(self, k, v, q):
        return self.hop2(k, v, torch.sigmoid(self.hop1(k, v, q)))


def build(name, g):
    if name == "gate-2hop":
        return GateTwoHop(g), True
    if name == "gate-2hop-shared":
        return GateTwoHop(g, share_qk=True), True
    if name == "gate-2hop-shared-aux":  # aux supervision handled in train()
        return GateTwoHop(g, share_qk=True), True
    if name == "gate-2hop-staged":      # stage-wise training handled in train()
        return GateTwoHop(g, share_qk=True), True
    if name == "gate-2hop-anneal":      # aux weight annealed 1 -> 0
        return GateTwoHop(g, share_qk=True), True
    if name == "gate-2hop-anneal5":     # aux weight annealed 5 -> 0 (soft staging)
        return GateTwoHop(g, share_qk=True), True
    if name == "gate-2hop-staged-joint":  # staged, then joint e2e fine-tune
        return GateTwoHop(g, share_qk=True), True
    if name in ("gate-2hop-warmup",       # hop2 lr warmup, e2e loss only
                "gate-2hop-aux-warmup",   # + aux on hop1 (soft staging, one run)
                "gate-2hop-aux-noise"):   # aux + noise on hop2's input
        return GateTwoHop(g, share_qk=True), True
    if name == "gate-1hop":
        return GateAttentionA(KEY_BITS, KEY_BITS, CODE_BITS, HIDDEN,
                              generator=g, residual_init=True), True
    if name == "softmax-2hop":
        return SoftmaxTwoHop(), False
    if name == "mlp":
        return MLPBaseline(N_TRAIN, KEY_BITS, KEY_BITS, hidden=256), False
    raise ValueError(name)


def train_staged(model, steps, g, batch=256):
    """Stage 1: hop1 alone on the intermediate target V[q] (= Phase 1 recall).
    Stage 2: hop1 frozen, hop2 alone on the final target, fed hop1's output."""
    eps = 1e-6
    opt1 = torch.optim.Adam(model.hop1.parameters(), lr=0.03)
    for _ in range(steps // 2):
        k, v, q, _, y1 = make_batch(batch, N_TRAIN, g)
        a1 = model.hop1(k, v, q)
        loss = F.binary_cross_entropy(a1.clamp(eps, 1 - eps), y1)
        opt1.zero_grad(); loss.backward(); opt1.step()
    opt2 = torch.optim.Adam(model.hop2.parameters(), lr=0.03)
    for _ in range(steps - steps // 2):
        k, v, q, y, _ = make_batch(batch, N_TRAIN, g)
        with torch.no_grad():
            a1 = model.hop1(k, v, q)
        out = model.hop2(k, v, a1)
        loss = F.binary_cross_entropy(out.clamp(eps, 1 - eps), y)
        opt2.zero_grad(); loss.backward(); opt2.step()


def train_anneal(model, steps, g, batch=256, aux_end=0.6, aux_w0=1.0):
    """Auxiliary loss on V[q] with weight annealed linearly aux_w0 -> 0 by
    `aux_end` of training; the remainder is pure end-to-end. A large aux_w0
    approximates staging without ever freezing anything."""
    eps = 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for step in range(steps):
        w = max(0.0, aux_w0 * (1.0 - step / (aux_end * steps)))
        k, v, q, y, y1 = make_batch(batch, N_TRAIN, g)
        a1 = model.hop1(k, v, q)
        out = model.hop2(k, v, a1)
        loss = F.binary_cross_entropy(out.clamp(eps, 1 - eps), y)
        if w > 0:
            loss = loss + w * F.binary_cross_entropy(a1.clamp(eps, 1 - eps), y1)
        opt.zero_grad(); loss.backward(); opt.step()


def train_staged_joint(model, steps, g, batch=256):
    """Staged training for the first 70%, then a joint end-to-end
    fine-tune (final loss only, all parameters, lower lr)."""
    eps = 1e-6
    train_staged(model, int(steps * 0.7), g, batch)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps - int(steps * 0.7)):
        k, v, q, y, _ = make_batch(batch, N_TRAIN, g)
        out = model(k, v, q)
        loss = F.binary_cross_entropy(out.clamp(eps, 1 - eps), y)
        opt.zero_grad(); loss.backward(); opt.step()


def train_coadapt(model, steps, g, batch=256, aux=False, warmup=False, noise=0.0):
    """Freeze-free co-adaptation prevention: hop2's lr is warmed up from 0
    between 40% and 70% of training (warmup=True), and/or hop1's output is
    corrupted with annealed random bits before hop2 (noise>0). Single run,
    no explicit stages."""
    eps = 1e-6
    opt1 = torch.optim.Adam(model.hop1.parameters(), lr=0.03)
    opt2 = torch.optim.Adam(model.hop2.parameters(), lr=0.03)
    for step in range(steps):
        frac = step / steps
        if warmup:
            lr2 = 0.03 * min(1.0, max(0.0, (frac - 0.4) / 0.3))
            for pg in opt2.param_groups:
                pg["lr"] = lr2
        k, v, q, y, y1 = make_batch(batch, N_TRAIN, g)
        a1 = model.hop1(k, v, q)
        a1_in = a1
        if noise > 0:
            p = noise * (1 - frac)
            flip = (torch.rand(a1.shape, generator=g) < p).float()
            rnd = (torch.rand(a1.shape, generator=g) < 0.5).float()
            a1_in = a1 * (1 - flip) + rnd * flip
        out = model.hop2(k, v, a1_in)
        loss = F.binary_cross_entropy(out.clamp(eps, 1 - eps), y)
        if aux:
            loss = loss + F.binary_cross_entropy(a1.clamp(eps, 1 - eps), y1)
        opt1.zero_grad(); opt2.zero_grad(); loss.backward()
        opt1.step(); opt2.step()


def train(model, is_gate, steps, g, batch=256, mode="none"):
    if mode == "warmup":
        return train_coadapt(model, steps, g, batch, aux=False, warmup=True)
    if mode == "aux-warmup":
        return train_coadapt(model, steps, g, batch, aux=True, warmup=True)
    if mode == "aux-noise":
        return train_coadapt(model, steps, g, batch, aux=True, noise=0.3)
    if mode == "staged":
        return train_staged(model, steps, g, batch)
    if mode == "anneal":
        return train_anneal(model, steps, g, batch)
    if mode == "anneal5":
        return train_anneal(model, steps, g, batch, aux_w0=5.0)
    if mode == "staged-joint":
        return train_staged_joint(model, steps, g, batch)
    opt = torch.optim.Adam(model.parameters(), lr=0.03 if is_gate else 1e-3)
    eps = 1e-6
    for _ in range(steps):
        k, v, q, y, y1 = make_batch(batch, N_TRAIN, g)
        if mode == "aux":  # deep supervision: the intermediate target V[q] is known
            a1 = model.hop1(k, v, q)
            out = model.hop2(k, v, a1)
            loss = (F.binary_cross_entropy(out.clamp(eps, 1 - eps), y)
                    + F.binary_cross_entropy(a1.clamp(eps, 1 - eps), y1))
        else:
            out = model(k, v, q)
            loss = (F.binary_cross_entropy(out.clamp(eps, 1 - eps), y) if is_gate
                    else F.binary_cross_entropy_with_logits(out, y))
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def evaluate(model, is_gate, n_pairs, g, n_samples=2000):
    k, v, q, y, _ = make_batch(n_samples, n_pairs, g)
    out = model(k, v, q)
    pred = (out if is_gate else torch.sigmoid(out)) > 0.5
    soft = (pred == y.bool()).all(-1).float().mean().item()
    hard = None
    if is_gate:
        hp = model.forward_hard(k.bool(), v.bool(), q.bool())
        hard = (hp == y.bool()).all(-1).float().mean().item()
    return soft, hard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", type=str, default="gate-2hop,gate-1hop,softmax-2hop,mlp")
    ap.add_argument("--out", type=str, default="phase2b.json")
    args = ap.parse_args()

    results = {}
    for name in args.models.split(","):
        results[name] = []
        for seed in range(args.seeds):
            torch.manual_seed(1000 + seed)
            g = torch.Generator().manual_seed(1000 + seed)
            model, is_gate = build(name, g)
            t0 = time.time()
            mode = ("aux-warmup" if name.endswith("-aux-warmup") else
                    "aux-noise" if name.endswith("-aux-noise") else
                    "warmup" if name.endswith("-warmup") else
                    "aux" if name.endswith("-aux") else
                    "staged-joint" if name.endswith("-staged-joint") else
                    "staged" if name.endswith("-staged") else
                    "anneal5" if name.endswith("-anneal5") else
                    "anneal" if name.endswith("-anneal") else "none")
            train(model, is_gate, args.steps, g, mode=mode)
            entry = {"seed": seed, "train_s": round(time.time() - t0, 1), "acc": {}}
            for n in EVAL_LENGTHS:
                if name == "mlp" and n != N_TRAIN:
                    continue
                soft, hard = evaluate(model, is_gate, n, g)
                entry["acc"][f"N={n}"] = {"soft": round(soft, 4),
                                          **({"hard": round(hard, 4)} if hard is not None else {})}
            print(name, entry)
            results[name].append(entry)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote results/{args.out}")

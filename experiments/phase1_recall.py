"""Phase 1: can gate networks learn input-dependent routing (attention)?

Task — associative recall: the context is N (key, value) pairs with distinct
random keys (6-bit keys, 4-bit values), plus a query key that appeared in the
context. The model must output the associated 4-bit value. Fresh random data
every step (nothing to memorize; only the routing rule generalizes).

Models:
  gateA-shared  : GateAttentionA, one encoder shared for keys and query
  gateA-sep     : GateAttentionA, separate key/query encoders (must align)
  gateB-sep     : GateAttentionB (popcount >= theta), separate encoders
  softmax-attn  : same-scale single-head dot-product attention (control)
  mlp           : fixed-length MLP, no attention (task-weakness control)

Trained at N=8, evaluated at N=8/16/32 (exact-match over the 4 value bits).
Gate models are additionally evaluated *hardened*: every neuron frozen to its
argmax gate, inference in pure boolean ops.

Usage: python experiments/phase1_recall.py [--steps 4000] [--seeds 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic import (GateAttentionA, GateAttentionB,
                       SoftmaxAttentionBaseline, MLPBaseline)

RESULTS = Path(__file__).resolve().parent.parent / "results"

KEY_BITS, VAL_BITS = 6, 4
N_TRAIN = 8
EVAL_LENGTHS = [8, 16, 32]

# overwritten from CLI flags in __main__
CODE_BITS, HIDDEN, RESIDUAL_INIT = 6, 64, False


def to_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(bits - 1, -1, -1, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).float()


def make_batch(batch: int, n_pairs: int, g: torch.Generator):
    """Distinct random keys, random values, query = one of the keys."""
    keys_int = torch.argsort(torch.rand(batch, 2 ** KEY_BITS, generator=g), dim=-1)[:, :n_pairs]
    vals_int = torch.randint(0, 2 ** VAL_BITS, (batch, n_pairs), generator=g)
    q_pos = torch.randint(0, n_pairs, (batch,), generator=g)
    query_int = keys_int.gather(1, q_pos.unsqueeze(1)).squeeze(1)
    target_int = vals_int.gather(1, q_pos.unsqueeze(1)).squeeze(1)
    return (to_bits(keys_int, KEY_BITS), to_bits(vals_int, VAL_BITS),
            to_bits(query_int, KEY_BITS), to_bits(target_int, VAL_BITS))


def build(name: str, g: torch.Generator):
    gate_kw = dict(generator=g, residual_init=RESIDUAL_INIT)
    if name == "gateA-shared":
        return GateAttentionA(KEY_BITS, VAL_BITS, CODE_BITS, HIDDEN, share_qk=True, **gate_kw), True
    if name == "gateA-sep":
        return GateAttentionA(KEY_BITS, VAL_BITS, CODE_BITS, HIDDEN, share_qk=False, **gate_kw), True
    if name == "gateB-sep":
        return GateAttentionB(KEY_BITS, VAL_BITS, CODE_BITS, HIDDEN, share_qk=False, **gate_kw), True
    if name == "gateB-shared":
        return GateAttentionB(KEY_BITS, VAL_BITS, CODE_BITS, HIDDEN, share_qk=True, **gate_kw), True
    if name == "softmax-attn":
        return SoftmaxAttentionBaseline(KEY_BITS, VAL_BITS, d_model=32), False
    if name == "mlp":
        return MLPBaseline(N_TRAIN, KEY_BITS, VAL_BITS, hidden=256), False
    raise ValueError(name)


def train(model, is_gate: bool, steps: int, g: torch.Generator, batch: int = 256,
          mix_lengths: bool = False):
    lr = 0.03 if is_gate else 1e-3
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        n = int(torch.randint(1, N_TRAIN + 1, (1,), generator=g)) if mix_lengths else N_TRAIN
        k, v, q, y = make_batch(batch, n, g)
        out = model(k, v, q)
        if is_gate:
            loss = F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), y)
        else:
            loss = F.binary_cross_entropy_with_logits(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, is_gate: bool, n_pairs: int, g: torch.Generator, n_samples: int = 2000):
    k, v, q, y = make_batch(n_samples, n_pairs, g)
    out = model(k, v, q)
    pred = (torch.sigmoid(out) if not is_gate else out) > 0.5
    soft = (pred == y.bool()).all(-1).float().mean().item()
    hard = None
    if is_gate:
        hp = model.forward_hard(k.bool(), v.bool(), q.bool())
        hard = (hp == y.bool()).all(-1).float().mean().item()
    return soft, hard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--code-bits", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--residual-init", action="store_true")
    ap.add_argument("--mix-lengths", action="store_true",
                    help="sample training length uniformly from 1..N_TRAIN")
    ap.add_argument("--models", type=str,
                    default="gateA-shared,gateA-sep,gateB-sep,softmax-attn,mlp")
    ap.add_argument("--out", type=str, default="phase1.json")
    args = ap.parse_args()

    CODE_BITS, HIDDEN, RESIDUAL_INIT = args.code_bits, args.hidden, args.residual_init
    models = args.models.split(",")
    results = {}
    for name in models:
        results[name] = []
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(1000 + seed)
            torch.manual_seed(1000 + seed)  # for nn.Linear inits
            model, is_gate = build(name, g)
            t0 = time.time()
            train(model, is_gate, args.steps, g, mix_lengths=args.mix_lengths)
            entry = {"seed": seed, "train_s": round(time.time() - t0, 1), "acc": {}}
            for n in EVAL_LENGTHS:
                if name == "mlp" and n != N_TRAIN:
                    continue  # fixed-length model
                soft, hard = evaluate(model, is_gate, n, g)
                entry["acc"][f"N={n}"] = {"soft": round(soft, 4),
                                          **({"hard": round(hard, 4)} if hard is not None else {})}
            print(name, entry)
            results[name].append(entry)

    RESULTS.mkdir(exist_ok=True)
    payload = {"config": {k: v for k, v in vars(args).items()}, "results": results}
    (RESULTS / args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote results/{args.out}")

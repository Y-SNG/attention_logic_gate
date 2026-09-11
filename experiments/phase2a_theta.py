"""Phase 2a: close gateB's hardening gap.

Phase 1 finding: GateAttentionB (popcount >= theta) trains to ~99% soft
accuracy, but hardened accuracy is seed-dependent because the learned
continuous theta can land in an ambiguous region between integer popcount
levels. Two remedies, applied together:

  1. beta annealing: sharpen the sigmoid threshold from 2 -> 10 over
     training, so the soft model approaches its own hard behavior;
  2. integer theta calibration: after training, sweep theta over
     {0.5, 1.5, ..., c-0.5} and keep the value with the best *hardened*
     validation accuracy at the training length.

Usage: python experiments/phase2a_theta.py [--steps 8000] [--seeds 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic import GateAttentionB
import experiments.phase1_recall as p1

RESULTS = Path(__file__).resolve().parent.parent / "results"


def train_annealed(model, steps, g, beta_start=2.0, beta_end=10.0, batch=256):
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for step in range(steps):
        model.beta = beta_start + (beta_end - beta_start) * step / max(1, steps - 1)
        k, v, q, y = p1.make_batch(batch, p1.N_TRAIN, g)
        out = model(k, v, q)
        loss = F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), y)
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def calibrate_theta(model, g, n_samples=2000):
    """Pick the integer threshold (as t - 0.5) with best hardened val acc."""
    k, v, q, y = p1.make_batch(n_samples, p1.N_TRAIN, g)
    best_t, best_acc = None, -1.0
    for t in range(1, model.code_bits + 1):
        model.theta.fill_(t - 0.5)
        pred = model.forward_hard(k.bool(), v.bool(), q.bool())
        acc = (pred == y.bool()).all(-1).float().mean().item()
        if acc > best_acc:
            best_t, best_acc = t, acc
    model.theta.fill_(best_t - 0.5)
    return best_t, best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    p1.CODE_BITS, p1.HIDDEN, p1.RESIDUAL_INIT = 12, 96, True
    results = {}
    for name, share in [("gateB-sep", False), ("gateB-shared", True)]:
        results[name] = []
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(1000 + seed)
            model = GateAttentionB(p1.KEY_BITS, p1.VAL_BITS, p1.CODE_BITS, p1.HIDDEN,
                                   share_qk=share, generator=g, residual_init=True)
            t0 = time.time()
            train_annealed(model, args.steps, g)
            theta_raw = float(model.theta.detach())
            theta_int, cal_acc = calibrate_theta(model, g)
            entry = {"seed": seed, "train_s": round(time.time() - t0, 1),
                     "theta_learned": round(theta_raw, 3), "theta_calibrated": theta_int,
                     "acc": {}}
            for n in p1.EVAL_LENGTHS:
                soft, hard = p1.evaluate(model, True, n, g)
                entry["acc"][f"N={n}"] = {"soft": round(soft, 4), "hard": round(hard, 4)}
            print(name, entry)
            results[name].append(entry)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase2a.json").write_text(json.dumps(results, indent=2))
    print("wrote results/phase2a.json")

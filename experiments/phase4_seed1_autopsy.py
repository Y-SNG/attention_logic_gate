"""Phase 4-3: autopsy of the seed-1 92% local optimum in two-hop recall.

Every training method (staged, warmup, aux) lands seed 1 at exactly
91.7/82.3/59.9% while seeds 0 and 2 reach 100%. This script isolates the
two hops of the staged seed-1 model:

  - hop1 alone: recall accuracy on the intermediate target V[q], plus code
    injectivity of its key encoder over all 64 keys;
  - hop2 alone: accuracy when fed the TRUE intermediate V[q] as its query
    (removes hop1's errors entirely), plus its own code injectivity;
  - the composed pipeline, for reference.

Whichever isolated hop falls short of 100% carries the defect, and the
injectivity counts say whether it is a code-collision problem (the wiring
lottery of that seed's random connectivity).

Usage: python experiments/phase4_seed1_autopsy.py [--seeds 0 1 2]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.phase2b_twohop as p2b
from experiments.phase2b_twohop import GateTwoHop, make_batch, train_staged, evaluate
from experiments.phase1_recall import to_bits, KEY_BITS, EVAL_LENGTHS

RESULTS = Path(__file__).resolve().parent.parent / "results"


@torch.no_grad()
def hop_recall_acc(att, n_pairs, g, use_true_query=False, n_samples=2000):
    """Exact-match recall of one GateAttentionA hop, hardened."""
    k, v, q, y, y1 = make_batch(n_samples, n_pairs, g)
    query = y1 if use_true_query else q       # hop2 is queried with V[q]
    target = y if use_true_query else y1      # and must return V[V[q]]
    pred = att.forward_hard(k.bool(), v.bool(), query.bool())
    return (pred == target.bool()).all(-1).float().mean().item()


@torch.no_grad()
def injectivity(att):
    keys = torch.arange(64)
    bits = to_bits(keys, KEY_BITS).bool()
    ck = att.k_enc.forward_hard(bits)
    cq = att.q_enc.forward_hard(bits)
    k_codes = {tuple(r.tolist()) for r in ck}
    qk_match = int((ck == cq).all(-1).sum())  # queries that hit their own key
    return len(k_codes), qk_match


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=16000)
    args = ap.parse_args()

    report = {}
    for seed in args.seeds:
        torch.manual_seed(1000 + seed)
        g = torch.Generator().manual_seed(1000 + seed)
        model = GateTwoHop(g, share_qk=True)
        train_staged(model, args.steps, g)

        entry = {"pipeline": {}, "hop1_alone": {}, "hop2_true_query": {}}
        for n in EVAL_LENGTHS:
            soft, hard = evaluate(model, True, n, g)
            entry["pipeline"][f"N={n}"] = round(hard, 4)
            entry["hop1_alone"][f"N={n}"] = round(hop_recall_acc(model.hop1, n, g), 4)
            entry["hop2_true_query"][f"N={n}"] = round(
                hop_recall_acc(model.hop2, n, g, use_true_query=True), 4)
        for hop_name, att in [("hop1", model.hop1), ("hop2", model.hop2)]:
            uniq, self_match = injectivity(att)
            entry[f"{hop_name}_unique_codes"] = uniq       # /64
            entry[f"{hop_name}_query_self_match"] = self_match  # /64
        print(f"seed {seed}:", json.dumps(entry))
        report[f"seed{seed}"] = entry

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase4_autopsy.json").write_text(json.dumps(report, indent=2))
    print("wrote results/phase4_autopsy.json")

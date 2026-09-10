"""Train the winning gateA-shared config, harden it, and export the circuit.

Output: results/circuit_gateA_shared.json with
  - per-layer neurons: chosen gate, input wire indices, liveness
    (backward reachability from the code bits, aware of which inputs a
    gate actually depends on — e.g. pass-through "A" only uses wire a)
  - the full 64-key -> 12-bit code table of the hardened encoder
  - eval accuracies as provenance

Usage: python experiments/export_circuit.py [--seed 0]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic import GateAttentionA
from gatelogic.layers import GATE_NAMES
import experiments.phase1_recall as p1

RESULTS = Path(__file__).resolve().parent.parent / "results"

# which of the two input wires each gate type actually reads
_DEPS = {0: (), 15: (), 3: ("a",), 12: ("a",), 5: ("b",), 10: ("b",)}


def export_encoder(enc, key_bits: int):
    layers = []
    for layer in enc.stack:
        gates = layer.weights.argmax(-1).tolist()
        layers.append({
            "in_dim": layer.in_dim,
            "out_dim": layer.out_dim,
            "gate": gates,
            "gate_name": [GATE_NAMES[g] for g in gates],
            "a": layer.idx_a.tolist(),
            "b": layer.idx_b.tolist(),
        })

    # backward reachability from all final-layer outputs
    live = [set() for _ in layers]
    live[-1] = set(range(layers[-1]["out_dim"]))
    for li in range(len(layers) - 1, 0, -1):
        L = layers[li]
        needed = set()
        for n in live[li]:
            deps = _DEPS.get(L["gate"][n], ("a", "b"))
            if "a" in deps:
                needed.add(L["a"][n])
            if "b" in deps:
                needed.add(L["b"][n])
        live[li - 1] = needed
    for li, L in enumerate(layers):
        L["live"] = sorted(live[li])
    # which primary input bits the first layer actually reads
    used_inputs = set()
    L0 = layers[0]
    for n in live[0]:
        deps = _DEPS.get(L0["gate"][n], ("a", "b"))
        if "a" in deps:
            used_inputs.add(L0["a"][n])
        if "b" in deps:
            used_inputs.add(L0["b"][n])
    return layers, sorted(used_inputs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args()

    p1.CODE_BITS, p1.HIDDEN, p1.RESIDUAL_INIT = 12, 96, True
    g = torch.Generator().manual_seed(1000 + args.seed)
    model = GateAttentionA(p1.KEY_BITS, p1.VAL_BITS, p1.CODE_BITS, p1.HIDDEN,
                           share_qk=True, generator=g, residual_init=True)
    p1.train(model, True, args.steps, g)

    acc = {}
    for n in p1.EVAL_LENGTHS:
        soft, hard = p1.evaluate(model, True, n, g)
        acc[f"N={n}"] = {"soft": soft, "hard": hard}
    print("accuracies:", acc)

    layers, used_inputs = export_encoder(model.k_enc, p1.KEY_BITS)

    # hardened code table for all 64 keys
    keys = torch.arange(64)
    bits = p1.to_bits(keys, p1.KEY_BITS).bool()
    codes = model.k_enc.forward_hard(bits).long()
    code_ints = [int("".join(str(b) for b in row), 2) for row in codes.tolist()]
    n_unique = len(set(code_ints))

    out = {
        "config": {"key_bits": p1.KEY_BITS, "code_bits": p1.CODE_BITS,
                   "hidden": p1.HIDDEN, "seed": args.seed, "steps": args.steps,
                   "residual_init": True, "share_qk": True},
        "accuracy": acc,
        "encoder_layers": layers,
        "used_input_bits": used_inputs,
        "key_code_table": codes.tolist(),
        "unique_codes": n_unique,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "circuit_gateA_shared.json").write_text(json.dumps(out))
    live_counts = [len(L["live"]) for L in layers]
    print(f"live gates per layer: {live_counts} / {[L['out_dim'] for L in layers]}")
    print(f"unique codes over 64 keys: {n_unique}")
    print("wrote results/circuit_gateA_shared.json")

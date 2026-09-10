"""Phase 0: validate the differentiable-logic-gate pipeline.

1. Toy boolean functions (XOR-2, parity-4, 2:1 MUX): train soft, then
   discretize (argmax gate per neuron) and check the hardened circuit
   reproduces the function with pure bit operations.
2. MNIST (14x14 binarized): train a small gate network, report soft vs.
   hardened test accuracy — the difflogic "train -> freeze gates -> bitwise
   inference" loop, reproduced on CPU.

Usage: python experiments/phase0_sanity.py [--skip-mnist]
"""

import argparse
import gzip
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic import LogicLayer, GroupSum, GateEncoder

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


# ---------------------------------------------------------------- toy tasks

def make_toy(task: str, n: int, g: torch.Generator):
    x = torch.randint(0, 2, (n, 4), generator=g).float()
    if task == "xor2":
        y = (x[:, 0] != x[:, 1]).float()
    elif task == "parity4":
        y = (x.sum(-1) % 2)
    elif task == "mux":  # x0 selects between x1 and x2
        y = torch.where(x[:, 0] > 0.5, x[:, 1], x[:, 2])
    else:
        raise ValueError(task)
    return x, y


def run_toy(task: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    enc = GateEncoder([4, 32, 32, 1], generator=g)
    opt = torch.optim.Adam(enc.parameters(), lr=0.05)
    for step in range(1500):
        x, y = make_toy(task, 256, g)
        p = enc(x).squeeze(-1).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(p, y)
        opt.zero_grad(); loss.backward(); opt.step()
    x, y = make_toy(task, 4096, g)
    soft_acc = (((enc(x).squeeze(-1)) > 0.5) == y.bool()).float().mean().item()
    hard = enc.forward_hard(x.bool()).squeeze(-1)
    hard_acc = (hard == y.bool()).float().mean().item()
    agree = (hard == (enc(x).squeeze(-1) > 0.5)).float().mean().item()
    return {"task": task, "soft_acc": soft_acc, "hard_acc": hard_acc, "soft_hard_agreement": agree}


# ------------------------------------------------------------------- MNIST

def _fetch(name: str) -> Path:
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    if not path.exists():
        urllib.request.urlretrieve(MNIST_URL + name, path)
    return path


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">i", f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "i" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist_bits():
    xs, ys = {}, {}
    for split, prefix in [("train", "train"), ("test", "t10k")]:
        img = _read_idx(_fetch(f"{prefix}-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
        lab = _read_idx(_fetch(f"{prefix}-labels-idx1-ubyte.gz"))
        t = torch.from_numpy(img).reshape(-1, 1, 28, 28)
        t = F.avg_pool2d(t, 2).reshape(-1, 196)  # 14x14
        xs[split] = (t > 0.5).float()
        ys[split] = torch.from_numpy(lab.astype(np.int64))
    return xs, ys


def run_mnist(seed: int = 0, steps: int = 4000, batch: int = 128):
    g = torch.Generator().manual_seed(seed)
    xs, ys = load_mnist_bits()
    net = nn.Sequential(
        LogicLayer(196, 2000, g), LogicLayer(2000, 2000, g), LogicLayer(2000, 1600, g),
    )
    group = GroupSum(k=10, tau=10.0)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    n = xs["train"].shape[0]
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        logits = group(net(xs["train"][idx]))
        loss = F.cross_entropy(logits, ys["train"][idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  mnist step {step+1}/{steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")

    def evaluate(hard: bool) -> float:
        correct = 0
        xb_all, yb_all = xs["test"], ys["test"]
        for i in range(0, len(xb_all), 1024):
            xb = xb_all[i:i + 1024]
            if hard:
                h = xb.bool()
                for layer in net:
                    h = layer.forward_hard(h)
                pred = group(h.float()).argmax(-1)
            else:
                pred = group(net(xb)).argmax(-1)
            correct += (pred == yb_all[i:i + 1024]).sum().item()
        return correct / len(xb_all)

    return {"task": "mnist14x14", "soft_acc": evaluate(False), "hard_acc": evaluate(True),
            "gates": 2000 + 2000 + 1600, "steps": steps}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mnist", action="store_true")
    args = ap.parse_args()

    results = []
    for task in ["xor2", "parity4", "mux"]:
        r = run_toy(task)
        print(r)
        results.append(r)
    if not args.skip_mnist:
        r = run_mnist()
        print(r)
        results.append(r)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase0.json").write_text(json.dumps(results, indent=2))
    print("wrote results/phase0.json")

#!/usr/bin/env python3
"""PCFG run directory -> cached statistics the hierarchy metrics read.

This is the seam Exp 2 was waiting on. `sae-training` trains a Matryoshka SAE on a
PCFG base transformer and saves it beside the model; `metrics` grades parent->child
edges from a cached-statistics object. Nothing joined the two, and sae-training's
README lists exactly that as its remaining open item:

    Metrics Handoff (Exp 0): Ensure the saved SAE models can be seamlessly handed
    off to the experiment_0 repository for calculating post-training graph metrics.

It is deliberately thin. The accumulation is `metrics/collect_statistics.py`'s
collect(), unchanged -- this file only supplies a model, an SAE, a token stream and
a block structure, then gets out of the way. That is the whole point: if grading a
PCFG SAE needed different metric code, "the same battery across every source" would
stop being true and the paper's central claim would go with it.

    python3 adapters/from_pcfg.py --run-dir data/pcfg-run --layer 1 --out /tmp/stats.pt

Reads only. It never writes into the run directory, which on the compute node lives
under another user's account.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

UMBRELLA = Path(__file__).resolve().parent.parent


def _add_path(p: Path) -> None:
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# The adapter is the one place that depends on both repos, which is why it lives
# here rather than inside either of them.
_add_path(UMBRELLA / "metrics")
_add_path(UMBRELLA / "sae-training" / "src")


class PCFGMatryoshkaSAE:
    """The saved Matryoshka SAE, exposing the three things collect() needs.

    Trained with batch_topk, which keeps the k*batch largest pre-activations. That
    rule cannot be applied at inference without changing results as a function of
    batch composition, so training also tracks an EMA of the selection boundary and
    saves it as `threshold`; inference is then a plain JumpReLU against it. Using
    batch_topk here instead would make every statistic depend on BATCH_DOCS.
    """

    def __init__(self, weights: dict, cfg: dict, device: str = "cpu"):
        self.W_enc = weights["W_enc"].to(device)
        self.W_dec = weights["W_dec"].to(device)
        self.b_enc = weights["b_enc"].to(device)
        self.b_dec = weights["b_dec"].to(device)
        self.cfg = cfg
        thr = weights.get("threshold")
        self.threshold = float(thr) if thr is not None else 0.0
        self.activation = cfg.get("activation_function", "batch_topk")

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        if self.activation == "relu" or self.threshold <= 0:
            return pre
        return pre * (pre > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def block_ranges(latent_sizes):
    """Matryoshka blocks are nested prefixes, so the ranges are contiguous."""
    out, prev = [], 0
    for s in latent_sizes:
        out.append((prev, s))
        prev = s
    return out


def load_sae(sae_dir: Path, device: str):
    cfg = json.loads((sae_dir / "cfg.json").read_text())
    weights = load_file(str(sae_dir / "sae_weights.safetensors"))
    return PCFGMatryoshkaSAE(weights, cfg, device), cfg


def load_corpus_seqs(corpus_path: Path, context: int, n_docs: int, pad_id: int):
    """Cut the flat uint16 token stream into fixed-length windows.

    collect() drops position 0 of every sequence -- for gemma that is BOS, whose
    residual is an attention-sink outlier that would contaminate every count. A
    PCFG corpus has no BOS, so this costs one real token per window (~0.2% at
    context 512). Uniform across windows, so it biases nothing.
    """
    tokens = np.memmap(corpus_path, dtype=np.uint16, mode="r")
    usable = (tokens.shape[0] // context) if n_docs <= 0 else min(n_docs, tokens.shape[0] // context)
    if usable == 0:
        raise SystemExit(f"corpus too short: {tokens.shape[0]} tokens < context {context}")
    seqs = []
    for i in range(usable):
        w = np.asarray(tokens[i * context : (i + 1) * context], dtype=np.int64)
        if (w == pad_id).any():
            raise SystemExit(
                f"pad id {pad_id} occurs in the corpus; pick one outside the vocabulary "
                "or keep_mask will silently drop real tokens"
            )
        seqs.append(torch.from_numpy(w))
    return seqs


def make_cfg(sae_cfg: dict, layer: int, out_dir: Path, context: int):
    """Block structure and thresholds for THIS SAE, not gemma's.

    collect(cfg=...) exists for this: metrics/config.py hardcodes a 32768-feature
    dictionary in 5 blocks, and a PCFG SAE is 1792 in 8. Passing gemma's ranges
    would slice the dictionary at the wrong boundaries and still return numbers.
    """
    ranges = block_ranges(sae_cfg["latent_sizes"])
    hook = f"blocks.{layer}.hook_resid_post"
    return SimpleNamespace(
        LAYER=layer,
        HOOK_NAME=hook,
        SAE_ID=f"matryoshka_hook_resid_post_L{layer}",
        SAE_RELEASE="pcfg-matryoshka",
        SAE_SOURCE="pcfg",
        MATRYOSHKA_STEPS=sae_cfg["latent_sizes"],
        BLOCK_RANGES=ranges,
        N_BLOCKS=len(ranges),
        D_SAE=sae_cfg["d_sae"],
        # gemma skips B3->B4 because that block pair's accumulators dominate RAM
        # (24576 children). Here every block is 224 wide, so all pairs are cheap.
        INCLUDE_B3_B4=True,
        FIRE_THRESHOLD=1e-3,
        BATCH_DOCS=8,
        CONTEXT_SIZE=context,
        # Sibling redundancy and in-block edges over every block that has a parent.
        SIBLING_BLOCKS=list(range(1, len(ranges))),
        IN_BLOCK_BLOCKS=list(range(len(ranges) - 1)),
        N_FREQ_BUCKETS=3,
        FREQ_HIGH_MASS=0.50,
        FREQ_MID_MASS=0.40,
        MIN_JOINT=30,
        CACHE_RESIDUALS=False,
        TOKEN_CACHE_DIR=out_dir / "token_cache",
        EXP0_STATS_PATH=out_dir / "exp0_stats.pt",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="PCFG run dir: model.pt + corpus.bin + sae/")
    ap.add_argument("--layer", type=int, required=True, help="which trained SAE layer to grade")
    ap.add_argument("--out", type=Path, default=None, help="output .pt (default: <run-dir>/exp0_stats.pt)")
    ap.add_argument("--docs", type=int, default=64, help="corpus windows to use; 0 = all")
    ap.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    ap.add_argument("--pad-id", type=int, default=None, help="default: vocab_size-1, asserted absent")
    args = ap.parse_args()

    from collect_statistics import collect  # noqa: E402  (needs the sys.path above)
    from sae_training.pcfg import load_pcfg_model  # noqa: E402

    run_dir = args.run_dir
    sae_dir = run_dir / "sae" / f"matryoshka_hook_resid_post_L{args.layer}"
    if not sae_dir.is_dir():
        have = sorted(p.name for p in (run_dir / "sae").glob("*")) if (run_dir / "sae").is_dir() else []
        raise SystemExit(f"no SAE at {sae_dir}\navailable: {have or 'none'}")

    print(f"[pcfg] run  : {run_dir}")
    model, model_cfg = load_pcfg_model(run_dir, device=args.device)
    sae, sae_cfg = load_sae(sae_dir, args.device)

    context = int(model_cfg["context_window"])
    pad_id = args.pad_id if args.pad_id is not None else int(model_cfg["vocab_size"]) - 1
    out_dir = args.out.parent if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(sae_cfg, args.layer, out_dir, context)

    grammar = {}
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        grammar = json.loads(manifest.read_text()).get("grammar", {}) or {}

    print(f"[pcfg] model: {model_cfg['n_layers']}L d_model={model_cfg['d_model']} vocab={model_cfg['vocab_size']}")
    print(f"[pcfg] sae  : d_sae={cfg.D_SAE} in {cfg.N_BLOCKS} blocks {cfg.MATRYOSHKA_STEPS}")
    print(f"[pcfg] act  : {sae.activation}, threshold={sae.threshold:.5g}")
    if grammar:
        print(f"[pcfg] zipf={grammar.get('zipf_exponent')} sections={grammar.get('sections')}")

    seqs = load_corpus_seqs(run_dir / "corpus.bin", context, args.docs, pad_id)
    print(f"[pcfg] corpus: {len(seqs)} windows x {context} tokens\n")

    collect(
        model,
        sae,
        seqs,
        device=args.device,
        cfg=cfg,
        out_path=args.out,
        pad_id=pad_id,
        extra_config={
            "source": "pcfg",
            "run_dir": str(run_dir),
            "n_docs": len(seqs),
            "grammar": grammar,
            "base_model": model_cfg,
            "sae_cfg": sae_cfg,
        },
    )

    # Fail loudly here rather than let a malformed object reach the metrics, which
    # would return plausible numbers computed from the wrong tensors.
    sys.path.insert(0, str(UMBRELLA / "contracts"))
    from validate_stats import validate  # noqa: E402

    stats = torch.load(args.out or cfg.EXP0_STATS_PATH, map_location="cpu", weights_only=False)
    rep = validate(stats)
    for e in rep.errors:
        print(f"  ERROR {e}")
    if rep.errors:
        print(f"\n[pcfg] FAIL - {len(rep.errors)} contract violation(s)")
        return 1
    print(f"[pcfg] contract OK (shape={rep.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

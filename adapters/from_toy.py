#!/usr/bin/env python3
"""Trained-toy SAE -> cached statistics, through the same path gemma and PCFG take.

Tier 2 already reports precision 1.00 / recall 0.67 on this checkpoint, but it gets
there by calling the metric functions directly with thresholds written into the
script. It never touches collect() or run_metrics.py, so the one published number
that could validate the whole pipeline validates only the metrics.

This routes the same toy through the production path. If it lands on the same
figures, that number now covers the accumulation and the report as well.

What made this awkward until now: the toy's groups are not contiguous. It indexes by
*which true feature each latent recovered* (decoder cosine >= 0.4), so parents and
children are scattered index lists like [0, 3, 8]. collect() used to slice blocks as
ranges. It now accepts `cfg.BLOCK_INDICES`, so no permutation of the dictionary is
needed -- which matters, because permuting would have made "B0->B1" mean "true
parents -> true children" here and "first 128 -> next 384" on gemma, one field name
carrying two meanings.

    python3 adapters/from_toy.py --out /tmp/toy_stats.pt

Reads a checkpoint dir (--ckpt, default `metrics/outputs/toy_trained/`) and
`sae-training/configs/tree.json` for the ground-truth tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

UMBRELLA = Path(__file__).resolve().parent.parent
for p in (UMBRELLA / "metrics", UMBRELLA / "contracts"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from validation.calibrate_on_trained_toy import (  # noqa: E402
    build_tree, match_latents, n_features, sample, true_edges,
)


def load_checkpoint(ckpt_dir: Path, n_true_features: int):
    """Read a trained-toy checkpoint, and refuse one that is not for this toy.

    Tier 2's own loader hardcodes `metrics/outputs/toy_trained`. Retraining the toy
    -- which the Aug-1 defaults would make a different SAE, `relu` in place of
    `batch_topk` among other changes -- writes somewhere else, so `--ckpt` has to be
    able to follow it.

    Which means the path can now be wrong, so the shape has to be checked here. A
    checkpoint from another source with a matching `d_in` would otherwise sail past,
    match latents against a tree it never saw, and produce a complete report of
    meaningless edges. Pointing at a PCFG SAE happens to raise inside `match_latents`
    because 1792x448 cannot meet 20x20 -- but that is the dimensions colliding by
    luck, not a check.
    """
    from safetensors.torch import load_file
    ckpt_dir = Path(ckpt_dir)
    missing = [f for f in ("sae_weights.safetensors", "cfg.json") if not (ckpt_dir / f).is_file()]
    if missing:
        raise SystemExit(f"{ckpt_dir}: missing {', '.join(missing)}")

    w = load_file(str(ckpt_dir / "sae_weights.safetensors"))
    cfg = json.loads((ckpt_dir / "cfg.json").read_text())

    absent = [k for k in ("W_enc", "W_dec", "b_enc", "b_dec") if k not in w]
    if absent:
        raise SystemExit(f"{ckpt_dir}: weights missing {', '.join(absent)}")

    d_in = int(w["W_dec"].shape[1])
    if d_in != n_true_features:
        raise SystemExit(
            f"{ckpt_dir}: this SAE has d_in={d_in}, but the toy tree in "
            f"sae-training/configs/tree.json has {n_true_features} read-out features.\n"
            f"      That is a checkpoint for a different world — matching its latents "
            f"against this tree would produce edges that mean nothing."
        )
    return w, cfg


class LookupModel:
    """The toy has no transformer. Its 'residual' is the ground-truth activation
    vector itself, so the model is a lookup table over distinct activation patterns.

    Each distinct pattern gets a token id. That is exact rather than a proxy: two
    positions share an id exactly when they are the same world state, so the
    frequency buckets are over genuine repeat structure. `calibrate_on_trained_toy`
    instead uses `fired.argmax(1)` as a stand-in token, which is coarser -- worth
    knowing when comparing metric-5 numbers between the two paths.
    """

    def __init__(self, patterns: torch.Tensor, hook: str):
        self.emb = patterns                      # [vocab, F]
        self.hook = hook
        self.cfg = SimpleNamespace(d_vocab=patterns.shape[0], d_model=patterns.shape[1])
        self.tokenizer = SimpleNamespace(pad_token_id=None)

    def run_with_cache(self, tokens, stop_at_layer=None, names_filter=None):
        return None, {self.hook: self.emb[tokens]}


class ToySAE:
    """The saved checkpoint, exposing what collect() needs.

    batch_topk cannot be replayed at inference without making every statistic a
    function of batch composition, and this checkpoint saves the EMA threshold, so
    inference is a JumpReLU against it.
    """

    def __init__(self, w: dict, cfg: dict):
        self.W_enc, self.W_dec = w["W_enc"], w["W_dec"]
        self.b_enc, self.b_dec = w["b_enc"], w["b_dec"]
        thr = w.get("threshold")
        self.threshold = float(thr) if thr is not None else 0.0
        self.activation = cfg.get("activation_function", "batch_topk")

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        if self.activation == "relu" or self.threshold <= 0:
            return pre
        return pre * (pre > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def distinct_patterns(gt: torch.Tensor):
    """[n, F] binary rows -> (vocab table [V, F], token id per row [n])."""
    uniq, inverse = torch.unique(gt, dim=0, return_inverse=True)
    return uniq, inverse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/toy_stats.pt"))
    ap.add_argument("--ckpt", type=Path, default=UMBRELLA / "metrics" / "outputs" / "toy_trained",
                    help="a trained-toy checkpoint dir (sae_weights.safetensors + cfg.json)")
    ap.add_argument("--samples", type=int, default=200_000, help="world draws (Tier 2 uses 200k)")
    ap.add_argument("--context", type=int, default=128, help="rows per pseudo-document")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from collect_statistics import collect  # noqa: E402

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    tree = build_tree()
    truth = true_edges(tree)
    F = n_features(tree)
    w, sae_cfg = load_checkpoint(args.ckpt, F)
    print(f"[toy] checkpoint: {args.ckpt}")
    match = match_latents(w, torch.eye(F))

    parents = sorted({p for p, _ in truth})
    children = sorted({c for _, c in truth})
    m = match.tolist()
    parent_lat = [i for i, t in enumerate(m) if t in parents]
    child_lat = [i for i, t in enumerate(m) if t in children]
    print(f"[toy] {F} true features, {len(truth)} true edges")
    print(f"[toy] latents matched to parents {parent_lat}, to children {child_lat}")
    if not parent_lat or not child_lat:
        raise SystemExit("SAE recovered no parents or no children")

    gt = sample(tree, args.samples, F, gen)          # [n, F] ground-truth firings
    vocab_table, token_ids = distinct_patterns(gt)
    print(f"[toy] {args.samples:,} draws -> {vocab_table.shape[0]} distinct world states")

    # collect() drops position 0 of every sequence, so prepend a repeated row rather
    # than lose a real draw: the same id everywhere, dropped everywhere.
    ctx = args.context
    usable = (token_ids.numel() // ctx) * ctx
    seqs = [torch.cat([token_ids[i:i + 1], token_ids[i:i + ctx]])
            for i in range(0, usable, ctx)]

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    hook = "blocks.0.hook_resid_post"
    cfg = SimpleNamespace(
        LAYER=0, HOOK_NAME=hook, SAE_ID="toy_trained",
        SAE_RELEASE="matryoshka-toy", SAE_SOURCE="trained_toy",
        MATRYOSHKA_STEPS=sae_cfg.get("latent_sizes"),
        BLOCK_RANGES=None,
        # Not contiguous: these are the latents that recovered true parents and true
        # children, in dictionary order. See contracts/stats_schema.md.
        BLOCK_INDICES=[parent_lat, child_lat],
        N_BLOCKS=2, D_SAE=int(sae_cfg["d_sae"]), INCLUDE_B3_B4=True,
        FIRE_THRESHOLD=1e-3, BATCH_DOCS=64, CONTEXT_SIZE=ctx,
        SIBLING_BLOCKS=[1], IN_BLOCK_BLOCKS=[0],
        N_FREQ_BUCKETS=3, FREQ_HIGH_MASS=0.50, FREQ_MID_MASS=0.40,
        MIN_JOINT=20, CACHE_RESIDUALS=False,
        TOKEN_CACHE_DIR=out_dir / "token_cache", EXP0_STATS_PATH=args.out,
    )

    collect(
        LookupModel(vocab_table.float(), hook), ToySAE(w, sae_cfg), seqs,
        device="cpu", cfg=cfg, out_path=args.out,
        pad_id=vocab_table.shape[0],          # one past the vocabulary; no padding is emitted
        extra_config={
            "source": "trained_toy", "n_samples": args.samples,
            "true_edges": sorted(truth), "match": m,
            "parent_latents": parent_lat, "child_latents": child_lat,
        },
    )

    from validate_stats import validate  # noqa: E402
    rep = validate(torch.load(args.out, map_location="cpu", weights_only=False))
    for e in rep.errors:
        print(f"  ERROR {e}")
    if rep.errors:
        print(f"[toy] FAIL — {len(rep.errors)} contract violation(s)")
        return 1
    print(f"[toy] contract OK (shape={rep.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

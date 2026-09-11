#!/usr/bin/env python3
"""Released Temporal SAE on gemma-2-2b layer 12 -> cached statistics.

The point of this file is that the metrics do not change. `collect()` is source-agnostic:
give it a model, an object with `.encode` / `.decode` / `.W_dec`, and token sequences, and it
returns a stats file the rest of the pipeline reads without modification. This adapter supplies
those three things for a T-SAE we did not train, exactly as `from_pcfg.py` does for a PCFG SAE.

Two things here are specific to this checkpoint and easy to get wrong.

**The layer.** The released config records `layer: 12` in `dictionary_learning`'s convention,
which means the *output of block 12*. TransformerLens names that `blocks.12.hook_resid_post`,
which is unambiguous, but HuggingFace would call the same tensor `hidden_states[13]`. We
confirmed the depth by reconstruction rather than by reading the convention: the checkpoint
peaks at FVE 0.756 there against 0.652 one block earlier. See `findings/I6-F007`.

**The blocks.** A T-SAE is a two-block Matryoshka, `group_sizes [3276, 13108]`, not gemma's five
nested prefixes. Passing `metrics/config.py`'s ranges would slice this dictionary at the wrong
boundaries and still produce a full, plausible report, which is the failure mode this project
keeps finding. The ranges below come from the checkpoint itself.

    python3 adapters/from_tsae_gemma.py --ckpt data/temporal-saes-gemma-l12 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

UMBRELLA = Path(__file__).resolve().parent.parent


def _add_path(p: Path) -> None:
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# The metrics checkout is a sibling here, but on the GPU node it is `metrics-current`.
# EXP0_METRICS overrides the location so the same file runs in both places.
import os                                                              # noqa: E402
_add_path(Path(os.environ.get("EXP0_METRICS", UMBRELLA / "metrics")))


class ReleasedTemporalSAE:
    """The published checkpoint, exposing the three things `collect()` needs.

    Trained with BatchTopK, which keeps the k largest pre-activations across a whole batch. That
    rule cannot be replayed at inference without making every statistic a function of batch
    composition, so training tracked an EMA of the selection boundary and saved it as
    `threshold`. Inference is a plain JumpReLU against it, matching how the reference trainer
    evaluates and how `from_pcfg.py` treats the same situation.
    """

    def __init__(self, weights: dict, device: str = "cpu"):
        self.W_enc = weights["W_enc"].to(device).float()
        self.W_dec = weights["W_dec"].to(device).float()
        self.b_enc = weights["b_enc"].to(device).float()
        self.b_dec = weights["b_dec"].to(device).float()
        self.threshold = float(weights["threshold"])
        self.k = int(weights["k"])
        self.group_sizes = [int(g) for g in weights["group_sizes"]]
        if self.threshold <= 0:
            raise SystemExit(
                f"checkpoint threshold is {self.threshold}, which every consumer reads as "
                f"'no sparsity' and grades a dense dictionary without warning."
            )

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return pre * (pre > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def block_ranges(group_sizes):
    """Matryoshka blocks are nested prefixes, so the ranges are contiguous."""
    out, prev = [], 0
    for g in group_sizes:
        out.append((prev, prev + g))
        prev += g
    return out


def make_cfg(sae, layer: int, out_dir: Path, context: int, n_docs: int):
    ranges = block_ranges(sae.group_sizes)
    return SimpleNamespace(
        LAYER=layer,
        HOOK_NAME=f"blocks.{layer}.hook_resid_post",
        SAE_ID=f"tsae_hook_resid_post_L{layer}",
        SAE_RELEASE="gemma-2-2b-tsae",
        SAE_SOURCE="gemma-tsae",
        MATRYOSHKA_STEPS=[r[1] for r in ranges],
        BLOCK_RANGES=ranges,
        N_BLOCKS=len(ranges),
        D_SAE=sum(sae.group_sizes),
        # Only one block pair exists here, 0->1, so the gemma-specific B3->B4 guard is moot.
        INCLUDE_B3_B4=True,
        FIRE_THRESHOLD=1e-3,
        BATCH_DOCS=8,
        CONTEXT_SIZE=context,
        SIBLING_BLOCKS=list(range(1, len(ranges))),
        IN_BLOCK_BLOCKS=list(range(len(ranges) - 1)),
        N_FREQ_BUCKETS=3,
        FREQ_HIGH_MASS=0.50,
        FREQ_MID_MASS=0.40,
        LOCAL_FREQ_BUCKETS=False,
        MIN_JOINT=30,
        # The second block holds 13,108 features at d_model 2304. The residual cache is what
        # S_res runs off, and it is the strict test; keep it unless disk is the binding limit.
        CACHE_RESIDUALS=True,
        TOKEN_CACHE_DIR=out_dir / "token_cache",
        EXP0_STATS_PATH=out_dir / "exp0_stats.pt",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="dir with ae.pt and config.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <ckpt>/exp0")
    ap.add_argument("--docs", type=int, default=400, help="pile-10k documents")
    ap.add_argument("--context", type=int, default=128, help="tokens per document")
    ap.add_argument("--device", default="cuda", help="cpu / cuda")
    ap.add_argument("--no-token-cache", action="store_true",
                    help="skip the residual cache; S_res then cannot run for this SAE")
    args = ap.parse_args()

    sd = torch.load(args.ckpt / "ae.pt", map_location="cpu")
    ref = json.loads((args.ckpt / "config.json").read_text())["trainer"]
    layer = int(ref["layer"])
    sae = ReleasedTemporalSAE(sd, args.device)

    if sum(sae.group_sizes) != int(ref["dict_size"]):
        raise SystemExit(f"group_sizes {sae.group_sizes} do not sum to dict_size "
                         f"{ref['dict_size']}; the block partition would be wrong.")

    out_dir = args.out_dir or (args.ckpt / "exp0")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(sae, layer, out_dir, args.context, args.docs)
    if args.no_token_cache:
        cfg.CACHE_RESIDUALS = False

    print(f"T-SAE  d_sae {cfg.D_SAE}  k {sae.k}  threshold {sae.threshold:.5f}")
    print(f"blocks {sae.group_sizes} -> ranges {cfg.BLOCK_RANGES}")
    print(f"reading {cfg.HOOK_NAME} of {ref['lm_name']}")

    from transformer_lens import HookedTransformer                     # noqa: E402
    from datasets import load_dataset                                  # noqa: E402
    from collect_statistics import collect                             # noqa: E402

    model = HookedTransformer.from_pretrained(
        ref["lm_name"], device=args.device, center_writing_weights=False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    seqs = []
    for i in range(args.docs):
        ids = model.to_tokens(ds[i]["text"], truncate=True)[0][: args.context]
        if ids.numel() > 1:
            seqs.append(ids.cpu())
    print(f"{len(seqs)} sequences, context {args.context}")

    collect(model, sae, seqs, device=args.device, cfg=cfg,
            out_path=cfg.EXP0_STATS_PATH,
            extra_config={"source": "gemma-tsae", "sae_source": "gemma-tsae",
                          "n_docs": len(seqs), "checkpoint": "alex-oesterling/temporal-saes",
                          "layer_convention": "blocks.N.hook_resid_post = output of block N; "
                                              "equals HF hidden_states[N+1]"})
    print(f"wrote {cfg.EXP0_STATS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

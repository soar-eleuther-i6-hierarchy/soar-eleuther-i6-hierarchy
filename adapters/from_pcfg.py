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

To publish a run as a page instead of a scratch file, write into the metrics site's
output tree and run the remaining stages against it. EXP0_RUN names the directory,
because this is not a gemma layer:

    export EXP0_RUN=pcfg
    python3 adapters/from_pcfg.py --run-dir data/pcfg-run --layer 1 \\
            --out metrics/outputs/pcfg/exp0_stats.pt
    cd metrics
    python3 run_metrics.py --stats outputs/pcfg/exp0_stats.pt --out-dir outputs/pcfg
    python3 run_token_metrics.py          # S_res, off the token cache + w_dec.pt
    python3 -m reporting.visualize        # -> outputs/pcfg/*.html

Keep the run directly under outputs/: the nav bar and the shared plotly bundle both
assume one level, and a `layer_NN` name would claim a gemma layer this is not.
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


EOS_ID = 1000


def document_starts(tokens: np.ndarray, sentences_per_doc: int) -> np.ndarray | None:
    """Offsets where each document begins, or None when the stream has no markers.

    The generator emits `EOS` after every sentence and a document is exactly
    `sections × paragraphs_per_section × sentences_per_paragraph` sentences, so every
    Nth EOS ends a document. With `formatting.eos` off there is no marker at all and
    documents cannot be located.
    """
    eos = np.flatnonzero(tokens == EOS_ID)
    if eos.size < sentences_per_doc:
        return None
    ends = eos[sentences_per_doc - 1 :: sentences_per_doc]
    return np.concatenate([[0], ends[:-1] + 1])


def load_corpus_seqs(corpus_path: Path, context: int, n_docs: int, pad_id: int,
                     sentences_per_doc: int | None = None):
    """Cut the token stream into windows that lie **inside a single document**.

    This matters for the within-context frequency control and for nothing else. The
    generator re-permutes which token id holds which Zipf rank per document, so
    "frequent in this context" is only meaningful inside one document — a window
    spanning a boundary mixes two permutations and dilutes exactly the concentration
    the zipf knob creates. Documents here run ~385 tokens against a 512-token context,
    so naive fixed windows crossed a boundary 100% of the time.

    Windows start at document boundaries and share one length, so no padding is ever
    emitted (there is no spare id in this vocabulary to pad with). Documents longer
    than that length are truncated; the tail is dropped, uniformly.

    collect() drops position 0 of every sequence -- for gemma that is BOS, whose
    residual is an attention-sink outlier that would contaminate every count. A PCFG
    corpus has no BOS, so this costs one real token per window. Uniform, so it biases
    nothing.
    """
    tokens = np.asarray(np.memmap(corpus_path, dtype=np.uint16, mode="r"), dtype=np.int64)

    starts = document_starts(tokens, sentences_per_doc) if sentences_per_doc else None
    if starts is None or starts.size < 2:
        # No document markers (formatting.eos off): fall back to fixed windows and say
        # so, because the local frequency control is diluted in this mode.
        print("[pcfg] WARNING: no document markers — fixed windows; within-context "
              "frequency control will span document boundaries")
        span = context
        starts = np.arange(0, tokens.shape[0] - span, span)
    else:
        lengths = np.diff(np.concatenate([starts, [tokens.shape[0]]]))
        span = int(min(context, lengths[:-1].min()))
        print(f"[pcfg] documents: {starts.size} found, median {int(np.median(lengths[:-1]))} tokens "
              f"-> windows of {span} inside one document")

    usable = starts.size if n_docs <= 0 else min(n_docs, starts.size)
    seqs = []
    for s in starts[:usable]:
        w = tokens[s : s + span]
        if w.shape[0] < span:
            break
        if (w == pad_id).any():
            raise SystemExit(
                f"pad id {pad_id} occurs in the corpus; pick one outside the vocabulary "
                "or keep_mask will silently drop real tokens"
            )
        seqs.append(torch.from_numpy(w))
    return seqs


def make_cfg(sae_cfg: dict, layer: int, out_dir: Path, context: int, local_freq: bool = False,
             cache_residuals: bool = True):
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
        # Accumulate a second frequency control bucketed within each window. The
        # zipf knob concentrates tokens inside a document but not across the corpus,
        # so the global control is blind to it; this is what sees it.
        LOCAL_FREQ_BUCKETS=local_freq,
        MIN_JOINT=30,
        # On by default here, unlike gemma. S_res is the strict test -- the one
        # the survival numbers actually turn on -- and it runs only off the token
        # cache, so without this a PCFG run is gradeable on three of the five
        # filter stages and cannot be compared with a gemma layer on the one that
        # matters. Affordable at this scale: gemma's cache is ~700 MB/layer at
        # d_model=2304, the toy's is a few hundred MB of fp16 at d_model in the
        # low hundreds. --no-token-cache turns it off.
        CACHE_RESIDUALS=cache_residuals,
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
    ap.add_argument("--pad-id", type=int, default=None, help="default: vocab_size, asserted absent")
    ap.add_argument("--local-freq", action="store_true",
                    help="also accumulate the frequency control bucketed within each window")
    ap.add_argument("--no-token-cache", action="store_true",
                    help="skip the fp16 residual/latent cache; stage 03 (S_res, the strict "
                         "test) then cannot run for this SAE")
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
    # One past the vocabulary. Every window is exactly `context` long, so right_pad
    # emits no padding and this id never reaches the embedding -- it only has to be
    # absent from the data, because keep_mask drops every position equal to it.
    # vocab_size-1 is NOT safe: it is DOCUMENT_DELIM, present whenever that
    # formatting flag is on, and using it would silently delete every document
    # boundary from the statistics.
    pad_id = args.pad_id if args.pad_id is not None else int(model_cfg["vocab_size"])
    out_dir = args.out.parent if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(sae_cfg, args.layer, out_dir, context, local_freq=args.local_freq,
                   cache_residuals=not args.no_token_cache)

    grammar = {}
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        grammar = json.loads(manifest.read_text()).get("grammar", {}) or {}

    print(f"[pcfg] model: {model_cfg['n_layers']}L d_model={model_cfg['d_model']} vocab={model_cfg['vocab_size']}")
    print(f"[pcfg] sae  : d_sae={cfg.D_SAE} in {cfg.N_BLOCKS} blocks {cfg.MATRYOSHKA_STEPS}")
    print(f"[pcfg] act  : {sae.activation}, threshold={sae.threshold:.5g}")
    if grammar:
        print(f"[pcfg] zipf={grammar.get('zipf_exponent')} sections={grammar.get('sections')}")

    # A document is sections x paragraphs_per_section x sentences_per_paragraph
    # sentences, and every sentence ends with EOS -- that is how windows get aligned
    # to document boundaries.
    spd = None
    if grammar:
        try:
            spd = int(grammar["sections"]) * int(grammar["paragraphs_per_section"]) \
                  * int(grammar["sentences_per_paragraph"])
        except (KeyError, TypeError, ValueError):
            spd = None
    seqs = load_corpus_seqs(run_dir / "corpus.bin", context, args.docs, pad_id,
                            sentences_per_doc=spd)
    print(f"[pcfg] corpus: {len(seqs)} windows x {len(seqs[0])} tokens\n")

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

    # The second pass needs this SAE's decoder to turn a probe direction into
    # per-feature correlations, and it has no way to find a PCFG run on its own --
    # its default is the released gemma SAE. Dropping W_dec beside the statistics
    # is the whole hand-off: run_token_metrics picks up RUN_DIR/w_dec.pt without
    # being told, and the metrics repo stays free of PCFG layout knowledge.
    if not args.no_token_cache:
        w_dec_path = (args.out.parent if args.out else run_dir) / "w_dec.pt"
        torch.save(sae.W_dec.detach().cpu(), w_dec_path)
        print(f"[pcfg] decoder -> {w_dec_path}")

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

#!/usr/bin/env python3
"""Exp 2, axis (b): hierarchy quality against formatting density.

The project's hypothesis is that a nested capacity bottleneck spends its scarce
top-block capacity on high-frequency tokens instead of concepts. Testing that needs
an axis which actually varies *global* token frequency, and the frequency-control
metric buckets by global corpus counts.

Axis (a) does not vary it. `zipf_exponent` weights terminal *ranks*, and
`_permutations(doc_id)` re-randomises which token id holds which rank per document,
so the corpus-wide marginal stays near uniform: measured on the zipf=1.5 corpus, the
top 10 tokens are 59.2% within a document but 1.3% corpus-wide (uniform is 1.0%).

Axis (b) does. Delimiter tokens are globally frequent by construction, and the sweep
holds `zipf_exponent` at 1.0 while varying only which delimiters are emitted.

This script collects one row per run from reports already produced by
`adapters/from_pcfg.py` → `metrics/run_metrics.py`, aggregating over every block pair.

    python3 pipeline/formatting_sweep_table.py --data data/fmt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def summarise(run_dir: Path) -> dict | None:
    report = run_dir / "report" / "metrics_report.json"
    stats_path = run_dir / "exp0_stats.pt"
    if not report.is_file() or not stats_path.is_file():
        return None

    r = json.loads(report.read_text())
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    grammar = r["config"].get("grammar", {})

    # formatting_density is a sibling of `grammar` in the manifest, not inside it,
    # so the stats file's copy of the grammar block does not carry it.
    manifest = run_dir / "manifest.json"
    density = None
    if manifest.is_file():
        density = json.loads(manifest.read_text()).get("formatting_density")

    # Aggregate over block pairs. Ratios are the trustworthy signal: raw candidate
    # counts scale with how many features are alive, which varies between runs.
    cand = recon_pass = superparents = 0
    testable = freq_driven = 0
    surv_num = surv_den = 0.0
    for p in r["pairs"]:
        cand += p["n_candidate_edges"]
        recon_pass += p["reconstruction"]["n_pass"]
        superparents += p["n_superparents"]
        fc = p["freq_control"]
        testable += fc["n_testable"]
        freq_driven += fc["n_freq_driven"]
        if fc["mean_survival"] is not None and fc["n_testable"]:
            surv_num += fc["mean_survival"] * fc["n_testable"]
            surv_den += fc["n_testable"]

    corpus = run_dir / "corpus.bin"
    top_token_share = float("nan")
    if corpus.is_file():
        t = np.asarray(np.memmap(corpus, dtype=np.uint16, mode="r")[:2_000_000], np.int64)
        counts = np.bincount(t, minlength=1004)
        top_token_share = 100 * counts.max() / counts.sum()

    return {
        "run": run_dir.name,
        "formatting_density": density,
        "formatting": grammar.get("formatting", {}),
        "zipf_exponent": grammar.get("zipf_exponent"),
        "top_token_pct": top_token_share,
        "bucket0_ids": int((stats["buckets"] == 0).sum()),
        "alive_features": int((stats["fire_count"] > 0).sum()),
        "d_sae": int(r["config"]["sae_cfg"]["d_sae"]),
        "candidate_edges": cand,
        "recon_pass_frac": recon_pass / max(cand, 1),
        "freq_driven_frac": freq_driven / max(testable, 1),
        "mean_survival": (surv_num / surv_den) if surv_den else float("nan"),
        "n_superparents": superparents,
        "total_tokens": r["total_tokens"],
    }


def density_of(row: dict) -> float:
    """Delimiter density; recomputed from flags when the manifest did not carry it."""
    d = row.get("formatting_density")
    return float(d) if d is not None else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/fmt"), help="dir of graded run dirs")
    ap.add_argument("--json-out", type=Path, default=None, help="also write the rows as JSON")
    args = ap.parse_args()

    rows = [s for d in sorted(args.data.iterdir()) if d.is_dir() for s in [summarise(d)] if s]
    if not rows:
        raise SystemExit(f"no graded runs under {args.data} — run the adapter and run_metrics first")

    by_density = defaultdict(list)
    for row in rows:
        by_density[round(density_of(row), 4)].append(row)

    print(f"Exp 2 axis (b) — formatting density, zipf held at {rows[0]['zipf_exponent']}")
    print(f"{len(rows)} runs, {rows[0]['total_tokens']:,} tokens each, "
          f"d_sae={rows[0]['d_sae']}\n")
    head = f"{'density':>8} {'seeds':>6} {'topTok':>7} {'bkt0':>5} {'alive':>6} " \
           f"{'cand':>6} {'recon':>7} {'freq-drv':>9} {'surv':>7} {'super':>6}"
    print(head)
    print("-" * len(head))

    for density in sorted(by_density):
        g = by_density[density]
        def mean(k):
            v = [x[k] for x in g if x[k] == x[k]]
            return sum(v) / len(v) if v else float("nan")
        print(f"{density:>8.4f} {len(g):>6} {mean('top_token_pct'):>6.1f}% "
              f"{mean('bucket0_ids'):>5.0f} {mean('alive_features'):>6.0f} "
              f"{mean('candidate_edges'):>6.0f} {100*mean('recon_pass_frac'):>6.1f}% "
              f"{100*mean('freq_driven_frac'):>8.1f}% {mean('mean_survival'):>7.3f} "
              f"{mean('n_superparents'):>6.1f}")

    print("\nRead the ratios, not the counts: candidate_edges scales with how many")
    print("features are alive, which varies run to run. freq-drv and surv are the signal.")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

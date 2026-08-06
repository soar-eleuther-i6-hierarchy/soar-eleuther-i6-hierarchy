#!/usr/bin/env python3
"""Metric 4: what each candidate superparent gate actually admits.

`metrics/outdegree.py` records why the gate changed: the old
`AND(fan-out >= 30%, fires >= 10%)` let L24's feature 14 -- fires on 41.9% of tokens,
fans out to 21.9% of the child block -- go unflagged. But the fan-out-only gate that
replaced it does not flag feature 14 either, because it fails on fan-out and fan-out
is the criterion that was kept. Dropping the firing conjunct catches the opposite
leak (high fan-out, low firing). Promoting it instead -- an OR -- is the other way to
read that same note, and it is the option this script tests.

An OR is a claim about a classifier, so this tests it as one: it rebuilds each run's
edge mask at the shipped thresholds and applies all three gates to the same mask.

    outdeg-alone : outdeg >= SUPERPARENT_OUTDEG_FRAC * |child block|   (ships)
    AND          : outdeg-alone AND fire >= SUPERPARENT_FIRE_FRAC      (old)
    OR           : outdeg-alone OR  fire >= SUPERPARENT_FIRE_FRAC      (proposed)

The deciding column is `childless`: `outdegree.py` defines a superparent as "one
parent holding most of the next block's in-edges", so a gate admitting parents that
hold *none* is not detecting that pathology, whatever else it detects.

Counts are parent-slots summed over block pairs -- a parent flagged in three pairs
counts three times, which is how `n_superparents` is already reported.

    python3 pipeline/superparent_gate_table.py --data data/fmt data/pcfg-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "metrics"))

import config as C  # noqa: E402
from metrics import coverage_legs, keep_edges  # noqa: E402


def gates_for_run(stats_path: Path) -> dict:
    """Apply the three gates to one graded run, aggregated over its block pairs."""
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    block_ranges = stats["config"]["block_ranges"]

    n_alone = n_and = n_or = 0
    add_fanouts: list[float] = []
    for key, cofire in stats["cofire"].items():
        p_blk, c_blk = (int(x) for x in key.split("->"))
        lo_p, hi_p = block_ranges[p_blk]
        lo_c, hi_c = block_ranges[c_blk]
        fire_p, fire_c = fire[lo_p:hi_p], fire[lo_c:hi_c]

        R, _ = coverage_legs(cofire.double(), fire_p, fire_c)
        edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                               cofire=cofire.double(), min_joint=C.MIN_JOINT)
        outdeg = edge_mask.sum(dim=1)
        n_children = edge_mask.shape[1]

        alone = outdeg >= C.SUPERPARENT_OUTDEG_FRAC * n_children
        hot = (fire_p / max(total, 1)) >= C.SUPERPARENT_FIRE_FRAC
        n_alone += int(alone.sum())
        n_and += int((alone & hot).sum())
        n_or += int((alone | hot).sum())

        adds = torch.nonzero(hot & ~alone).flatten()          # what OR adds over what ships
        add_fanouts += (outdeg[adds].double() / n_children).tolist()

    fanouts = torch.tensor(add_fanouts) if add_fanouts else torch.zeros(0)
    return {
        "run": stats_path.parent.name,
        "total_tokens": total,
        "d_sae": int(fire.numel()),
        "n_pairs": len(stats["cofire"]),
        "alone": n_alone,
        "and": n_and,
        "or": n_or,
        "adds": int(fanouts.numel()),
        "childless": int((fanouts == 0).sum()),
        "median_add_fanout": float(fanouts.median()) if fanouts.numel() else float("nan"),
        "max_add_fanout": float(fanouts.max()) if fanouts.numel() else float("nan"),
    }


def find_runs(roots: list[Path]) -> list[Path]:
    """A graded run is any dir holding an exp0_stats*.pt; prefer the full one."""
    out = []
    for root in roots:
        candidates = [root] if (root / "exp0_stats.pt").is_file() else sorted(
            d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
        for d in candidates:
            full, plain = d / "exp0_stats_full.pt", d / "exp0_stats.pt"
            if full.is_file():
                out.append(full)
            elif plain.is_file():
                out.append(plain)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+",
                    default=[Path("data/fmt"), Path("data/pcfg-run")],
                    help="dirs of graded runs, or a graded run dir itself")
    args = ap.parse_args()

    runs = find_runs(args.data)
    if not runs:
        raise SystemExit(f"no graded runs under {args.data} — run adapters/from_pcfg.py first")

    print(f"Superparent gates at outdeg >= {C.SUPERPARENT_OUTDEG_FRAC:.0%}, "
          f"fire >= {C.SUPERPARENT_FIRE_FRAC:.0%} (edge tau {C.EDGE_TAU})\n")
    head = (f"{'run':<22} {'tokens':>10} {'alone':>6} {'AND':>5} {'OR':>6} "
            f"{'OR adds':>8} {'childless':>10} {'maxFan':>8}")
    print(head)
    print("-" * len(head))

    agg = {"alone": 0, "and": 0, "or": 0, "adds": 0, "childless": 0, "pairs": 0}
    for stats_path in runs:
        r = gates_for_run(stats_path)
        for k in ("alone", "and", "or", "adds", "childless"):
            agg[k] += r[k]
        agg["pairs"] += r["n_pairs"]
        print(f"{r['run']:<22} {r['total_tokens']:>10,} {r['alone']:>6} {r['and']:>5} "
              f"{r['or']:>6} {r['adds']:>8} {r['childless']:>10} "
              f"{r['max_add_fanout']:>7.1%}")

    print("-" * len(head))
    pct = 100.0 * agg["childless"] / agg["adds"] if agg["adds"] else 0.0
    print(f"{'TOTAL':<22} {'':>10} {agg['alone']:>6} {agg['and']:>5} {agg['or']:>6} "
          f"{agg['adds']:>8} {agg['childless']:>10}")
    print(f"\n{len(runs)} runs, {agg['pairs']} block pairs. "
          f"{pct:.1f}% of what OR adds has no children at all.")
    print("A parent holding no in-edges is not a superparent under metrics/outdegree.py's")
    print("own definition, so OR is a firing-rate threshold wearing the superparent label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

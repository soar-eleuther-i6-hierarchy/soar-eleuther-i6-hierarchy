#!/usr/bin/env python3
"""Validate a cached-statistics object against the Exp-0 stats contract.

The hierarchy metrics are pure functions over cached statistics, which is what lets
the same metric code grade a gemma-2-2b SAE, a trained toy SAE and a PCFG SAE. That
property is only real if every producer emits the same shape — this script is what
makes it checkable.

There are TWO shapes in the codebase, and both are legitimate:

  full   — what collect_statistics.py writes to exp0_stats.pt and run_metrics.py
           loads. Covers every block pair at once, so the per-pair accumulators are
           dicts keyed "{parent}->{child}" and the per-child ones are keyed by block
           index. Carries schema_version / config / pairs.

  slice  — one block pair, flattened: cofire is a bare [P, C] tensor, not a dict.
           This is what validation/toy_world.py emits and what every metric function
           actually consumes; run_metrics.analyse_pair() cuts a slice out of a full
           file before calling them.

The mode is auto-detected. A PCFG/TinyStories adapter should emit `full`, so that
run_metrics.py runs against it unmodified.

    python3 contracts/validate_stats.py metrics/outputs/layer_06/exp0_stats.pt
    python3 contracts/validate_stats.py --self-test        # checks a toy slice

Exit 0 = conforms. Exit 1 = does not (every violation is printed, not just the first).
Only torch is required; it deliberately does not import the metrics package, so an
adapter can be checked without a working metrics environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCHEMA_VERSION = 2

# --- full-file shape --------------------------------------------------------
FULL_REQUIRED = [
    "schema_version",
    "fire_count",
    "total_tokens",
    "token_counts",
    "buckets",
    "pairs",
    "cofire",
    "cofire_by_bucket",
    "g_parent_sum",
    "err_sum_c",
    "g_child_sum",
    "fire_c_by_bucket",
    "within_cofire",
    "config",
    # v2 extras: exact joint-child coverage (R_supp / R_mass) + energy shares.
    "energy_cofire",
    "energy_total",
    "union_count",
    "union_energy",
]

FULL_REQUIRED_CONFIG = [
    "layer",
    "sae_id",
    "matryoshka_steps",
    "block_ranges",
    "fire_threshold",
    "context_size",
    "sibling_blocks",
    "min_joint",
    "bos_excluded",
]

PAIR_KEYED_2D = ["cofire", "g_parent_sum", "energy_cofire"]      # [P, C] per pair
PAIR_KEYED_1D = ["union_count", "union_energy", "energy_total"]  # [P] per pair
CHILD_KEYED_1D = ["err_sum_c", "g_child_sum"]                    # [C] per child block

# --- pair-slice shape -------------------------------------------------------
SLICE_REQUIRED = [
    "P",
    "C",
    "fire_p",
    "fire_c",
    "fire_count",
    "total_tokens",
    "token_counts",
    "buckets",
    "cofire",
    "cofire_by_bucket",
    "g_parent_sum",
    "err_sum_c",
    "g_child_sum",
    "fire_c_by_bucket",
    "within_cofire",
    "energy_cofire",
    "energy_total",
    "union_count",
    "union_energy",
]


class Report:
    def __init__(self, mode: str = "?") -> None:
        self.mode = mode
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def detect_mode(stats: dict) -> str:
    if "schema_version" in stats or "config" in stats or "pairs" in stats:
        return "full"
    if "P" in stats and "C" in stats:
        return "slice"
    return "unknown"


def block_lengths(block_ranges) -> list[int]:
    return [int(end) - int(start) for start, end in block_ranges]


# Reconstruction gains are signed: ablating a feature can *improve* the
# reconstruction, which is a negative gain and a real measurement, not corruption.
# Everything else here is a count or a sum of magnitudes and must stay >= 0.
SIGNED = {"g_parent_sum", "g_child_sum"}


def check_tensor(rep: Report, obj, name: str, shape: tuple[int, ...] | None = None) -> None:
    """Assert obj is a tensor of the expected shape and finite; counts must be >= 0."""
    if not isinstance(obj, torch.Tensor):
        rep.err(f"{name}: expected Tensor, got {type(obj).__name__}")
        return
    if shape is not None and tuple(obj.shape) != shape:
        rep.err(f"{name}: expected shape {shape}, got {tuple(obj.shape)}")
        return
    if obj.numel() == 0:
        rep.warn(f"{name}: empty tensor")
        return
    if obj.is_floating_point():
        if not torch.isfinite(obj).all():
            rep.err(f"{name}: contains NaN or Inf")
        signed = name.split("[")[0] in SIGNED
        if not signed and (obj < 0).any():
            rep.err(f"{name}: contains negative values (counts/sums must be >= 0)")


def check_common(rep: Report, stats: dict) -> int:
    """Corpus-level fields shared by both shapes. Returns the bucket count."""
    total = stats.get("total_tokens")
    if not isinstance(total, int) or total <= 0:
        rep.err(f"total_tokens: expected positive int, got {total!r}")

    counts, buckets = stats.get("token_counts"), stats.get("buckets")
    check_tensor(rep, counts, "token_counts")
    check_tensor(rep, buckets, "buckets")
    if isinstance(counts, torch.Tensor) and isinstance(buckets, torch.Tensor):
        if counts.shape != buckets.shape:
            rep.err(
                f"token_counts {tuple(counts.shape)} and buckets {tuple(buckets.shape)} "
                "must match (both [vocab])"
            )

    if isinstance(buckets, torch.Tensor) and buckets.numel():
        return int(buckets.max()) + 1
    return 0


def validate_slice(stats: dict) -> Report:
    rep = Report("slice")

    for k in SLICE_REQUIRED:
        if k not in stats:
            rep.err(f"missing required key: {k!r}")
    if rep.errors:
        return rep

    P, C = stats["P"], stats["C"]
    if not (isinstance(P, int) and isinstance(C, int) and P > 0 and C > 0):
        rep.err(f"P/C: expected positive ints, got {P!r}/{C!r}")
        return rep

    K = check_common(rep, stats)

    check_tensor(rep, stats["fire_p"], "fire_p", (P,))
    check_tensor(rep, stats["fire_c"], "fire_c", (C,))
    check_tensor(rep, stats["fire_count"], "fire_count", (P + C,))

    check_tensor(rep, stats["cofire"], "cofire", (P, C))
    check_tensor(rep, stats["g_parent_sum"], "g_parent_sum", (P, C))
    check_tensor(rep, stats["energy_cofire"], "energy_cofire", (P, C))
    check_tensor(rep, stats["cofire_by_bucket"], "cofire_by_bucket", (K, P, C))

    for name in ("energy_total", "union_count", "union_energy"):
        check_tensor(rep, stats[name], name, (P,))

    check_tensor(rep, stats["err_sum_c"], "err_sum_c", (C,))
    check_tensor(rep, stats["g_child_sum"], "g_child_sum", (C,))
    check_tensor(rep, stats["fire_c_by_bucket"], "fire_c_by_bucket", (K, C))
    check_tensor(rep, stats["within_cofire"], "within_cofire", (C, C))

    # A pair can never co-fire more often than either side fires on its own.
    cof, fp, fc = stats["cofire"], stats["fire_p"], stats["fire_c"]
    if all(isinstance(t, torch.Tensor) for t in (cof, fp, fc)) and tuple(cof.shape) == (P, C):
        if (cof.double() > fp.double()[:, None] + 1e-6).any():
            rep.err("cofire: exceeds the parent's own fire count")
        if (cof.double() > fc.double()[None, :] + 1e-6).any():
            rep.err("cofire: exceeds the child's own fire count")

    # The bucketed counts must sum back to the plain ones — this is what catches an
    # adapter that bucketed on the wrong vocabulary.
    cbb = stats["cofire_by_bucket"]
    if isinstance(cbb, torch.Tensor) and tuple(cbb.shape) == (K, P, C) and tuple(cof.shape) == (P, C):
        if not torch.allclose(cbb.double().sum(0), cof.double(), atol=1e-4, rtol=1e-4):
            rep.err("cofire_by_bucket: does not sum over buckets back to cofire")

    return rep


def validate_full(stats: dict) -> Report:
    rep = Report("full")

    for k in FULL_REQUIRED:
        if k not in stats:
            rep.err(f"missing required key: {k!r}")

    version = stats.get("schema_version")
    if version != SCHEMA_VERSION:
        rep.err(f"schema_version: expected {SCHEMA_VERSION}, got {version!r}")

    cfg = stats.get("config")
    if not isinstance(cfg, dict):
        rep.err("config: missing or not a dict — cannot check shapes")
        return rep
    for k in FULL_REQUIRED_CONFIG:
        if k not in cfg:
            rep.err(f"config: missing field {k!r}")

    block_ranges = cfg.get("block_ranges")
    if not block_ranges:
        rep.err("config.block_ranges: missing — this is what makes the object source-agnostic")
        return rep

    blen = block_lengths(block_ranges)
    d_sae = int(block_ranges[-1][1])
    n_blocks = len(blen)

    K = check_common(rep, stats)
    check_tensor(rep, stats.get("fire_count"), "fire_count", (d_sae,))

    pairs = stats.get("pairs")
    if not isinstance(pairs, (list, tuple)) or not pairs:
        rep.err(f"pairs: expected a non-empty list of (parent, child), got {pairs!r}")
        return rep

    for pr in pairs:
        if not (isinstance(pr, (tuple, list)) and len(pr) == 2):
            rep.err(f"pairs: malformed entry {pr!r}")
            continue
        p, c = int(pr[0]), int(pr[1])
        if not (0 <= p < n_blocks and 0 <= c < n_blocks):
            rep.err(f"pairs: block index out of range in {pr!r} (n_blocks={n_blocks})")
            continue

        key = f"{p}->{c}"
        for name, shape in [(n, (blen[p], blen[c])) for n in PAIR_KEYED_2D] + [
            (n, (blen[p],)) for n in PAIR_KEYED_1D
        ]:
            d = stats.get(name)
            if not isinstance(d, dict) or key not in d:
                rep.err(f"{name}: missing key {key!r}")
                continue
            check_tensor(rep, d[key], f"{name}[{key}]", shape)

        d = stats.get("cofire_by_bucket")
        if not isinstance(d, dict) or key not in d:
            rep.err(f"cofire_by_bucket: missing key {key!r}")
        else:
            check_tensor(rep, d[key], f"cofire_by_bucket[{key}]", (K, blen[p], blen[c]))

        cof, fire = stats.get("cofire"), stats.get("fire_count")
        if isinstance(cof, dict) and key in cof and isinstance(fire, torch.Tensor):
            m = cof[key]
            if isinstance(m, torch.Tensor) and tuple(m.shape) == (blen[p], blen[c]):
                fire_p = fire[block_ranges[p][0] : block_ranges[p][1]].double()
                if (m.double() > fire_p[:, None] + 1e-6).any():
                    rep.err(f"cofire[{key}]: co-firing exceeds the parent's own fire count")

    for b in sorted({int(c) for _, c in pairs}):
        for name in CHILD_KEYED_1D:
            d = stats.get(name)
            if not isinstance(d, dict) or b not in d:
                rep.err(f"{name}: missing block key {b}")
                continue
            check_tensor(rep, d[b], f"{name}[{b}]", (blen[b],))

        d = stats.get("fire_c_by_bucket")
        if not isinstance(d, dict) or b not in d:
            rep.err(f"fire_c_by_bucket: missing block key {b}")
        else:
            check_tensor(rep, d[b], f"fire_c_by_bucket[{b}]", (K, blen[b]))

    within = stats.get("within_cofire")
    if not isinstance(within, dict) or not within:
        rep.err("within_cofire: missing or empty — sibling redundancy and metric 7 need it")
    else:
        for b, m in within.items():
            b = int(b)
            if not (0 <= b < n_blocks):
                rep.err(f"within_cofire: block index {b} out of range")
                continue
            check_tensor(rep, m, f"within_cofire[{b}]", (blen[b], blen[b]))

    return rep


def validate(stats: dict) -> Report:
    mode = detect_mode(stats)
    if mode == "full":
        return validate_full(stats)
    if mode == "slice":
        return validate_slice(stats)
    rep = Report("unknown")
    rep.err(
        "cannot tell which shape this is: a full file needs schema_version/config/pairs, "
        "a pair slice needs P and C"
    )
    return rep


def self_test() -> int:
    """Check the validator against the synthetic toy, then against deliberate breakage.

    A validator that only ever passes is worthless, so this asserts both directions.
    """
    repo = Path(__file__).resolve().parent.parent / "metrics"
    sys.path.insert(0, str(repo))
    try:
        from validation.toy_world import build_world
    except ImportError as e:
        print(f"self-test SKIPPED — cannot import the toy world from {repo}: {e}")
        return 0

    stats, _ = build_world(seed=0)
    rep = validate(stats)
    print(f"[1] toy slice (seed 0), detected mode = {rep.mode}")
    for e in rep.errors:
        print(f"    ERROR {e}")
    if rep.errors:
        print("    FAIL — the toy is known good, so the contract spec is wrong, not the data")
        return 1
    print("    OK")

    checks = [
        ("truncated cofire", lambda s: s.update(cofire=s["cofire"][:-1])),
        ("negative count", lambda s: s["fire_p"].neg_()),
        ("dropped key", lambda s: s.pop("union_count")),
        ("bucket sum broken", lambda s: s["cofire_by_bucket"].mul_(2)),
    ]
    for name, break_it in checks:
        broken, _ = build_world(seed=0)
        break_it(broken)
        rep = validate(broken)
        status = "OK (rejected)" if rep.errors else "FAIL (accepted broken input)"
        print(f"[2] {name:20s} -> {status}")
        if not rep.errors:
            return 1

    print("\nself-test passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stats", type=Path, nargs="*", help="one or more .pt files")
    ap.add_argument("--self-test", action="store_true", help="validate the synthetic toy instead")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.stats:
        ap.error("give at least one .pt file, or --self-test")

    failed = 0
    for path in args.stats:
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  FAIL — file does not exist (a broken symlink counts)")
            failed += 1
            continue

        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            print(f"  FAIL — top level is {type(obj).__name__}, expected dict")
            failed += 1
            continue

        rep = validate(obj)
        print(f"  shape: {rep.mode}")
        for w in rep.warnings:
            print(f"  warn  {w}")
        for e in rep.errors:
            print(f"  ERROR {e}")

        if rep.errors:
            print(f"  FAIL — {len(rep.errors)} violation(s)")
            failed += 1
        else:
            print("  OK — conforms to the stats contract")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

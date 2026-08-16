#!/usr/bin/env python3
"""One unchanged battery, run on every source: the cross-run summary table.

One row per graded run -- every gemma layer, both zipf layers and the twelve
formatting-sweep runs -- with the same columns the Exp-0 slide carries:

    Run | Dictionary | Blocks | Tokens | Candidate edges | Reconstruction | S_res

All numbers are for the B0->B1 block pair, read from what the pipeline already
wrote (`metrics_report.json`, and `second_pass.json` where stage 03 has run).
Nothing is recomputed here: a row this script cannot fill from disk prints an
em-dash, which is the honest picture of which runs carry a second pass.

    python3 pipeline/battery_table.py                 # markdown to stdout
    python3 pipeline/battery_table.py --out table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

UMBRELLA = Path(__file__).resolve().parent.parent
OUTPUTS = UMBRELLA / "metrics" / "outputs"

# Density is not recoverable from the directory name's 2-decimal label alone,
# and the manifest lives outside this repo's outputs -- so the four sweep values
# are stated once, here, matching config.py's run list.
DENSITY = {"0000": "0.0000", "1667": "0.1667", "2308": "0.2308", "2400": "0.2400"}


def row(source_label: str, run_dir: Path, run_label: str) -> dict | None:
    report = run_dir / "metrics_report.json"
    if not report.is_file():
        return None
    r = json.loads(report.read_text())
    pair0 = next((p for p in r["pairs"] if str(p.get("pair")).replace(" ", "") in
                  ("0->1", "(0,1)", "[0,1]")), r["pairs"][0])

    steps = (r.get("config") or {}).get("matryoshka_steps")
    sres = "—"
    sp = run_dir / "second_pass.json"
    if sp.is_file():
        s = json.loads(sp.read_text()).get("0->1", {}).get("sres", {})
        if s.get("n_edges_scored"):
            sres = f"{s['n_pass']:,} / {s['n_edges_scored']:,}"

    return {
        "run": f"{source_label} · {run_label}",
        "dict": f"{steps[-1]:,}" if steps else "—",
        "blocks": str(len(steps)) if steps else "—",
        "tokens": f"{int(r['total_tokens']):,}" if r.get("total_tokens") else "—",
        "cand": f"{pair0['n_candidate_edges']:,}",
        "recon": f"{pair0['reconstruction']['frac_pass']:.0%}",
        "sres": sres,
    }


def rows() -> list[dict]:
    out = []
    gemma = OUTPUTS / "gemma-2-2b"
    for d in sorted(gemma.glob("layer_*")):
        out.append(row("gemma-2-2b", d, f"layer {int(d.name.split('_')[1])}"))
    pcfg = OUTPUTS / "pcfg-matryoshka"
    for d in sorted(pcfg.glob("layer_*")):
        out.append(row("PCFG zipf 1.5", d, f"layer {d.name.split('_')[1]}"))
    for d in sorted(pcfg.glob("fmt_*")):
        code, seed = d.name.split("_")[1], d.name.split("_")[2]
        out.append(row("PCFG fmt", d, f"{DENSITY.get(code, code)} {seed}"))
    return [r for r in out if r]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="also write the markdown here")
    args = ap.parse_args()

    L = ["| Run | Dictionary | Blocks | Tokens | Candidate edges | Reconstruction | S_res |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows():
        L.append(f"| **{r['run']}** | {r['dict']} | {r['blocks']} | {r['tokens']} "
                 f"| {r['cand']} | {r['recon']} | {r['sres']} |")
    L += ["", "Block pair B0→B1 in every row. An S_res of — means stage 03 "
              "(`run_token_metrics.py`) has not run for that layer: its token "
              "cache is not on this machine.", ""]
    text = "\n".join(L)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"[battery] wrote {args.out}")


if __name__ == "__main__":
    main()

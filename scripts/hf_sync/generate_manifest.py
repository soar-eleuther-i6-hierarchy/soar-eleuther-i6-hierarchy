"""Scan the repo for SAE/model checkpoints and activation caches, and write
an upload manifest (local_path, repo_id, repo_type, kind) for
upload_to_hub.sh / download_from_hub.sh. `kind` is "file" for a lone .pt
checkpoint or "dir" for a checkpoint/token_cache folder -- it tells
download_from_hub.sh whether to restore into local_path itself (dir) or
into its parent directory (file), since a single-file repo holds just the
basename while a folder repo holds the folder's contents.

Usage:
    python3 scripts/hf_sync/generate_manifest.py

Then review scripts/hf_sync/upload_manifest.tsv before running upload_to_hub.sh.

Classification rules:
  - a directory containing sae_weights.safetensors -> one model repo (whole dir)
  - a standalone *.pt file (not exp0_stats.pt, not inside a token_cache/ dir)
    -> one model repo (single file)
  - a directory named token_cache -> one dataset repo (whole dir)

Every matched path gets its own repo (no deduping of identical bytes across
paths) -- that mirrors how the checkpoints already living under this org were
uploaded on 2026-08-23.
"""
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ORG = "soar-eleuther-i6-hierarchy"
PRUNE_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache"}
SKIP_PT_FILENAMES = {"exp0_stats.pt"}

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "upload_manifest.tsv")


def slug(path_no_ext):
    s = path_no_ext.lower()
    s = re.sub(r"[/_]", "-", s)
    s = re.sub(r"[^a-z0-9.-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def main():
    rows = []  # (local_path relative to REPO_ROOT, repo_id, repo_type, kind[file|dir])

    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE_DIRS)
        rel_dir = os.path.relpath(dirpath, REPO_ROOT)

        if "sae_weights.safetensors" in filenames:
            name = slug(rel_dir)
            rows.append((rel_dir, f"{ORG}/{name}", "model", "dir"))

        if os.path.basename(dirpath) == "token_cache" and any(
            f.endswith(".pt") for f in filenames
        ):
            name = slug(rel_dir)
            rows.append((rel_dir, f"{ORG}/{name}", "dataset", "dir"))
            continue  # don't also treat the shards inside as standalone models

        if os.path.basename(dirpath) == "token_cache":
            continue

        for f in sorted(filenames):
            if f.endswith(".pt") and f not in SKIP_PT_FILENAMES:
                rel_file = os.path.relpath(os.path.join(dirpath, f), REPO_ROOT)
                name = slug(os.path.splitext(rel_file)[0]) + "-pt"
                rows.append((rel_file, f"{ORG}/{name}", "model", "file"))

    seen = {}
    for local, repo, rtype, kind in rows:
        seen.setdefault(repo, []).append(local)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        print("!!! DUPLICATE REPO NAMES -- fix slug() or rename paths:")
        for k, v in dupes.items():
            print(" ", k, v)
        raise SystemExit(1)

    with open(MANIFEST_PATH, "w") as fh:
        for local, repo, rtype, kind in rows:
            fh.write(f"{local}\t{repo}\t{rtype}\t{kind}\n")

    n_model = sum(1 for r in rows if r[2] == "model")
    n_dataset = sum(1 for r in rows if r[2] == "dataset")
    print(f"wrote {len(rows)} rows to {MANIFEST_PATH}")
    print(f"models={n_model} datasets={n_dataset}")
    print("Review the manifest, then run: scripts/hf_sync/upload_to_hub.sh")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Refill data/ from the compute node: the thirteen graded PCFG runs.
#
# data/ is gitignored and holds read-only copies, so a fresh clone -- or a cleaned
# working directory -- has none of it, and nothing else does either: the Hub
# dataset carries the five gemma layer caches only. The runs themselves live under
# another user's account on the node, read-only, and this pulls the four files per
# run that adapters/from_pcfg.py actually opens:
#
#     manifest.json   grammar block -> document length, and the sweep's knob values
#     model.pt        the base transformer, 43 MB
#     corpus.bin      uint16 token stream -- a PREFIX, see below
#     sae/matryoshka_hook_resid_post_L<layer>/   cfg.json + sae_weights.safetensors, 6 MB
#
# Everything else in a run directory (checkpoints, wandb, logs) is left on the node.
#
# The corpus is the reason this is not a plain rsync. A full one is ~200M tokens
# (382-400 MB, and on some runs a symlink into a sibling directory), while grading
# reads windows from the START of the stream and stops: the runs in the log used
# 1.02M tokens. So we copy a prefix -- 32 MB is ~16M tokens, ~40k documents, an
# order of magnitude more than any grading run has asked for, at a twelfth of the
# transfer. --full-corpus if you ever need the tail.
#
#     pipeline/fetch_pcfg_runs.sh user@node            # all thirteen
#     pipeline/fetch_pcfg_runs.sh user@node -n         # dry run, shows sizes
#     pipeline/fetch_pcfg_runs.sh user@node zipf       # just the zipf 1.5 run
#     PCFG_CORPUS_MB=64 pipeline/fetch_pcfg_runs.sh user@node zipf
#     PCFG_SRC=/mnt/ssd-1/someone/pcfg-experiments pipeline/fetch_pcfg_runs.sh user@node
#
# Layer numbers are not uniform and are not guessable: the formatting sweep was
# trained and graded at layer 2, the zipf run at layer 1. Fetching the wrong one
# gives a run directory the adapter refuses, which is the good outcome -- it lists
# what it found instead of grading something else.

set -euo pipefail

HOST=${1:?usage: fetch_pcfg_runs.sh user@node [-n] [zipf|fmt|all]}
shift || true

DRY=""
WHICH="all"
FULL=0
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY="--dry-run" ;;
        --full-corpus) FULL=1 ;;
        zipf|fmt|all) WHICH="$arg" ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done
CORPUS_MB=${PCFG_CORPUS_MB:-32}

SRC=${PCFG_SRC:-/mnt/ssd-1/april/pcfg-experiments}
DEST=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data

# experiment/run-on-node : local-dir : layer
FMT_RUNS=()
for h in c325cc965ffa 3915659d6f6c f98ccd6c7355 f6edabf8ccde; do
    for s in "" -s1 -s2; do
        FMT_RUNS+=("formatting_sweep/${h}${s}:fmt/${h}${s}:2")
    done
done
ZIPF_RUNS=("zipf_sweep/13df3dd54c16-s1:pcfg-run:1")

case "$WHICH" in
    zipf) RUNS=("${ZIPF_RUNS[@]}") ;;
    fmt)  RUNS=("${FMT_RUNS[@]}") ;;
    all)  RUNS=("${ZIPF_RUNS[@]}" "${FMT_RUNS[@]}") ;;
esac

echo "from : $HOST:$SRC"
echo "to   : $DEST"
echo "runs : ${#RUNS[@]}${DRY:+  (dry run)}"
echo

for entry in "${RUNS[@]}"; do
    IFS=: read -r remote local layer <<<"$entry"
    sae="sae/matryoshka_hook_resid_post_L${layer}"
    echo "--- $remote  ->  data/$local  (layer $layer)"
    if [ -n "$DRY" ]; then
        ssh "$HOST" "du -Lhc '$SRC/$remote/manifest.json' '$SRC/$remote/model.pt' \
                        '$SRC/$remote/$sae' 2>/dev/null | tail -1"
        echo "    corpus.bin: would take the first ${CORPUS_MB} MB"
        continue
    fi

    mkdir -p "$DEST/$local/sae"
    # cat and tar over ssh, not rsync: the node has no rsync. `tar -h` and `cat`
    # both resolve symlinks, which matters -- some runs' corpus.bin points into a
    # sibling run's directory, and a copied link is a dangling path locally.
    for f in manifest.json model.pt; do
        echo "    $f"
        ssh "$HOST" "cat '$SRC/$remote/$f'" > "$DEST/$local/$f"
    done
    echo "    $sae"
    ssh "$HOST" "tar chf - -C '$SRC/$remote/sae' '$(basename "$sae")'" \
        | tar xf - -C "$DEST/$local/sae"

    if [ "$FULL" = 1 ]; then
        echo "    corpus.bin: whole file"
        ssh "$HOST" "cat '$SRC/$remote/corpus.bin'" > "$DEST/$local/corpus.bin"
    else
        # The point of the prefix is to never move the other 350 MB, so the cut
        # happens on the node, before the bytes go over the wire.
        echo "    corpus.bin: first ${CORPUS_MB} MB"
        ssh "$HOST" "head -c $((CORPUS_MB * 1024 * 1024)) '$SRC/$remote/corpus.bin'" \
            > "$DEST/$local/corpus.bin"
    fi
done

echo
echo "next: grade a run, e.g."
echo "  python3 adapters/from_pcfg.py --run-dir data/pcfg-run --layer 1 \\"
echo "          --out metrics/outputs/pcfg/exp0_stats.pt"

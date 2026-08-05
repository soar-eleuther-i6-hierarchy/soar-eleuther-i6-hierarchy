# Docker

One image per sub-repo, over one shared artifact volume.

```bash
docker compose build
docker compose run --rm pcfg ./scripts/run.sh smoke
docker compose run --rm metrics python3 /workspace/contracts/validate_stats.py --self-test
```

On the compute node, add the GPU overlay:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm sae-training \
    uv run scripts/train_pcfg.py --run-dir "$PCFG_OUTPUT_ROOT/zipf_sweep/<hash>" --layer 2
```

## Why three images and not one

`sae-training` requires Python ≥3.12; `pcfg_bridge` requires ≥3.10 and keeps a deliberately lean
dependency set. The team already split `env.sh` from `env_sae.sh` for exactly this reason —
`sae_lens` and `transformer_lens` collide with the PCFG environment. The images inherit that split
rather than fighting it.

Each image builds from its repo's own `uv.lock` with `uv sync --frozen`, so an image can never drift
from the committed lock. `metrics` is the exception — see below.

## What these images are for, and what they are not

**They target the compute node** (linux/amd64, NVIDIA). Both locks pin torch to the CUDA 12.6 index,
whose wheels are linux/amd64 only.

**They are not a way to run this on a Mac.** On Apple Silicon the images run under x86 emulation,
where torch is slow at best and crashes on unsupported instructions at worst. For local work use the
native path the team already has:

```bash
source PCFG/env.sh
source sae-training/env_sae.sh
```

The one thing that *is* worth running locally is the contract validator — it needs only torch:

```bash
python3 contracts/validate_stats.py --self-test
```

## The metrics image is weaker than its siblings, on purpose visible

`metrics` has no `pyproject.toml` and no lockfile; its README installs dependencies with a bare
`pip install torch sae_lens datasets plotly numpy matplotlib`. Nothing records which versions
produced the published numbers.

The stopgap: [`requirements-metrics.in`](requirements-metrics.in) lists what is needed, and the
Dockerfile installs a *compiled* pin that must be generated and committed first:

```bash
uv pip compile docker/requirements-metrics.in -o docker/requirements-metrics.txt
```

Until that `.txt` is committed, `docker compose build metrics` fails — deliberately. An image built
from unpinned versions would look reproducible without being reproducible, which is worse than not
building.

The real fix is a `pyproject.toml` in the metrics repo, matching its two siblings. That is open
decision #2 in [`INTEGRATION_PLAN.md`](../INTEGRATION_PLAN.md).

## Volumes

| Volume | Mounted at | Holds |
| --- | --- | --- |
| `artifacts` | `/data/pcfg-experiments` | base models, corpora, SAEs — the node's `/mnt/ssd-1/$USER/pcfg-experiments` |
| `hf-cache` | `/hf-cache` | gemma-2-2b and the ~700 MB/layer stat caches |
| tmpfs | `/scratch` | in-run intermediates — the node's `/dev/shm/$USER/pcfg` |

The `metrics` service also gets `./sae-training` mounted read-only at `/workspace/sae-training`, with
`EXP0_SAE_TRAINING` pointing at it. Upstream, Tier-2 calibration finds `configs/tree.json` by probing
`../sae-training` and `./sae-training`; here the path is declared instead of guessed.

## Secrets

Nothing is baked into an image. `HF_TOKEN`, `WANDB_API_KEY` and friends are read from the
environment at run time:

```bash
HF_TOKEN=... docker compose run --rm metrics python3 collect_statistics.py --docs 16
```

`WANDB_MODE` defaults to `offline`, matching the guidance for a node that cannot always reach
api.wandb.ai.

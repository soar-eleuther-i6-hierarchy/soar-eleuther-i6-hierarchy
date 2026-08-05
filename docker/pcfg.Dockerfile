# syntax=docker/dockerfile:1
#
# Exp 2 — PCFG corpus generation + base-transformer training.
# Build context is the umbrella root, so the submodule is at PCFG/.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv builds into /opt/venv rather than the repo, mirroring how env.sh redirects
# UV_PROJECT_ENVIRONMENT to $HOME so the venv never lands on the shared SSD.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/pcfg

# Lock first: this layer stays cached until the dependencies actually change.
# --frozen resolves from uv.lock only, so the image can never drift from the lock.
COPY PCFG/pyproject.toml PCFG/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY PCFG/ ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Same storage conventions as env.sh: finished runs on the persistent volume,
# in-run intermediates on tmpfs.
ENV PCFG_OUTPUT_ROOT=/data/pcfg-experiments \
    PCFG_SCRATCH=/scratch \
    WANDB_DIR=/scratch

CMD ["bash"]

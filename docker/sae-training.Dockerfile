# syntax=docker/dockerfile:1
#
# SAE training — Matryoshka on toy models, TinyStories and PCFG base models.
# A separate image from pcfg by necessity: this repo pins Python >=3.12 and pulls
# sae-lens / transformer-lens, which is exactly why the team keeps env_sae.sh
# separate from env.sh rather than sharing one environment.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/sae-training

COPY sae-training/pyproject.toml sae-training/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY sae-training/ ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Reuses the PCFG pipeline's storage vars on purpose: SAEs are written beside their
# base model at <run_dir>/sae/..., so both pipelines share one output root.
ENV PCFG_OUTPUT_ROOT=/data/pcfg-experiments \
    PCFG_SCRATCH=/scratch \
    WANDB_DIR=/scratch \
    HF_HOME=/hf-cache

CMD ["bash"]

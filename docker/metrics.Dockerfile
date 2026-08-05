# syntax=docker/dockerfile:1
#
# Exp 0 — the metric battery.
#
# Unlike its two siblings this repo has no pyproject.toml and no lockfile; its
# README installs dependencies ad hoc. Until that is fixed upstream the pin lives
# here, in requirements-metrics.txt. See docker/README.md — this is the project's
# largest reproducibility hole, not a stylistic difference.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache

WORKDIR /workspace/metrics

COPY docker/requirements-metrics.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --requirement /tmp/requirements.txt

COPY metrics/ ./

# Tier-2 calibration reads sae-training/configs/tree.json. Upstream it probes
# ../sae-training and ./sae-training; here the path is declared instead of guessed,
# and compose mounts the sibling repo read-only at that location.
ENV EXP0_SAE_TRAINING=/workspace/sae-training \
    PCFG_OUTPUT_ROOT=/data/pcfg-experiments

CMD ["bash"]

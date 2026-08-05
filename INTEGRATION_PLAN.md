# SOAR I-6 — Integration Plan

How the three project repositories become one reproducible artifact for the paper.

**Status:** proposal, not yet executed. Written 2026-08-05 (project Week 4; write-up starts Week 5,
10–16 Aug).

---

## 1. What "integration" has to mean here

The three repos are not three parts of one program. They are three *producers and consumers of the
same measurement*. The paper's structure makes this explicit: **Experiment 0 builds one metric
battery, and Experiments 1–4 apply that same battery to SAEs from different sources.**

| Paper element | SAE source | Produced by |
| --- | --- | --- |
| Tier 1 calibration | synthetic ground-truth toy | `metrics/validation/toy_world.py` |
| Tier 2 calibration / Exp 1 | Matryoshka trained on Bussmann's tree | `sae-training` |
| Exp 2 — distributional bridge | Matryoshka on PCFG base transformers | `pcfg` → `sae-training` |
| Exp 3 — cross-method comparison | Matryoshka / T-SAE / Priors-in-Time on TinyStories | `sae-training` |
| Exp 4 — production scale | released gemma-2-2b checkpoints | none (checkpoints only) |
| Exp 0 results R1–R4 | gemma-2-2b, layers 3–24 | `metrics` |

So the integration target is **one metric implementation running unmodified across all six rows**.
Everything else — directory layout, Docker, CI — is in service of that.

## 2. The contract that already exists

`metrics/__init__.py` states the design property the whole integration rests on:

> Every metric is a pure function over cached statistics (co-firing counts, per-edge reconstruction
> sums, …) so the same code runs on the synthetic ground-truth toy in `validation/` (calibration)
> and the real gemma-2-2b Matryoshka caches.

`OUTLINE.md` §4 restates it as the bridge to Exp 1:

> the calibration harness (`validation/`) transfers unchanged because every metric is a pure
> function over cached stats.

This means the integration boundary is **not** a shared codebase and **not** a shared filesystem. It
is a single data structure: the cached-statistics object that `collect_statistics.py` writes to
`outputs/layer_NN/exp0_stats.pt`. It is already versioned (`schema_version: 2`) and `run_metrics.py`
already handles a missing optional key (`if "union_count" in stats`), i.e. the contract is
half-formalized but undocumented and unvalidated.

### 2.1 The stats contract (schema v2), as currently written

| Key | Shape | Meaning |
| --- | --- | --- |
| `schema_version` | `int` | 2 = BOS excluded, energy/union extras present |
| `fire_count` | `Tensor[D]` | per-feature fire count over the corpus slice |
| `total_tokens` | `int` | tokens after BOS exclusion |
| `token_counts`, `buckets` | — | token-frequency bucketing for the frequency control (metric 5) |
| `pairs` | `list[(p_blk, c_blk)]` | which block pairs were accumulated |
| `cofire`, `cofire_by_bucket` | per-pair `[P, C]` | co-firing counts, plain and per frequency bucket |
| `energy_cofire`, `energy_total`, `union_count`, `union_energy` | per-pair | joint-child coverage (`R_supp` / `R_mass` / energy share) |
| `g_parent_sum` | per-pair | reconstruction gain, parent side (metric 2a) |
| `err_sum_c`, `g_child_sum`, `fire_c_by_bucket`, `within_cofire` | per-child-block | reconstruction error, child gain, bucketed child fires, sibling redundancy input |
| `config` | dict | `layer`, `sae_release`, `sae_source`, `sae_id`, `matryoshka_steps`, `block_ranges`, `fire_threshold`, `n_docs`, `context_size`, `sibling_blocks`, `freq_high_mass`, `freq_mid_mass`, `bos_excluded`, `min_joint` |

`config.block_ranges` is what makes the object source-agnostic: it declares the Matryoshka block
boundaries, so the same metric code does not care whether the dictionary came from gemma-2-2b, a
PCFG transformer, or a toy.

### 2.2 The two secondary contracts

Both are currently implicit, described in prose in two different READMEs, and enforced by nothing:

- **Artifact layout.** `pcfg` writes a run to `$PCFG_OUTPUT_ROOT/<experiment>/<grammar_hash>/`
  containing `model.pt` + `corpus.bin`; `sae-training` writes the SAE *beside it* at
  `<run_dir>/sae/matryoshka_hook_resid_post_L{layer}/`. A rename on either side breaks the other
  silently.
- **Toy hierarchy.** `metrics/validation/calibrate_on_trained_toy.py` reads
  `sae-training/configs/tree.json`, locating it by probing `../sae-training` and `./sae-training`,
  overridable with `EXP0_SAE_TRAINING`. This sideways path probe is what the umbrella replaces with
  an explicit mount.

## 3. Why not a monorepo

Merging the three into one repository is the obvious move and the wrong one, for four concrete
reasons found in the code:

1. **Incompatible interpreters.** `pcfg` requires Python ≥3.10, `sae-training` requires ≥3.12.
2. **Deliberately separated environments.** The team already split `env.sh` from `env_sae.sh`
   precisely because `sae_lens` + `transformer_lens` collide with the lean `pcfg_bridge` env. The
   comment in `env_sae.sh` says so explicitly.
3. **Independent publication.** `metrics` publishes a GitHub Pages site at
   `soar-eleuther-i6-hierarchy.github.io/metrics/`. Folding it into a subdirectory breaks that URL —
   and that URL is a paper artifact.
4. **Distributed ownership.** Five contributors own different repos; a merge rewrites history and
   forces everyone onto one review queue mid-write-up.

Submodules give the property the paper actually needs: **the umbrella commit pins three exact
SHAs**, so a single citation reproduces the whole pipeline. That is a stronger reproducibility
claim than a monorepo, not a weaker one.

## 4. Target architecture

```
soar-eleuther-i6-hierarchy/
├── README.md                  # pipeline story + figure → command table
├── INTEGRATION_PLAN.md        # this file
├── .gitmodules                # pins metrics / sae-training / pcfg at exact SHAs
│
├── metrics/                   # submodule → soar-eleuther-i6-hierarchy/metrics.git
├── sae-training/              # submodule → .../sae-training.git
├── pcfg/                      # submodule → .../PCFG.git
│
├── contracts/
│   ├── stats_schema.md        # §2.1 as the normative spec
│   ├── validate_stats.py      # assert a .pt conforms before metrics touch it
│   ├── run_dir_layout.md      # §2.2 artifact layout
│   └── validate_run_dir.py    # assert a PCFG run dir conforms
│
├── adapters/                  # THE integration layer
│   ├── from_toy.py            # trained-toy SAE   → exp0_stats.pt
│   ├── from_pcfg.py           # PCFG run dir      → exp0_stats.pt
│   ├── from_tinystories.py    # TinyStories SAE   → exp0_stats.pt
│   └── README.md              # how to add a source
│
├── pipeline/
│   ├── run_e2e.sh             # corpus → base model → SAE → stats → metrics → figures
│   └── build_claims_table.py  # one table: R1–R4 across all sources
│
├── docker/
│   ├── metrics.Dockerfile  sae-training.Dockerfile  pcfg.Dockerfile
│   └── requirements-metrics.txt
└── compose.yaml
```

The two directories that carry the real work are `contracts/` and `adapters/`. Neither can live in
any single sub-repo, because both are *about the relationship between* the repos.

## 5. Phases

### Phase 0 — Prerequisites (blocking, ~1 hour)

- [ ] Land `metrics` work onto `main`. It currently sits on `exp0/metrics-table-scope` with **no
      upstream configured**; a submodule cannot pin an unpushed local branch.
- [ ] Reconcile the two diverging copies of the project plan: `I-6-Hierarchy-in-SAEs/` has 147
      lines, `SOAR-I-6/` has 182. Pick one as authoritative.
- [ ] Fix the dead artifact URL. `OUTLINE.md` §3.2 cites
      `github.io/experiment_0/` → **HTTP 404** (repo was renamed). The live URL is
      `github.io/metrics/` → **HTTP 200**.
- [ ] Strike `OUTLINE.md` §6 item 5 — it claims the `sae-training` URL is not publicly reachable.
      Verified: all three remotes resolve unauthenticated. The blocker is stale.

### Phase 1 — Umbrella skeleton (~1 hour)

- [ ] `git init`; add the three submodules, tracking `main`:
      `metrics` → `metrics.git`, `sae-training` → `sae-training.git`, `pcfg` → `PCFG.git`.
- [ ] Write `README.md` around the figure → command table, so every paper figure names the command
      that regenerates it.
- [ ] Record the three pinned SHAs in the README as the reproduction anchor.

### Phase 2 — Formalize the stats contract (~half a day)

- [ ] Write `contracts/stats_schema.md` from §2.1 — required keys, shapes, dtypes, and which keys
      are optional at v2.
- [ ] Write `contracts/validate_stats.py`: load a `.pt`, assert every required key, check per-pair
      shapes against `config.block_ranges`, fail loudly on a version mismatch.
- [ ] Run it against the five existing gemma layer caches. It must pass on data already known good —
      if it does not, the schema doc is wrong, not the data.
- [ ] Same treatment for `contracts/validate_run_dir.py` against the PCFG layout.

This phase is the highest-value one and is independent of Docker. It converts "the metrics happen to
work on the toy too" into a checked property.

### Phase 3 — Adapters (~1–2 days, the real work)

- [ ] `adapters/from_toy.py` first: it has a known-good reference. Tier 2 already reports
      **precision 1.00 / recall 0.67 (6 of 9 edges, 0 false positives)**. The adapter is correct iff
      it reproduces those exact numbers through the generic path.
- [ ] `adapters/from_pcfg.py`: read `<run_dir>/sae/matryoshka_hook_resid_post_L{layer}/`, stream
      residual activations from `model.pt`, emit schema-v2 stats. This is what unlocks Exp 2.
- [ ] `adapters/from_tinystories.py`: unlocks Exp 3. Lowest priority — Exp 3 also needs T-SAE and
      Priors-in-Time training, which the project plan itself flags as non-trivial.

Each adapter's acceptance test is `validate_stats.py` plus a metrics run that reproduces the
source's already-published numbers.

### Phase 4 — Docker (~half a day)

Three services over **one shared volume mounted at `PCFG_OUTPUT_ROOT`**, which is what makes the
artifact-layout contract executable rather than documentary:

| Service | Base | Notes |
| --- | --- | --- |
| `pcfg` | Python 3.10 + uv, `uv sync --frozen` | reads `uv.lock`; cu126 torch is already pinned for linux only |
| `sae-training` | Python 3.12 + uv, `uv sync --frozen` | separate image by necessity, not by preference |
| `metrics` | Python 3.12 + pinned `requirements-metrics.txt` | this repo has **no** `pyproject.toml` — see risks |

Shared mounts: the artifact volume (`PCFG_OUTPUT_ROOT`), a Hugging Face cache volume (gemma-2-2b and
the ~700 MB/layer stat caches are large), and `sae-training` mounted into the `metrics` container
with `EXP0_SAE_TRAINING` set explicitly — replacing the `../sae-training` path probe.
`PCFG_SCRATCH` maps to tmpfs, mirroring the node's `/dev/shm` convention.

**Open decision:** CPU-default images with a compose `gpu` profile that installs the cu126 wheels
(runs on a Mac for smoke tests, on the node for real runs), versus GPU-only images that match the
node exactly. See §7.

### Phase 5 — Paper pipeline (~1 day)

- [ ] `pipeline/run_e2e.sh`: one command from PCFG corpus to metrics report.
- [ ] `pipeline/build_claims_table.py`: emit R1–R4 for every available source into one table, with
      provenance (submodule SHA + grammar hash) attached to each row. This table *is* the paper's
      results section, and it is the artifact that only the umbrella can produce.

## 6. What this does not fix

The integration work is orthogonal to the four unresolved paper blockers in `OUTLINE.md` §6, and
does not address them:

1. "Five metrics" is stale — there are now ten (1a/1b/1c/2a/2b/3/4/5/6/7).
2. R5 is written as an open limitation, but the AND-gate was already replaced by out-degree-only.
3. **R4's depth numbers predate that fix and must be regenerated.** If the L24 collapse does not
   survive the new gate, R4 changes — and R4 is the paper's headline.
4. `MEETING_NOTES.md` documents a regeneration step (`add_links_docx.py`) that no longer exists.

Item 3 is a higher priority than anything in this plan: it can change a published claim, whereas
integration only changes how the claim is reproduced.

## 7. Open decisions

| # | Decision | Recommendation |
| --- | --- | --- |
| 1 | Docker target: CPU + GPU profile, or GPU-only | CPU-default + `gpu` profile — a Mac smoke test catches contract breaks without node access |
| 2 | `metrics` dependency pinning: `requirements.txt` in the umbrella, or a `pyproject.toml` committed into the `metrics` repo | `pyproject.toml` in `metrics`, matching its two siblings — it is the only repo with no pinned deps, the single largest reproducibility hole |
| 3 | Who owns `adapters/` | The umbrella. Putting them in `metrics` re-couples it to `sae-training`, which is what Phase 3 undoes |
| 4 | Publish the umbrella to GitHub now or after Phase 3 | After Phase 1, so the submodule pins exist and are citable early |

## 8. Risks

- **`metrics` has no dependency pinning.** Its README installs `torch sae_lens datasets plotly numpy
  matplotlib` with no versions. Any of these can break the published numbers, and nothing records
  which versions produced them. Highest reproducibility risk in the project.
- **Adapter drift is silent.** If an adapter emits subtly wrong stats, the metrics still return
  numbers — just wrong ones. Mitigation: every adapter must reproduce an already-published number
  (Tier 2's 1.00 / 0.67) before it is trusted.
- **Timeline.** Phases 0–2 are ~1 day and unblock everything. Phase 3 is the expensive part and
  overlaps the Week 5 write-up. If time runs short, ship Phases 0–2 + 4 and document Exp 2/3
  integration as future work rather than half-building the adapters.
- **`PCFG/sae-training/` is an empty stale placeholder** for a sibling clone. The umbrella layout
  supersedes it; delete it to avoid a second, contradictory convention.

# Experiment log

Newest first. Template and conventions: [`README.md`](README.md).

---

## 2026-08-06 19:20 +03 — The depth-degradation result was a BOS contamination artifact

**Question.** The published depth numbers were produced on 18 July; the superparent gate
changed on 24 July in `a266f8c`. Do they survive regeneration under the current code?

**How it can be answered.** Re-run the pipeline from stage 01 on all five layers and compare.
An earlier attempt re-ran only stages 02 and 02b against the *same* cache and found nothing
changed — which proved only that stage 02's thresholds are not what moves, since the input
was identical. The cache itself has to be rebuilt.

Inspecting it showed why that matters: the shipped caches carry no `schema_version`,
`bos_excluded` is `None`, and the energy/union accumulators are absent. They are **v1** —
written before BOS exclusion.

**What we ran.** A fresh clone of `main` on the compute node, all five layers, GPU 3:

```bash
EXP0_LAYER=$L python3 run_pipeline.py --only 01 02 02b
```

L24 was additionally run twice, from `aed1e1a` and from `17326db`, to check the result does
not depend on which merge the code sits at. Identical both times.

**Result.** Every layer's token count drops 48,971 → 48,571, exactly 400 — the document
count, one BOS each.

Candidate counts collapse on **all five** layers, not only L24:

| layer | B0→B1 | B1→B2 | B2→B3 |
| --- | --- | --- | --- |
| L3 | 3,067 → 1,971 | 28,588 → 1,387 | 274,313 → **4,379** |
| L6 | 8,156 → 2,428 | 271,644 → **280** | 4,704,312 → **762** |
| L12 | 3,262 → 1,473 | 34,142 → 621 | 431,127 → 1,747 |
| L18 | 4,901 → 1,129 | 141,272 → 1,748 | 3,820,801 → **4,195** |
| L24 | 4,940 → 2,273 | 108,810 → **424** | 3,695,288 → **4,867** |

B2→B3 at L6 falls by a factor of 6,173.

The reconstruction pass rate inverts with it — B0→B1: L3 42.5→**89.8%**, L6 6.3→**85.9%**,
L12 16.3→53.9%, L18 7.3→74.8%, L24 17.8→**73.6%**. Deep pairs go from 0.0% to 9–41%.

Distinct parents among the 8 survivors, B0→B1:

| | L3 | L6 | L12 | L18 | **L24** |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 (published) | 5 | 7 | 6 | 7 | **2** |
| v2 (regenerated) | 5 | 7 | 6 | 7 | **6** |

Busiest surviving parent's share:

| | L3 | L6 | L12 | L18 | **L24** |
| --- | --- | --- | --- | --- | --- |
| v1 | 1/8 | 1/8 | 1/8 | 2/8 | **6/8** |
| v2 | 1/8 | 1/8 | 2/8 | 2/8 | **2/8** |

Its firing rate moves in both directions and shows no depth trend: L6 20.5→14.0%,
L12 15.5→11.0%, L18 18.2→**33.8%**, L24 41.9→27.0%.

**Every metric that reads co-firing moves, and in the same direction.** B0→B1:

| | frequency-driven edges | | mean survival | | sibling redundancy | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | v1 | v2 | v1 | v2 | v1 | v2 |
| L3 | 28.4% | **0.9%** | 0.758 | **1.040** | 0.229 | **0.059** |
| L6 | **60.8%** | **1.0%** | 0.441 | **1.031** | 0.351 | **0.054** |
| L12 | 43.8% | **1.9%** | 0.594 | 0.998 | 0.273 | **0.033** |
| L18 | **65.2%** | **1.3%** | 0.396 | **1.004** | 0.340 | **0.042** |
| L24 | 53.3% | **2.2%** | 0.517 | 0.994 | 0.350 | **0.060** |

Both follow from the same cause. BOS is a *single token id*, so it lands in the
high-frequency bucket — which made every edge look carried by frequent tokens, and made
every pair of children look co-active. Removing it takes frequency-driven edges from a
majority to ~1% and cuts sibling redundancy six-fold.

Joint-child coverage could not be compared: `energy_cofire` / `union_*` do not exist in v1,
so metrics 1c, 8 and 9 were **never computed on gemma**. Now they are, and they are
saturated — `R_supp` and `R_mass` both ≈ 1.000 at every layer, i.e. a parent's children
jointly cover it entirely in both support and mass. Only L18 shows anything: 9 parents where
one child holds ≥90% of the parent's energy, the rename/split signature.

`n_chance_level` is computed in `run_metrics.py` but is absent from the emitted report on
both sides, so the independence null could not be compared here at all.

Multi-parenting is the one figure that does **not** move: 99.0 / 100 / 99.7 / 88.8 / 100%
against 99.7 / 100 / 99.7 / 94.3 / 100% before. It is a ratio over children that have a
parent at all, so deleting phantom pairs leaves it alone.

**Interpretation.** BOS is an attention sink, so every feature fires there. With 400 BOS
positions in the corpus, every pair in the dictionary received 400 joint firings for free —
and `MIN_JOINT` is 30. The joint-support guard was therefore satisfied by BOS alone for
**every pair**, which is what produced 3.7M "candidates" at B2→B3 and what kept feature 14's
edges alive at L24.

The candidate sets were mostly phantom. Which means the headline framing — coverage proposes
far more than survives — was measuring the phantoms: the 94–99.9% that "died" were pairs that
should never have been candidates. Corrected, the majority of B0→B1 candidates now *pass*
reconstruction, and the deep pairs carry a real if small signal instead of exactly zero.

On depth: four layers' distinct-parent counts do not move, and L24 moves all the way to the
others' value. The pattern was one outlier and the outlier was contamination. Corrected, the
counts read 5 · 7 · 6 · 7 · 6 — flat.

Why only L24's survivor structure moved, when every layer's candidate set collapsed, is
unexplained. The plausible story is that earlier layers' features fire often enough to clear
`MIN_JOINT` on their own, while at L24 they are sparse enough that BOS was carrying feature
14's edges by itself. That is a hypothesis, not a measurement.

**Answer.** No. Three of the four published claims do not survive:

| claim | after regeneration |
| --- | --- |
| coverage over-proposes; 94–99.9% of edges die | **inverted** — 74–90% of B0→B1 candidates now pass |
| deep block pairs carry no signal | **inverted** — B2→B3 passes 9–41%, not 0.0% |
| quality degrades with depth | **gone** — no trend |
| it is not a tree | **holds** — 89–100% of children have ≥2 parents |

Every metric that reads the co-firing matrix moved: coverage, reconstruction, frequency
control (60.8% → 1.0% at L6), sibling redundancy (six-fold down), superparents. The one that
holds is the one that does not depend on the candidate set — multi-parenting is a ratio over
children that already have a parent.

That is the shape of the whole result. A single contaminating token inflated one matrix, and
five of six metrics read that matrix. This is also, in retrospect, what the metrics-battery
design was supposed to protect against: independent detectors that fail independently. They
did not fail independently, because they share an input.

**Caveats.** The regenerated caches use the same corpus slice, thresholds and SAE as the
originals, so BOS exclusion is the only intended difference; verified by comparing the
recorded `config` blocks. Stage 03 and the in-block metric have still not been run on any
layer. The qualitative reading of survivor labels has not been redone, so whether the new
survivors are semantically coherent is unknown — the numbers moved, the human check has not
been repeated.

**A note on how this was nearly missed.** Three separate times today a committed artifact was
read as though it were fresh output: re-running stages 02/02b against an unchanged cache and
declaring the question closed; L12 and L18 showing reports that came from the clone; and L3
compared against itself, which produced a spurious "L3 was never contaminated" reading that
survived until the token count was checked. `run_pipeline.py` exists to refuse exactly this
and was used for only the last of the five layers. The lesson is not "be careful" — it is
that output directories under version control need the guard, because a stale file there is
indistinguishable from a fresh one by inspection.

---

## 2026-08-06 17:05 +03 — L24's feature 14 is a base-rate artifact, but not a frequency one

**Question.** The gate entry below closes with the question the gate was the wrong instrument
for: is the 41.9% firing rate of L24's feature 14 a finding — a genuine high-firing parent —
or an artifact? It dominates 6 of the 8 survivors that R4 counts, so the answer changes how
the depth claim is worded.

**How it can be answered.** Read the two per-edge diagnostics on feature 14's own edges
rather than any node-level flag. Metric 6 (`independence_scores`) asks whether the co-firing
exceeds what the parent's base rate already forces; metric 5 (`frequency_controlled_coverage`)
asks whether the edge survives once globally frequent tokens are removed. The prediction
stated below was that they would agree: near-chance PMI *with* survival ≈ 0 for capture,
healthy PMI for a real parent.

**What we ran.** The layer-24 cache, `outputs/layer_24/exp0_stats.pt`, 696 MB. Edge set
rebuilt at the shipped thresholds (`EDGE_TAU` 0.5, `MIN_FIRE_COUNT` 20, `MIN_JOINT` 30) for
block pair B0→B1, then PMI and survival read off feature 14's kept edges.

```bash
hf download soar-eleuther-i6-hierarchy/experiment_0-stats --repo-type dataset --local-dir outputs/
```

(The run behind these numbers pulled the same file off the compute node by `scp` instead,
from `/mnt/ssd-2/soar-hierarchy/ruqiya/experiment_0/outputs/layer_24/`. The Hub command is
the reproducible route — it does not depend on node access or on one person's directory
layout, and it is what the metrics README documents.)

**Result.** 48,971 tokens. Feature 14 fires on **41.9%** of them, fans out to **21.9%**
(84 of 384), and is unflagged by the shipped gate.

Over its 84 kept edges:

| | min | median | max |
| --- | ---: | ---: | ---: |
| coverage `R` | 0.501 | 0.661 | 0.952 |
| PMI | **0.179** | **0.456** | 0.821 |
| frequency survival | **0.723** | **0.985** | 1.429 |

| | |
| --- | --- |
| PMI < 0.5 — at chance level | **49 / 84 (58%)** |
| survival < 0.5 — frequency-driven | **0 / 84** |

**Interpretation.** The prediction was wrong: the two diagnostics disagree, and neither of
the two anticipated boxes is the right one.

They disagree because they measure different frequencies. Survival asks whether an edge is
carried by *globally frequent tokens* — it is not, and cleanly so; every edge holds up on the
rare-token tail. PMI asks whether the co-firing exceeds what *the parent's own firing rate*
already forces, and for 58% of the edges it does not. A parent firing 41.9% needs only 1.19×
enrichment to clear `EDGE_TAU = 0.5`, where a parent firing 1% needs 50×, and the median edge
here sits at 1.58× — above chance, but not by much.

So feature 14 is a base-rate artifact without being a token-frequency artifact. That is a
third category the question did not allow for, and it is precisely what the project means by
"superparent": a feature active often enough to co-fire with everything by arithmetic.

Two consequences for R4. The **claim survives** — a parent whose edges are majority
chance-level, dominating 6 of 8 survivors, is not evidence of hierarchy, so the L24 collapse
stands. The **explanation does not**: describing it as frequency capture is contradicted by
metric 5, which exonerates every one of the 84 edges. The depth result should attribute it to
the parent's firing rate, not to frequent tokens.

It also settles the gate argument empirically rather than by reasoning. PMI already flags
these edges, at edge granularity, which is what the entry below argued a node-level OR would
do worse. That was an inference from thresholds; this is a measurement.

**Answer.** Neither box: base-rate artifact, not frequency capture. R4's L24 collapse holds
and its wording needs correcting. No gate change is warranted — the leak the OR was meant to
close is already caught per edge by metric 6.

**Caveats.** One feature, one block pair, one layer, 48,971 tokens. The 0.5 PMI cutoff is
itself a fixed threshold and carries the same transfer problem logged for the reconstruction
threshold. Whether the other four layers' busiest survivors show the same split is unchecked;
that is the obvious next probe and needs their caches.

---

## 2026-08-06 15:44 +03 — The OR gate admits parents with no children, and would delete the depth result it is meant to sharpen

**Question.** [`metrics/outdegree.py`](../metrics/metrics/outdegree.py#L63) records why the
superparent gate changed: the old `AND(fan-out ≥ 30%, fires ≥ 10%)` let L24's feature 14 —
fires on 41.9% of tokens, fans out to 21.9% of the child block — go unflagged. But the
fan-out-only gate that shipped does not flag feature 14 either: it fails on **fan-out**, and
fan-out is the criterion that was *kept*. Dropping the firing conjunct catches the opposite
leak, high fan-out with low firing. Promoting it instead — an **OR** — is the other way to
act on that same note, and it is the one not yet tested. Is the OR the better gate?

**How it can be answered.** An OR is a claim about a classifier, so test it as one: run all
three gates over every graded run we hold and look at what the OR actually admits.
`outdegree.py` defines a superparent as "one parent holding most of the next block's
in-edges". A gate that admits parents holding *no* in-edges is not detecting that
pathology, whatever else it detects.

**What we ran.** All thirteen graded PCFG runs we hold — the twelve formatting-sweep runs
(4 densities × 3 seeds, layer 2) plus the zipf 1.5 run (layer 1). Edge sets rebuilt from the
cached statistics with `coverage_legs` + `keep_edges` at the shipped thresholds (`EDGE_TAU`
0.5, `MIN_FIRE_COUNT` 20, `MIN_JOINT` 30), then all three gates applied to the same masks,
so the gates are the only thing that differs. 1,022,000 tokens per run, 1792 latents in 8
blocks, 7 block pairs each — 91 (run × pair) cells.

```bash
python3 pipeline/superparent_gate_table.py --data data/fmt data/pcfg-run
```

Runs: `c325cc965ffa` (density 0.0), `3915659d6f6c` (0.1667), `f98ccd6c7355` (0.2308),
`f6edabf8ccde` (0.24), each with `-s1` / `-s2`; plus `data/pcfg-run` (zipf 1.5), which is
graded from `exp0_stats_full.pt` rather than the 200-window `exp0_stats.pt` beside it.

**Result.** Counts are parent-slots summed over the 7 block pairs.

| run | fan-out alone (ships) | AND (old) | OR | OR-only adds | of those, childless |
| --- | ---: | ---: | ---: | ---: | ---: |
| fmt 0.0000 s0/s1/s2 | 0 · 0 · 1 | 0 · 0 · 1 | 87 · 68 · 78 | 232 | 202 |
| fmt 0.1667 s0/s1/s2 | 0 · 0 · 2 | 0 · 0 · 2 | 118 · 98 · 103 | 317 | 287 |
| fmt 0.2308 s0/s1/s2 | 1 · 0 · 0 | 1 · 0 · 0 | 91 · 115 · 97 | 302 | 276 |
| fmt 0.2400 s0/s1/s2 | 2 · 0 · 1 | 2 · 0 · 1 | 66 · 102 · 134 | 299 | 258 |
| zipf 1.5 | 0 | 0 | 100 | 100 | 89 |
| **total** | **7** | **7** | **1257** | **1250** | **1112 (89.0%)** |

Fan-out of the 1250 additions: median **0.000%**, mean 0.269%. Only 138 (11.0%) have any
child at all; 53 (4.2%) exceed 1% fan-out; 12 (1.0%) exceed 10%; 2 exceed 20%. The single
closest call anywhere is 29.0%, just under the threshold.

Two things fall out that were not the question. The old AND gate and the shipped fan-out
gate give **identical counts on all thirteen runs** — on PCFG every parent clearing 30%
fan-out also fires ≥ 10%. And the shipped gate flags 7 parent-slots in 91 (run × pair)
cells: superparents are near-absent on PCFG at any setting.

**The same two gates on gemma.** PCFG is one source, and a gate is meant to hold across
all of them, so the five published gemma layers were checked too — from the committed
reports, which carry `outdeg_frac` and `fire_frac` per flagged parent:

| layer | flagged (listed) | fire ≥ 10% | fire < 10% | fan-out range | fire range |
| --- | ---: | ---: | ---: | --- | --- |
| L03 | 8 | 8 | **0** | 38.3–99.7% | 30.4–99.3% |
| L06 | 24 | 24 | **0** | 46.1–99.7% | 10.1–99.0% |
| L12 | 5 | 5 | **0** | 31.0–100% | 38.2–98.8% |
| L18 | 15 | 15 | **0** | 34.1–100% | 10.3–98.9% |
| L24 | 15 | 15 | **0** | 32.8–100% | 11.6–99.5% |
| **total** | **67** | **67** | **0** | | |

Not one flagged gemma superparent fires under 10%. So AND and fan-out-alone agree on gemma
as well, over 67 of the 72 flagged parents — reports list the top 10 per pair, and the 5
unlisted have the lowest out-degree of their pair, so this is not quite exhaustive.

The OR arm **cannot** be computed on gemma from committed artifacts. Its additions are by
definition parents with fire ≥ 10% and fan-out < 30%, and those appear in no report — the
reports only list parents the gate already flagged. Measuring it needs
`outputs/layer_NN/exp0_stats.pt`, which is not in the repo.

**Interpretation.** The OR inflates the count 180× on PCFG and 89% of what it adds has no
children at all. It is not a stricter superparent detector; it is a firing-rate threshold
wearing the superparent label, and it would make `n_superparents` mean "frequent feature" on
PCFG and "fans out widely" on gemma — one field, two meanings, which is the failure the
block-indices work was done to avoid.

The stronger objection is structural, and it holds on every source because it is in the
code rather than the data. Flagging is not inert:
[`validation/qualitative_check.py:169`](../metrics/validation/qualitative_check.py#L169)
builds the survivor set as `em & passes & survive & ~sp_locals`, and flagged parents are
re-filed under `reject:superparent`. Any figure quoted over survivors therefore moves when
the gate moves. The depth comparison is exactly such a figure: it reports the busiest
*surviving* parent's firing rate per layer, and at L24 that parent is feature 14 at 41.9% —
which is a survivor **because** it is unflagged. Implement the OR and it lands in the reject
bucket, the L24 figure is replaced by the next survivor, and the distinct-parent count
changes with it. The OR does not sharpen the evidence for collapse at depth; it removes it,
by definition rather than by measurement.

Nor is the firing leak unaddressed today, which is the natural way to describe it if the OR
is declined. At `EDGE_TAU = 0.5` a parent firing 41.9% needs only 1.19× enrichment over
chance to clear the bar, where a parent firing 1% needs 50×; that asymmetry is exactly what
metric 3 measures. An edge from feature 14 that just clears the bar has
`pmi = ln(0.5/0.419) = 0.18` nat, under the `< 0.5` cutoff `run_metrics.py` already counts as
`n_chance_level` and labels a "C-freq artifact". The leak is caught — per edge, which is the
right granularity, since frequency spoils particular edges and not a whole feature.

**Answer.** No. Keep fan-out alone; do not implement the OR. Report the firing criterion as
*deliberately excluded from the node-level gate and handled per edge by metrics 3 and 5*,
rather than as an unaddressed leak. Separately, the shipped fan-out-only change should not
be described as fixing the case it cites: it is a no-op against both gates' agreement on
every source we can measure — 13/13 PCFG runs and 67/67 flagged gemma parents — and feature
14 stays unflagged under it.

This does not settle whether the L24 41.9% is a finding or an artifact. That is a separate,
answerable question, and the gate is the wrong instrument for it. Read PMI and frequency
survival on feature 14's own edges: near-chance PMI with survival ≈ 0 means the figure was
frequency capture and the depth claim needs rewording; healthy PMI means feature 14 is a
genuine high-firing parent, the claim stands, and flagging it would have deleted a true
result. That check needs `metrics/outputs/layer_24/exp0_stats.pt`, which is not in the repo
— it is the `[02] WAIT` in `run_pipeline.py --list`, so it is a compute-node run.

Caveats, and the largest is the source. **The decisive column — 89% of OR's additions have
no children — is PCFG only.** It could not be computed on gemma, because the additions are
precisely the parents no report lists. What gemma does establish is narrower and from
committed data: the AND and fan-out-only gates agree there too, so the shipped change is
inert on both sources. The 180× inflation is measured where superparents are near-absent to
begin with (7 slots in 91 cells), so the gemma inflation factor is unknown and could differ
in either direction. The `~sp_locals` and PMI arguments are read from the code and hold on
any source; the counts are not. Closing the gap needs stage 01 re-run for one gemma layer —
gemma-2-2b, the released Matryoshka SAE and pile-10k, 400 docs × 128 ctx — and until it is
run, the recommendation rests on one source plus two source-independent arguments.

---

## 2026-08-06 12:40 +03 — Tier 2 reaches the same verdict through the production path

**Question.** Tier 2's precision 1.00 / recall 0.67 is the only published number with both
ground truth and a real training run behind it, but it is produced by a bespoke path:
`calibrate_on_trained_toy.py` calls the metric functions directly with thresholds written
into the script, and never touches `collect()`, `run_metrics.py` or the contract. Does the
production path reach the same verdict?

**How it can be answered.** Route the same checkpoint through the real pipeline and compare
the surviving edge set, not just the two summary figures — matching figures with different
edges would be coincidence. If they agree, that number covers the accumulation and the
report as well, and the pipeline gains its only end-to-end check against a known answer.

Getting there needed one change first: the toy's groups are not contiguous. It indexes by
which true feature each latent recovered, so parents and children are scattered lists
(`[0, 3, 8]` and `[5, 7, 9, 10, 11, 14, 17]`). `collect()` sliced blocks as ranges. The
alternative to generalising it was permuting the dictionary until the groups were
contiguous, which would have made `B0→B1` mean "true parents → true children" here and
"first 128 → next 384" on gemma — one field name with two meanings and nothing in the file
to say so.

**What we ran.**

Inputs, both committed and therefore reproducible from a clone:

| | |
| --- | --- |
| checkpoint | `metrics/outputs/toy_trained/{sae_weights.safetensors,cfg.json}` — trained 2026-07-24, `d_in=d_sae=20`, `batch_topk` `k=2`, saved threshold 4.5e-5 |
| ground-truth tree | `sae-training/configs/tree.json` — 20 read-out features, 9 true edges |

```bash
# production path — the .pt is a scratch artifact, regenerate rather than keep
python3 adapters/from_toy.py --out "$(mktemp -d)/toy_stats.pt"     # 200k draws, seed 0

# reference path, unchanged
cd metrics && python3 validation/calibrate_on_trained_toy.py
```

200,000 world draws collapse to 1,337 distinct states; 199,936 tokens after position 0 is
dropped; 17 of 20 features alive. Nothing is written to either repo — the stats file is
derived and both inputs are in git, so the run is reproducible without carrying 19 MB
around.

**Result.** Identical, edge for edge.

| | production path | reference path |
| --- | --- | --- |
| surviving edges | (0,1) (0,2) (4,5) (4,7) (8,9) (8,10) | same six |
| precision | **1.00** | 1.00 |
| recall | **0.67** | 0.67 |
| false positives | 0 | 0 |
| missed | (0,3) (4,6) (8,11) | same three |

Guards, checked separately: a missing path, a checkpoint from another world (`d_in=448`
against the toy's 20), and weights with keys removed are all refused with a specific
message. Seed 0 twice is byte-identical; seed 0 against seed 1 differs.

**Interpretation.** The two paths differ in more than plumbing, which is what makes the
agreement worth something. Thresholds come from `config.py` on one and from literals in the
script on the other. The token encoding differs outright: the reference uses
`fired.argmax(1)` as a stand-in token, the adapter enumerates distinct world states, so
metric 5's buckets are over different objects. Reaching the same six edges through both says
the result is not sensitive to either — which was not previously established.

What this does **not** show: that the pipeline is correct in general. It shows that on one
20-feature toy with a known tree, the accumulation and the report preserve a verdict the
metrics already produced. gemma and PCFG remain unchecked against any known answer, because
neither has one.

**Answer.** Yes. The production path reproduces Tier 2 exactly, so 1.00 / 0.67 now validates
`collect()` and `run_metrics.py` as well as the metric functions. It is the only number in
the project that can do that.

---

## 2026-08-06 10:15 +03 — The reconstruction threshold does not discriminate on PCFG

**Question.** Every PCFG run reports 100% of candidate edges passing the reconstruction
condition, against 0.9–42.5% on gemma. Is that a clean result, or a threshold calibrated
for one dictionary and inert on another?

**How it can be answered.** The pass mask is `parent_gain >= RECON_REL_GAIN_MIN` with the
threshold at 0.01. Measure the actual distribution of `parent_gain` over kept edges. If the
weakest edge sits far above 0.01, the filter is passing everything by construction and the
100% says nothing about the edges.

**What we ran.** Recomputed `edge_reconstruction_condition` over the edge set of three
graded PCFG runs and took the distribution of `parent_gain` on kept edges.

```bash
# stats produced earlier by adapters/from_pcfg.py, thresholds from metrics/config.py
data/fmt/f6edabf8ccde/exp0_stats.pt      # formatting 0.2400
data/fmt/c325cc965ffa/exp0_stats.pt      # formatting 0.0000
data/pcfg-run/exp0_stats_full.pt         # zipf 1.5
```

**Result.**

| run | n edges | min gain | median | below 0.01 |
| --- | ---: | ---: | ---: | ---: |
| formatting 0.2400 | 357 | 0.041 | 0.337 | **0** |
| formatting 0.0000 | 22 | 0.035 | 0.262 | **0** |
| zipf 1.5 | 21 | 0.039 | 0.468 | **0** |

The weakest edge anywhere in PCFG is 3.5× the threshold; the median is 26–47×. On gemma
layer 6 the same threshold removes 93.7% of B0→B1 candidates.

**Interpretation.** The filter is inert here, so "100% pass reconstruction" is a property of
the threshold and not of the edges. The likely mechanism is dictionary size against model
width: 1792 latents over `d_model` 448 versus gemma's 32768 over 2304, so a single PCFG
latent carries far more of the representation and ablating it necessarily moves the error a
lot. Nothing about the edges follows.

This matters beyond one number. It means the filtering on PCFG is being done by **coverage
alone** — metric 5 is idle on the zipf axis for a separate reason already logged, and metric
2 is idle everywhere on this source. The 356 edges at density 0.24 survived one filter, not
a battery, and should not be described as having passed the battery.

It also qualifies a claim we make deliberately. Holding thresholds fixed across sources is
the point — it is what makes "the same battery everywhere" true. But a fixed threshold on a
quantity whose distribution shifts by an order of magnitude is not measuring the same thing,
and the outline already records the same failure across *depth*: "fixed thresholds do not
transfer across depth; any single global threshold is wrong somewhere." They do not transfer
across *sources* either, and more sharply.

What we cannot do is pick a better number. Calibrating `RECON_REL_GAIN_MIN` for PCFG needs
ground truth about the SAE's features, and we only have ground truth about the grammar. This
is a limitation to declare, not a constant to tune.

**Answer.** Not a clean result. The reconstruction condition passes every PCFG edge by
construction and carries no information on this source. Any PCFG number quoting a
reconstruction pass rate should be reported as inert, and the surviving-edge counts should be
attributed to coverage alone.

---

## 2026-08-06 08:40 +03 — Formatting density degrades hierarchy quality; Zipfianness does not

**Question.** The bottleneck-hijacking hypothesis says scarce top-block capacity gets
spent on globally frequent tokens instead of concepts. Which PCFG sweep axis actually
varies that quantity — (a) `zipf_exponent`, or (b) formatting density?

**How it can be answered.** Our frequency-control metric buckets tokens by **global**
corpus counts, so the axis that tests the hypothesis must change the global token
distribution. Measure that distribution directly on each corpus, then run the metric
battery across each axis and see which one produces frequency-driven edges. If an axis
leaves the global distribution flat, the metric has nothing to detect there regardless
of what the SAE learned.

**What we ran.** Twelve runs of `formatting_sweep` (4 densities × 3 seeds), plus one
`zipf_sweep` run for contrast. All from `/mnt/ssd-1/april/pcfg-experiments/`, read-only.

```bash
# per run: pull model.pt + corpus prefix + sae/…L2, then grade
python3 adapters/from_pcfg.py --run-dir data/fmt/<hash> --layer 2 --docs 2000 \
        --out data/fmt/<hash>/exp0_stats.pt
cd metrics && python3 run_metrics.py --stats ../data/fmt/<hash>/exp0_stats.pt \
        --out-dir ../data/fmt/<hash>/report
python3 pipeline/formatting_sweep_table.py --data data/fmt
```

Runs: `c325cc965ffa` (0.0), `3915659d6f6c` (0.1667), `f98ccd6c7355` (0.2308),
`f6edabf8ccde` (0.24), each with `-s1` / `-s2`. Layer 2, 1,022,000 tokens each,
`zipf_exponent` held at 1.0 throughout. Contrast run: `zipf_sweep/13df3dd54c16-s1`,
`zipf_exponent` 1.5, layer 1.

**Result.**

Axis (b), formatting density, mean over 3 seeds:

| density | delimiters | top token | freq-driven | mean survival | superparents |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.0000 | none | 0.1% | **0.0%** | **0.994** | 0.3 |
| 0.1667 | EOS | 16.6% | **0.0%** | **0.991** | 0.7 |
| 0.2308 | EOS + para | 15.4% | **3.7%** | **0.963** | 0.3 |
| 0.2400 | all four | 15.2% | **10.6%** | **0.886** | 1.0 |

Per-seed at the densest point: 9.8% / 10.8% / 11.3% freq-driven — tight.
At 0.2308 it is noisy: 4.8% / 0.0% / 6.4%.

Axis (a) for contrast — `zipf=1.5`: 0.0% frequency-driven, survival 1.007, 0
superparents.

Token distribution of the `zipf=1.5` corpus, top 10 tokens:

| | share | uniform would be |
| --- | ---: | ---: |
| within one document | **59.2%** | 9.3% |
| across the corpus | **1.3%** | 1.0% |

**Interpretation.** Axis (a) concentrates tokens *within* a document but not across the
corpus: `weights = ranks ** (-s)` scores ranks, and `_permutations(doc_id)` re-randomises
which token id holds which rank per document, so the corpus-wide marginal averages back to
flat. The only globally frequent token at `zipf=1.5` is `EOS` at 16.6%, a formatting token.
So the earlier PCFG reading of "0% frequency-driven, survival 1.007 — the edges are clean"
was wrong: the confound that metric detects does not exist in that corpus, so the metric
had nothing to do.

Axis (b) does vary global frequency, by construction, and hierarchy quality degrades
monotonically with it on two independent ratios. Those two are the trustworthy signal —
candidate-edge counts are not, because they scale with how many features are alive
(571–784 of 1792 across runs).

What this does **not** establish: that within-document concentration is irrelevant. It may
matter through a different mechanism, and metric 5 cannot see it either way — it is a
global-frequency instrument. Also unsettled is why `EOS` alone (16.6% of tokens, density
0.1667) produces no frequency-driven edges at all; a plausible read is that a token
appearing after *every* sentence co-fires with everything equally and so creates no
differential signal, but we have not tested that.

Caveats: single layer (L2), 1.02M-token prefix of a 191M-token corpus, and the 0.2308
point is within noise of zero on one seed of three.

**Answer.** Axis (b), formatting density, is the axis that tests the hypothesis as we have
stated and instrumented it. Axis (a) as currently generated does not vary global token
frequency, so it cannot exercise the mechanism and the frequency-control metric is inert on
it. Training more SAEs along the zipf axis should wait until the mentors settle whether the
hypothesis is about global or within-context frequency.

---

## 2026-08-06 08:05 +03 — The default pad id silently deleted document boundaries

**Question.** Does the PCFG adapter's default padding id collide with a real token?

**How it can be answered.** `keep_mask` drops every position equal to `pad_id`, so a
collision removes real tokens with no error and no visible symptom. Assert the id is
absent from the corpus before accumulating, and see whether any run trips it.

**What we ran.** The formatting sweep at density 0.24 (`f6edabf8ccde`), the only
configuration with `document_delim` enabled.

**Result.** The guard refused to run:

```
pad id 1003 occurs in the corpus; pick one outside the vocabulary
```

`DOCUMENT_DELIM = 1003` and `vocab_size = 1004`, so the default `vocab_size - 1` is
exactly the document delimiter. The three sparser densities were unaffected — the
guard verified 1003 is absent from them.

**Interpretation.** Without the assertion this would have removed every document
boundary from the statistics and produced a complete, plausible report from the
remains. It is the failure mode this whole contract layer exists for: wrong
statistics do not crash, they return numbers.

**Answer.** Yes, it collided. Fixed in `5589030`: `pad_id` is now `vocab_size`, one
past the vocabulary. Every window is exactly `context` long so no padding is emitted
and the id never reaches the embedding — it only has to be absent from the data.

---

## 2026-08-05 17:20 +03 — Matryoshka nesting does respect the tree on clean ground truth

**Question.** Does the Matryoshka nesting itself put a parent in an earlier block than
its children, or is the architecture structurally incapable of producing a hierarchy?

**How it can be answered.** On gemma this cannot be asked — the correct ordering is
unknown, so a violation is indistinguishable from a concept we misread. The trained toy
can answer it: it has ten Matryoshka blocks *and* a known tree. Match each learned latent
to the true feature it recovered, then check every true edge runs early block → late block.

**What we ran.**

```bash
python3 -m validation.block_tree_alignment    # metrics repo, needs outputs/toy_trained/
```

**Result.** 6/6 testable edges respect the nesting. Mean block 1.7 for parents, 4.5 for
children. The 3 untestable edges are exactly the 3 children the SAE never learned — the
same ceiling that recall 0.67 reports. Feature splitting is present: 3 true features are
recovered by two latents each (19, 9, 16).

**Interpretation.** This is the control for a specific objection to the gemma result:
*maybe Matryoshka simply cannot produce a coherent hierarchy, so you measured a broken
architecture rather than found something*. It does not generalise to gemma's scale or
data — it establishes capability, not performance.

**Answer.** The nesting is not structurally incapable. On clean ground truth it produces
the right ordering, which leaves what the data distribution does to it as the explanation
for the production failure — the thing Exp 2 sweeps.

---

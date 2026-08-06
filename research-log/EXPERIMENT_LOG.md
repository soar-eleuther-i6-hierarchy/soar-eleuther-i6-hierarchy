# Experiment log

Newest first. Template and conventions: [`README.md`](README.md).

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

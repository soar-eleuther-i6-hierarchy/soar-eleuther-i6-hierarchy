ا 

# Error log

Newest first. One entry per error that cost time or could have corrupted a result.

The entries worth writing are the ones that produced **no error message**. A crash
teaches you something once; a wrong number that looks right can survive into a paper.
Each entry therefore records *how it surfaced* — and when the honest answer is "it did
not, we went looking", that is the most useful line in the entry.

## Status

Every heading carries one of `fixed` · `open` · `blocked`, so `grep '^## '` shows what is
still live without opening anything.

The status tracks the **blast radius, not the commit**. An entry is `fixed` when the affected
results are known-good again — not when the code changed. Those come apart: BOS exclusion
landed in the code long before the numbers it had corrupted were regenerated and re-read, and
an entry marked `fixed` on the strength of the commit would have said the opposite of the
truth for that whole stretch. The question this log answers is *which of my results should I
distrust*, and a code diff does not answer it.

Consequences worth stating, since they are what make the field cost something:

- **`open` and `blocked` must carry a `Closes when:` line** naming a condition someone else
  could check. Without one the status is a mood.
- **`blocked` must name what it is blocked on.** If that is a person or a machine, say so.
- **An entry cannot be `fixed` while its Prevention is a promise.** "Be careful" was already
  disallowed; the status is what makes it enforceable.

Not `WIP`: that claims someone is working on it right now, and turns into a lie the moment
you look away for a week. `open` is true whether or not anyone is on it.

If `open` ever stops being rare, the field has stopped meaning anything — that is the failure
mode to watch, not stale wording.

## The template

```markdown
## YYYY-MM-DD — <short title>   `fixed` | `open` | `blocked`

**Symptom.** What was observed. If nothing was observed, say what would have been
believed instead, and for how long.

**How it surfaced.** Crash · assertion · a number that looked wrong · found while
reading unrelated code · not found by us at all. Be specific: this is what tells the
next person which of their own results to distrust.

**Root cause.** The actual mechanism, not the layer where it appeared.

**Blast radius.** Which results are affected, which are provably not, and how you
know. Name the runs.

**Fix.** Commit, and what changed.

**Prevention.** What now makes this class of error loud instead of silent — a guard,
a test, a contract. "Be careful next time" is not prevention.

**Closes when.** Required unless `fixed`. A condition someone else could check.

---
```

---

## 2026-09-12 — Two copies of the same training sweep ran at once, writing to the same directories   `fixed`

**Symptom.** Training slowed to about 16 minutes per run against a measured 10. Nothing else
looked wrong. The progress log counted up normally, because it was one of the two logs.

**How it surfaced.** Not from the slowdown, which was read at first as ordinary variation. It
surfaced on a routine check of the process list, which showed two `train_toy.py` processes
where the script runs one at a time.

**Root cause.** The sweep was launched twice. The first launch wrapped the script in `nohup`
inside a backgrounded Bash call. That call was reported as completed, and the log it had been
writing stopped after one line, so the run was taken to be dead. It was not. The `nohup`
process had been orphaned to init and kept running for two hours, while a second launch ran
the same sweep beside it.

The mistake in reasoning is worth stating on its own: a background process whose output stops
arriving has not necessarily stopped. Output and liveness are different facts, and only one of
them was checked.

**Blast radius.** No result is affected, and this was verified rather than assumed. All 20
checkpoints written up to that point load, produce finite outputs, and are 11,116 bytes each,
so none was truncated by a second writer. Within every cell all checkpoints have distinct
weight hashes, so no run was silently overwritten by another. The two processes were executing
the same command with the same seed, so whichever wrote last produced what was intended.

The cost was wall-clock time. Each process held about 67% of a core instead of a whole one.

The near miss is the part worth keeping. Had the two launches differed in seed or in tree, the
checkpoints would have been a mixture of two sweeps, every file would still have loaded, and
nothing downstream would have objected.

**Fix.** The orphaned script and its child were killed. The surviving sweep finished at the
measured rate. Every checkpoint was validated as described above before any measurement was
run on it.

**Prevention.** Two changes in habit, neither yet in code. Launch long jobs through the task
runner rather than `nohup`, so the process is tracked rather than orphaned. And when a job is
believed finished, check `pgrep`, not only its log.

A guard is also now in the sweep script. It checked for an existing `cfg.json` before
training a cell, which reports what finished but does nothing when two processes start the
same cell within seconds of each other. Each cell is now claimed with `mkdir`, which is
atomic, so exactly one process can hold it and the other reports the cell as busy and moves
on. Tested with two copies started together: one claims, one declines.

**Closes when.** `fixed`.

---

## 2026-09-11 — The Temporal SAE was compared against a Matryoshka baseline graded before the BOS fix   `fixed`

**Symptom.** `I6-F008` reported that the T-SAE's edge density is 1,760 times lower than
Matryoshka's. The true figure against a matched baseline is 794 times, and 28 times against the
Matryoshka pair closest in block shape. The Matryoshka row also claimed 5 superparents where the
correct count is 2, and 34,061 candidate edges at `1->2` where the correct count is 621.

**How it surfaced.** Not from the comparison itself, which looked plausible. It surfaced while
running an unrelated job: a fresh collection on the node to add the `B3->B4` block pair. That run
reproduced the committed Matryoshka report exactly, 48,571 tokens and 1,473 / 621 / 1,747 edges,
which meant the baseline in use, at 48,971 tokens and 3,260 / 34,061, could not be the same
measurement.

**Root cause.** The baseline was a report graded before BOS positions were excluded from the
co-firing counts. BOS is an attention sink. With 400 documents, every feature pair collected 400
joint firings from the BOS position alone, which clears `MIN_JOINT = 30` on its own, so the
co-fire guard admitted nearly every pair in the dictionary.

Three things had to line up for this to pass unnoticed. The file was pulled from the GPU node,
where an old statistics cache still sat beside the current one. `run_metrics.py` echoes the
config recorded inside the cache, so the report it wrote carried the *old* config block and did
not raise. And the 400-token gap was read as one extra document, which is the right order of
magnitude and the wrong mechanism.

**Blast radius.** `I6-F008` and the `tsae-gemma-pipeline` table in `tsaes-tables.tex`, both now
regraded. `I6-F009` is provably unaffected: the probe table reads
`gemma-2-2b/layer_12/second_pass.json`, which was never the withdrawn file, and that file records
`bos_excluded: true`. `I6-F007` is unaffected, being a property of the T-SAE checkpoint alone. The
paper's own gemma numbers were never wrong; the committed report has been correct throughout, and
the fresh run confirms it.

**Fix.** `reporting/make_tsae_report.py` reads `gemma-2-2b/layer_12/metrics_report.json`. The old
report moved to `gemma-2-2b/layer_12/withdrawn/` with a README naming the four signals that
identify a pre-BOS file. `I6-F008`, the T-SAE directory README and the 2026-09-11 log entry are
rewritten with the corrected table.

**Prevention.** Two checks, because a rule that lives only in prose did not hold.

`reporting/temporal_common.py:read` now refuses any metrics report whose `config` block omits
`bos_excluded` or `min_joint`. Those keys entered the config block together with the guards they
name, so a report that omits one predates it. The check is on absence rather than value, which
makes it work for files written before the key existed. Verified both ways: the two current
reports load, and the withdrawn one raises.

`validation/audit_comparability.py` sweeps all 26 graded reports and reports what may be compared
with what. Run after this fix: nothing flagged. The audit itself produced two false positives on
its first run, both of the same class it exists to catch, and both are now fixed and commented.
Its limits are stated in its docstring: it does not match block width, seed or architecture.

This is another instance of a pattern this project has hit repeatedly: a plausible number
computed on a wrong basis, passing without an error. It also shows the standing rule is not
enough on its own. "Check the denominators match" was applied here, found a 0.8 percent
difference in token counts, and concluded the difference was too small to matter. The missing
step was asking what *mechanism* produced the gap, rather than whether its size seemed tolerable.
A 0.8 percent difference in tokens produced a 246-fold difference in edges, because the missing
tokens were all BOS, and BOS co-fires with everything.

**Closes when.** `fixed`.

---

## 2026-09-11 — The toy evaluator always loaded one fixed tree, whatever the checkpoint was trained on   `fixed`

**Symptom.** None. `eval_metrics.py --dataset toy` produced a complete report for nine
checkpoints trained on `configs/tree_decorrelated.json`, having scored every one of them on
activations from `configs/tree.json`. FVE, L0, feature density and the decoder heatmap were all
written without complaint.

**How it surfaced.** We ran the evaluation, then paused before quoting the numbers to check
which tree the loader had used. Nothing in the output said. The defect had already produced its
full set of files by then.

**Root cause.** `get_toy_activation_loader` takes `tree_file` with a default of
`configs/tree.json`. `eval_metrics.py` called it with two arguments and never passed the third,
and the script had no `--tree_file` option at all. Until 2026-09-11 only one toy tree existed,
so the hardcoded default was invisibly correct. Adding a second tree made it wrong without
changing any line of the evaluator.

**Blast radius.** One evaluation batch, deleted before any number was read or reported. The
nine affected directories under `eval_results/decorr_toy/` were removed and regenerated against
the correct tree. Evaluations of the `temporal_toy` checkpoints are unaffected, because those
models were trained on `configs/tree.json`, which is what the default loads. No published or
recorded number came from the bad run.

**Fix.** `eval_metrics.py` gained `--tree_file`. `SAEConfig` gained a `tree_file` field, and
`train_toy.py` records it, so a checkpoint now carries the tree it was trained on. The evaluator
defaults to that recorded tree rather than to a constant.

**Prevention.** The evaluator refuses a mismatch instead of proceeding. If a checkpoint records
one tree and `--tree_file` names another, it exits with both names in the message. For
checkpoints trained before the field existed, it prints that no tree is recorded and names the
one it is using. The general failure is that a default is correct only while one option exists;
the guard removes the silence, not the default.

---

## 2026-09-10 — Every T-SAE ever trained on the toy optimised its temporal term at exactly zero   `fixed`

**Symptom.** None. `scripts/train_toy.py --arch tsae` ran to completion, printed a falling
loss, passed the BatchTopK threshold guard and wrote a checkpoint whose `cfg.json` says
`TemporalSAEConfig`, `use_contrastive_loss: true`, `contrastive_weight: 0.1`. Nothing in the
checkpoint, the logs or the config distinguishes it from a trained Temporal SAE. What it
actually contains is a BatchTopK Matryoshka SAE. Had this not been caught, the paper's
"Temporal SAE on the toy model" row would have been a Matryoshka SAE under another name —
and the comparison against Matryoshka would have been a comparison of a model with itself.

**How it surfaced.** Not by a failure, and not by us in the sense that mattered: the
`sae-training` README already stated it in plain words under Outstanding Items — "the toy
loader yields 2D `[tokens, d]` activations, so the contrastive branch never fires on the toy
at all". It had been written down and left standing while `--arch tsae` remained a documented,
runnable option on that exact loader. It surfaced properly only on going to *run* the thing
and reading `TemporalSAE.forward` before trusting its output.

**Root cause.** `TemporalSAE.forward` gates the contrastive term on `x.dim() == 3`, and
`activations.get_toy_activation_loader` yields `[batch, d_in]` — the Bussmann tree has no time
axis to give it. On 2D input the term is `x_flat.new_zeros(())`: a constant with
`requires_grad=False`, added to nothing. Directly measured, 2D input gives
`contrastive_loss = 0.0, requires_grad=False`; 3D gives `2.079, requires_grad=True`.

The same gate exists in `PriorsInTimeSAE.forward`, so `--arch priors_in_time` on the toy has
the identical property. That one is *documented* behaviour ("trains only its novel-code path
there — a plain SAE") rather than a silent surprise, and is left as-is; it now warns.

**Blast radius.** No published result. `metrics/outputs/toy_trained/` — the Tier-2 checkpoint
behind the project's only ground-truth number (precision 1.00 / recall 0.67, 6 of 9 true
edges) — is `--arch matryoshka` and never had a temporal term, so it is provably unaffected;
`configs/recipes/toy_trained_tier2.json` pins `"arch": "matryoshka"`. No T-SAE toy checkpoint
had been trained before today. Every T-SAE number in this project post-dates the fix.

**Fix.** `toy_model.TemporalTreeSampler` supplies the missing time axis by depth-scaled
persistence, `activations.get_toy_sequence_loader` yields `[batch, seq_len, d_in]`, and
`train_toy.py` routes `--arch tsae` to it by default. The persistence construction leaves the
per-token marginal exactly unchanged (expected L0 stays at the tree's 1.12), so
`--persistence 0` is a genuine null control rather than a different dataset.

**Prevention.** Three guards, none of which is a promise:

- `train_toy.py` **refuses** `--temporal_data off` on an architecture with a temporal term,
  naming `--persistence 0` as the way to get a no-temporal-structure control without
  reverting to flat tokens; `priors_in_time` under the default warns loudly.
- `trainer.train_sae` now records `contrastive_loss` in `history`, not only in W&B. A run with
  no W&B project previously left no trace anywhere that the term had read zero throughout.
- `tests/test_tsae.py::test_contrastive_term_is_dead_on_flat_tokens` pins the zero-and-no-grad
  behaviour on 2D and the live-with-grad behaviour on 3D, so the gate cannot be quietly
  widened or narrowed. `tests/test_toy_tree.py` pins that persistence does not move the
  marginal, which is what makes the null a control.

---

## 2026-08-07 — The metrics' maths was "checked" by reading it, twice   `fixed`

**Symptom.** None, and there was nothing to see: no formula was wrong. The defect is that
nothing could have told us if one had been. "Check the math of the metrics" was reported
complete on 7 August at 18:20 and again at 22:40, both times on the strength of reading each
module against its docstring.

**How it surfaced.** Not by a failure. On being asked to *confirm* the task rather than
report it, the honest answer was that reading is not a check, and the reason it is not was
already visible in the repo's own design.

**Root cause.** Tier 1 grades **behaviour**, and the production function is what produces the
numbers it grades. So a formula that is consistently wrong still separates the two classes and
still scores 14/14. Divide coverage by the parent instead of the child everywhere, and genuine
edges still out-score pathological ones by the same ratio; nothing in the scorecard moves.

This is the same shape as the two entries above it. A calibration that names a tier rather
than a file is not checkable by reading either tier; a claim verified by reading is not
checkable at all. In each case the artefact looked verified because something adjacent to it
was.

The one that would have hurt most is metric 2a. `reconstruction.py` uses a closed form for the
error increase from ablating a feature, `g = 2a⟨d, err⟩ + a²‖d‖²`. Every number it reports is
a *ratio* of these, so a dropped factor of two rescales every gain by the same constant,
changes no verdict, appears in no scorecard row, and is invisible in every published figure.

**Blast radius.** None found. Twelve definitions were recomputed independently and all twelve
match the code, so no published number is affected. What was wrong was the confidence, not the
arithmetic — and the entry exists because that distinction is the whole subject of this log.

**Fix.** [`metrics/tests/test_metric_math.py`](../metrics/tests/test_metric_math.py)
recomputes each metric from the definition in its own docstring — by explicit loops over
tokens where possible, never by reusing the implementation — and asserts equality. The
ablation gain is checked against `‖err + a·d‖² − ‖err‖²` computed literally, one token and one
feature at a time.

**Prevention.** The test is the prevention, and it was verified in the failing direction by
injecting four real errors, none of which Tier 1 would have caught:


| injected error                                     | caught |
| ---------------------------------------------------- | :------: |
| drop the factor of 2 in the ablation gain          |   ✅   |
| divide coverage by the parent instead of the child |   ✅   |
| forget to normalise the independence null by`N`    |   ✅   |
| let the in-block graph become cyclic               |   ✅   |

It also asserts that the toy world carries the six signatures it documents, so a structure
quietly losing its intended shape fails here rather than in a scorecard row that still passes.

---

## 2026-08-07 — Fifteen published pages rendered their nav bar and nothing else   `fixed`

**Symptom.** Every `metrics_dashboard`, `superparent_sankey` and `qualitative_dashboard` on
all five gemma layers — the site's main results — opened to a working nav bar above an empty
white page. Since the layer directories were regrouped on 5 August, so roughly two days,
and on the published site rather than locally.

**How it surfaced.** Not by any check. A reader opened layer 3's qualitative page, saw
nothing, and asked whether the reason was that `qualitative_check.py` had never been run. It
had: the JSON holds 20 rows per layer and the HTML carries a 15.7 KB plotly table payload.
Both halves of the page were present and correct.

**Root cause.** The `<script src>` for the shared plotly bundle. `write_page` computes it
per page — `outputs/gemma2_2b/layer_NN/` needs `../../assets/plotly.min.js` — but the value
is baked into the file at generation time. Grouping results under `outputs/<source>/` in
`0139852` added a directory level, so every page not regenerated afterwards kept
`../assets/` and pointed one directory short. The bundle 404s, plotly never loads, and the
`<div>` it would have filled stays empty.

Nothing looked broken because the nav bar is plain HTML and CSS. It rendered, so each page
announced its layer, its source and its six sibling pages while showing no result at all —
the most confident possible presentation of nothing.

`d27a83f` fixed 295 links broken by the same move. It missed this one because a
`<script src>` is not a link in the sense that pass was walking, and because the pages it
did fix looked fixed.

**Blast radius.** 15 pages under `outputs/` and 15 more under `outputs_archive/`, which were
blank on the same terms — the archived pages carry a banner saying they are withdrawn, not
that they are empty. Nothing computed is affected: every number, every JSON report and every
figure is untouched, and the payloads were in the files the whole time. What was lost is two
days of anyone being able to read them. Provably unaffected: the in-block pages published
today, both PCFG layers and both calibration pages, all generated after the move and
verified to resolve.

**Fix.** `refresh_nav` now rewrites the plotly `src` alongside the nav block. It is the same
class of thing — a path derivable from the file's own location — so it needs no
regeneration, which matters because rebuilding these pages needs the ~810 MB cache per
layer. 30 pages repaired; all 28 live pages plus the archive now resolve their bundle.

**Prevention.** [`metrics/tests/test_site_links.py`](../metrics/tests/test_site_links.py)
resolves every `href` and `src` across `outputs/` and `outputs_archive/` — 1,151 references
over 93 pages — and exits non-zero on a target that is not there. It knows the two conventions
the site relies on, so it does not cry wolf: Jekyll serves `foo.md` at `foo.html`, and a
directory link resolves to its `README.md`. Checked in the failing direction on this exact
breakage: rewriting one page's `../../assets` back to `../assets` makes it fail and name the
page.

It earned its place the day it was written. Renaming the synthetic tier's files touched 34
files of references, and the coverage guard beside it caught a path the bulk rewrite had
missed — the class of thing that used to be found by someone opening a page.

---

## 2026-08-07 — Nine pages were dead ends, and both guards were blind to them by construction   `fixed`

**Symptom.** Nine generated pages carried no nav bar: the seven `in_block_edges.md` reports,
the paper-figures index and the archive index. Reaching one by link left you on it — no route
to the layer, the source or the site root.

**How it surfaced.** While auditing the site after the blank-pages repair, by counting pages
that contain the nav marker rather than by following links.

**Root cause.** `in_block_edges.to_md` never injected the bar, unlike `run_metrics.to_markdown`
— a consequence of that script having been written while it sat outside the pipeline, the same
origin as its gemma-only constants.

**The part worth keeping is why nothing caught it.** Two guards cover this area and each is
blind to a missing nav for a structural reason:

- `test_site_links.py` resolves every `href` and `src` and finds nothing wrong, because a page
  with no nav has no links to resolve. Its check is *are the links right*, and the failure is
  *there are no links*.
- `refresh_nav` **replaces** a nav block and needs one to match. It cannot add a missing bar,
  so it walked past all nine on every run and reported success.

A defect that sits exactly in the gap between two guards is not caught by adding a third of
the same kind. This is the same lesson as the BOS entry — six metrics failed together because
they shared an input — one level up: the guards were assumed to compose, and they do not.

**Blast radius.** Navigation only. No number, report or figure is affected; the pages
themselves were correct and complete.

**Fix.** `in_block_edges.to_md` and `make_report_figures.write_readme` inject the bar the way
every other generator does. The archive index is hand-written, so it gets the block once and
`refresh_nav` owns it from there.

**Prevention.** The count itself: zero pages under `outputs/` and `outputs_archive/` without a
nav marker, checked after every regeneration. Stated as a fact to re-check rather than a test,
which is the honest status — a page with no bar is not currently a build failure, only a
number that would move.

---

## 2026-08-07 — The strict test had no calibration, and a page said it did   `fixed`

**Symptom.** None, and none was possible. `validation/calibrate_on_synthetic_toy.py` closed its
published scorecard with: *"The 4 per-token functions (`train_probe`, `sres_rank_check`,
`negative_parent_composition`, `parent_conditioned_redundancy`) ... are calibrated in Tier 2."*
Tier 2 imports `coverage_legs`, `keep_edges`, `edge_reconstruction_condition`,
`frequency_controlled_coverage` and `frequency_buckets`, and nothing else. So **metric 2b — the
probe-based `S_res`, the strict test that decides which edges survive on gemma — had never been run
against a known answer**, and neither had the within-block metric. Believed since the file was
written.

**How it surfaced.** Not by running anything. Found while auditing which metrics the toys actually
cover, prompted by a request to check the metrics' maths. The maths was fine; the coverage was not.
No test, no dashboard and no metric could have flagged it, because an ungraded function does not
behave differently from a graded one.

**Root cause.** The claim named a **tier**, not a file. "Calibrated in Tier 2" is not checkable by
reading Tier 2 — you have to read its import list and know which functions the sentence meant — so
it inherited the credibility of the 9/9 scorecard printed above it without ever being tested.

The mechanical reason it was never closed: the four functions read per-token residuals and firing
masks, and `toy_world._reduce` computed exactly those (`feats`, `resid_err`, `W_dec`) and threw them
away, returning only the reduced statistics. The toy could have graded them the whole time; nothing
said it did not.

**Blast radius.** No published number. The gemma and PCFG `S_res` figures were produced by the same
code before and after, and the metric functions themselves were unchanged — what was missing was
evidence that they do what they claim. Concretely, the sentence "10 of 1700 edges pass `S_res`" was
never wrong; it was ungrounded. Tiers 1 and 2 both re-run to their documented values (9/9 across
seeds 0–5; precision 1.00 / recall 0.67), so nothing they *did* cover is affected.

**Fix.** `build_world` now also returns `resid`, `fired` and `W_dec`; the toy gained three
structures with known answers (an absorbed child, a shared-topic pair, a within-block containment +
duplicate pair) and the scorecard five rows. **14/14 across seeds 0–7, covering 21/21 metric
functions** — was 13. Not yet committed at time of writing.

Two things had to be corrected while closing it, and both are findings rather than plumbing:

- The toy's own geometry was wrong for this test. At `d_model = 16`, 42 random unit directions
  correlate up to 0.73, and three genuine edges failed the rank rule **with the child at rank 0 and
  the true parent pushed to rank 6–8 by features it has nothing to do with**. That is a fact about
  42 directions in 16 dimensions. `D_MODEL` is now 64, with the residual-error energy held fixed so
  raising it does not shrink metric 2a's denominator.
- The first version of the new row asserted the superparent's edges would be *rejected* by `S_res`.
  Wrong: the rank rule is a geometry test, so an unrelated parent passes exactly when chance puts it
  in the top *k* of *D*. Observed 2–4 of 20 against 2.4 expected. The row now asserts what the rule
  claims — every true parent accepted, an unrelated one no better than chance.

**Prevention.** [`metrics/tests/test_calibration_covers_metrics.py`](../metrics/tests/test_calibration_covers_metrics.py):
parses the calibration and asserts every function in `metrics.__all__` (plus the four graded ones
outside it) appears as a call there. Checked in both directions — it passes now, and injecting a
name nothing calibrates makes it fail. A prose sentence about which tier covers what is not
checkable; a call site is. Adding a metric without a scorecard row now fails a test instead of
inheriting the previous sentence's credibility.

---

## 2026-08-07 — `in_block_edges.py` was still gemma-only, because it was not a stage   `fixed`

**Symptom.** None yet. The script read `config.BLOCK_RANGES` — gemma's 32768 latents in five blocks
— and loaded the released gemma decoder whatever cache it was handed. Pointing it at the published
PCFG cache (1792 latents in eight blocks) would have graded four block pairs from the wrong feature
columns and then raised `IndexError` on the fifth. The first four would have looked entirely normal.

**How it surfaced.** Not by running it. Found while answering whether the in-block metric could run
on PCFG at all.

**Root cause.** The same two constants stages 03 and 04 stopped holding in `89294a4` (see the entry
below). This file kept them for a reason worth recording: **it was not part of the pipeline.** It
sat outside `run_pipeline.py` as a commented-out `ASIDE`, so the sweep that converted every stage to
read its structure from the file being graded simply never reached it.

That is also why it had never been run on any layer, gemma included. Nothing declared it, so nothing
missed it.

**Blast radius.** Nothing published. The script had produced no output on any source, which is the
one circumstance in which this class of bug costs nothing.

**Fix.** Structure now comes from `run_metrics.source_structure(stats)`; the decoder from the run's
own `w_dec.pt` via the shared `load_w_dec`, with a feature-count assertion and a `--w-dec` override;
and the blocks to analyse are read from what the file actually carries (`within_cofire`) rather than
from `config.IN_BLOCK_BLOCKS`, which is a *collection* directive for stage 01 and reimposed gemma's
`[0, 1, 2]` on a source with eight blocks. Verified by running it end to end on
`outputs/pcfg/layer_01/`: eight blocks graded, 550 directed edges and 78 duplicates in B0.

**Prevention.** It is now stage **01c** in `run_pipeline.py`, numbered before 02 because that is
where its dependencies put it — it needs 01 and nothing after. `--list` shows it as `WAIT` beside
the other unrun stages, so "never run on any layer" is now visible rather than something you have to
already know. Its two pure functions are also graded by the Tier-1 calibration as of today, which is
the guard the entry above adds.

---

## 2026-08-07 — The reporting stages were still gemma-only after `collect()` stopped being   `fixed`

**Symptom.** None yet, and that is the entry. `run_metrics.py` was taught to read the block
structure from the stats file when the PCFG adapter landed; `run_token_metrics.py` (stage 03)
and `reporting/visualize.py` (stage 04) were not, and kept slicing `config.BLOCK_RANGES` —
gemma's 32768 latents in five blocks. Grading a PCFG file (1792 in eight) through them would
have drawn dashboards and computed S_res over the wrong feature columns for pairs B0→B1
through B3→B4, then raised an `IndexError` on the fifth pair, which does not exist in gemma's
structure. The first four would have looked completely normal: right shapes, plausible
numbers, a page that renders.

Stage 03 had a second one. It called `sae_utils.load_sae()` unconditionally — the *released
gemma decoder* — to turn each probe into per-feature correlations, whatever dictionary the
token cache came from.

**How it surfaced.** Not by running. Found while reading the reporting path to answer whether
a PCFG run could be published as a page beside the gemma layers. Nobody had run stage 04 on a
non-gemma cache, because until today no non-gemma cache had a token cache to run stage 03
from, so there was nothing to publish and no reason to look.

**Root cause.** `collect()` was split out of stage 01 and made source-agnostic; the claim
"the same battery across every source" was then treated as established. But the battery is
five stages, and only stages 01 and 02 were ever converted. The block structure travels in
the stats file exactly so that no module has to hold it as a constant — and the two stages
that still held it as a constant were the two nobody had exercised off-gemma.

**Blast radius.** No published number is affected. Every gemma result was produced with
`BLOCK_RANGES` describing gemma, which is what those slices are for. Verified rather than
assumed: layer 6's `metrics_report.md` regenerates byte-identical from its committed
`metrics_report.json` under the new code, and the toy-calibration page's nav bar is
byte-identical to the published one. No PCFG page existed before today, so nothing wrong was
ever shown. This is a near miss, logged because the next person to grade a PCFG run would have
been the first to hit it — and would have hit it as four correct-looking pairs, not as a crash.

**Fix.** `metrics` `89294a4`. Both stages now call `run_metrics.source_structure(stats)`;
stage 03 takes the run's own decoder from `RUN_DIR/w_dec.pt` (written by the adapter,
overridable with `--w-dec`) and asserts its feature count against the statistics, so a
mismatch names itself instead of failing inside `sres_for_pair`.

**Prevention.** `metrics/tests/test_dashboards_generic.py` — an 8-block stub through stages
02, 03 and 04, asserting the pages describe the file they were built from. It was checked
against the old behaviour: reverting the block-structure fix makes it fail. This is the same
guard `test_collect_generic.py` provides for stage 01, which is precisely the guard that did
not extend to the stages downstream of it.

---

## 2026-08-06 — BOS satisfied the joint-support guard for every pair in the dictionary   `open`

**Symptom.** None. Five layers of results, a published site, and four claims — three of which
were false. Believed for 19 days, from the 18 July runs until 6 August. Two of them were
inverted, not merely imprecise: "coverage over-proposes, 94–99.9% of edges die" became 74–90%
of B0→B1 candidates *passing*, and "deep block pairs carry no signal" became 9–41%.

**How it surfaced.** It did not. We went looking, and only because a *different* question was
being asked: whether the 24 July superparent-gate change had invalidated the 18 July numbers.
The first attempt re-ran stages 02 and 02b against the same cache, found nothing moved, and
nearly closed the question — which proved only that stage 02's thresholds are not what moves,
since the input was byte-identical. Rebuilding the cache was what exposed it. Nothing in six
metrics, five layers or any dashboard had flagged anything.

**Root cause.** BOS is an attention sink: effectively every feature fires on it. With
`PREPEND_BOS = True` and `N_DOCS = 400`, every parent/child pair in the dictionary — including
pairs that never co-occur anywhere else — accumulated 400 joint firings. `MIN_JOINT` is 30.

The guard exists precisely to kill pairs whose co-firing is coincidence, and one token handed
every pair 13× the co-firing it needed to clear it. So the guard passed everything, and each
metric downstream was grading a candidate set that should never have existed.

**Blast radius.** Every gemma number published before 6 August, on all five layers. Concretely
at L6: B2→B3 candidates 4,704,312 → 762; B0→B1 reconstruction pass 6.3% → 85.9%;
frequency-driven share 60.8% → 1.0%; survival 0.441 → 1.031. Token counts 48,971 → 48,571 on
every layer — exactly 400, one BOS per document, which is the cheapest way to tell a v1
artifact from a v2 one at a glance.

Provably unaffected: **one** claim, "it is not a tree" (89–100% of children keep ≥2 parents).
It survives because it is the only one that does not depend on the candidate set — a ratio
over children that already have a parent. Also unaffected: the PCFG work, which never used
these caches, and both toy calibration tiers, which build their own data.

That single survivor is the finding underneath the finding. Five of the six metrics read the
same co-firing matrix. The battery was designed as independent detectors that would fail
independently; they share an input, so one contaminated token position defeated them together.
Agreement among them is much weaker evidence than the design implies.

**Fix.** BOS exclusion was already in the code (`schema_version: 2`, `bos_excluded: True`) —
the corrupted caches predated it. All five layers regenerated from stage 01 through
`run_pipeline.py` on GPU 3, v2 caches uploaded to the Hub under `v2/layer_NN/exp0_stats.pt`,
v1 results archived under `metrics/outputs_archive/`, and the site updated. L24 was rerun from
two different commits to check the result does not depend on which merge the code sits at —
identical. `cf96fd4` withdrew the two hand-built pages whose entire content was the inverted
fractions.

**Prevention.** Not yet in place, and saying otherwise would be the same mistake in a
different form. `contracts/validate_stats.py` *would* reject a v1 cache — it requires
`schema_version == 2` — but **nothing in the pipeline calls it**: neither `run_metrics.py` nor
`run_pipeline.py` imports it. The guard exists and is not wired in. It is also across a repo
boundary: the validator lives in the umbrella repo, the pipeline in the `metrics` submodule,
so wiring it is a real change and not a one-line import.

Second gap, from the same episode: three separate times a committed artifact under
`outputs/` was read as fresh output, once producing a spurious "L3 was never contaminated"
reading that survived until a token count was checked. An output directory under version
control makes a stale file indistinguishable from a fresh one by inspection.

**Closes when.** Two conditions; **one is now met.**

- ~~(2) the Tier-3 semantic reading of the v2 survivors has been done~~ — **done 7 Aug.** All
  40 survivors at B0→B1, eight per layer across the five layers, read against Neuronpedia
  labels. About half are genuine refinement; the commonest failure is a semantic parent with a
  function-word or formatting child, which is topical co-occurrence and which nothing in the
  battery detects. See the 21:30 entry in `EXPERIMENT_LOG.md`. The site no longer shows
  numbers no human has read.
- **(1) Stage 01 refuses to hand a cache to stage 02 unless it validates**, so a v1 file cannot
  be graded at all. **Still open**, and unchanged: `contracts/validate_stats.py` rejects a v1
  cache — verified today against all three committed stats files plus its self-test — but no
  stage imports it. It is also across a repo boundary, the validator here and the pipeline in
  the submodule, so wiring it is a real change.

Leaving this entry `open` on (1) alone is the honest state. It was `open` on both for a day
after (2) was satisfied, which overstated what was blocking and understated what had been
done — the kind of drift the status field exists to prevent.

---

## 2026-08-06 — The validator inferred a dictionary size that does not exist   `fixed`

**Symptom.** `validate_stats.py` rejected a correct stats file:
`fire_count: expected shape (10,), got (12,)`.

**How it surfaced.** The first run of the new index-block path, on a deliberately
built stub — before any real file used it.

**Root cause.** Extending the contract to allow `block_indices`, I derived the
dictionary size the same way the range form does — from the blocks. `d_sae` became
`max(index) + 1`.

That holds for `block_ranges`, where blocks tile the dictionary by construction. It
does not hold for index blocks, and the reason is the whole point of them: the
trained toy groups only the latents that **matched a true feature** and leaves the
rest out. Seventeen of twenty latents matched, so the blocks cover 17 features
while `fire_count` is over all 20. A dictionary is longer than its blocks, and
nothing in the file says how much longer.

**Blast radius.** None. Caught on the first stub run, before the toy adapter or any
real checkpoint went through it.

**Fix.** `block_sizes()` returns `None` for `d_sae` in the index form, and the
caller checks each index against `fire_count`'s actual length instead. A second
guard came out of the same reading: no feature may appear in two blocks, since it
would be counted as parent and child of itself somewhere and no metric could tell.

**Prevention.** Already in place, and it worked: the validator is exercised against
data known to be good, not only against corruption. This is the third time that
has caught a wrong **spec** rather than wrong data — the `g_parent_sum` sign check
and the stub test's own pad id were the others. A validator that has never been run
against something correct is untested in the direction that matters.

---

## 2026-08-06 — Default pad id was the document delimiter   `fixed`

**Symptom.** None, by design of the guard. Without it: every document boundary would
have been dropped from the statistics and a complete, plausible report produced from
the remains.

**How it surfaced.** An assertion in `adapters/from_pcfg.py` refused to run:
`pad id 1003 occurs in the corpus`. It fired on the first configuration that enables
`document_delim` — the formatting sweep at density 0.24 — after three sparser
densities had already passed.

**Root cause.** `pad_id` defaulted to `vocab_size - 1` = 1003, which is
`DOCUMENT_DELIM`. `keep_mask` drops every position equal to `pad_id`, so the delimiter
would have been treated as padding.

**Blast radius.** Nothing. The three runs completed before the guard fired
(`c325cc965ffa`, `3915659d6f6c`, `f98ccd6c7355`) have `document_delim` off, and the
guard verified 1003 is absent from each. No published result touched.

**Fix.** `5589030` — `pad_id` is now `vocab_size`, one past the vocabulary. Every
window is exactly `context` long so `right_pad` emits no padding and the id never
reaches the embedding; it only has to be absent from the data.

**Prevention.** The guard is the prevention, and it worked before any number existed.
Keep it: an adapter that quietly drops a token class is indistinguishable from one that
works.

---

## 2026-08-05 — Stage 02 accepted a non-gemma stats file and returned a full report   `fixed`

**Symptom.** None. `run_metrics.py` would take a stats file from any source and produce
a complete `metrics_report.json` with plausible numbers.

**How it surfaced.** Not by running it. Found while reading `run_metrics.py` to check
whether the stage-01 refactor was sufficient — lines 84–85 read `C.BLOCK_RANGES` from
the gemma config module rather than from the file being graded.

**Root cause.** Block boundaries were module globals. A PCFG dictionary (1792 latents
in 8 blocks) sliced with gemma's ranges (32768 in 5) yields tensors of the wrong
columns, and every metric downstream is a pure function of those tensors — so they all
return numbers, and none of them can tell.

**Blast radius.** No published gemma result: that path was always correct, and the
fallback keeps it byte-identical (verified against a config with no `block_ranges`, and
against a file with no `config` at all). Any non-gemma report produced before
`4a17f76` is wrong — as far as we know none exists, since the adapter did not exist
either.

**Fix.** `4a17f76` — `source_structure(stats)` reads `block_ranges` and
`sibling_blocks` from the file's own config, falling back to the module for older files.

**Prevention.** `contracts/validate_stats.py` checks shapes against the file's declared
`block_ranges` before the metrics see it, and `tests/test_collect_generic.py` runs the
accumulation on a 28-feature dictionary so a reintroduced global fails loudly.

---

## 2026-08-05 — Contract validator rejected known-good data   `fixed`

**Symptom.** `validate_stats.py --self-test` failed on the synthetic toy:
`g_parent_sum: contains negative values`.

**How it surfaced.** The self-test, on its first run, against data known to be correct.

**Root cause.** The spec was wrong, not the data. `g_parent_sum` and `g_child_sum` are
reconstruction *gains*: ablating a feature can improve the reconstruction, so a negative
entry is a real measurement. The validator asserted non-negativity across all
accumulators because counts and energy sums are non-negative.

**Blast radius.** None — caught before the validator was used on anything.

**Fix.** A `SIGNED` set exempting the two gain accumulators; every other tensor still
must be ≥ 0.

**Prevention.** This is why the self-test runs against known-good data *and* against
deliberate corruption. A validator that only ever passes is worthless; one that fails on
correct data is worse than none, because it trains you to ignore it.

---

## 2026-08-05 — Stub test dropped a quarter of its own tokens   `fixed`

**Symptom.** `tests/test_collect_generic.py` reported 124 tokens where 180 were
expected.

**How it surfaced.** An explicit assertion in the test comparing `total_tokens` against
`sum(len(s) - 1 for s in seqs)`.

**Root cause.** The test's own fixture, not the code under test. It sampled token ids
from `0..D_VOCAB-1` while passing `pad_id=0`, so every genuine token 0 was masked as
padding.

**Blast radius.** The test only. But it is the same class of error as the pad-id
collision above, found a day earlier and in a fixture rather than in an adapter — which
is a fair warning about how easy this mistake is to make.

**Fix.** Sample ids from `1..D_VOCAB-1` and reserve 0 for padding.

**Prevention.** Assert a token-count invariant in any harness that masks positions. The
assertion is what turned a silent 31% data loss into a one-line failure.

---

## 2026-08-05 — `\b` word boundaries silently no-op in macOS sed   `fixed`

**Symptom.** A normalisation step meant to prove the refactored accumulation loop was
byte-identical reported spurious differences.

**How it surfaced.** The diff showed `C.` on one side and `cfg.` on the other after a
substitution that should have unified them.

**Root cause.** BSD `sed` does not support `\b`, so `s/\bcfg\./C./g` matched nothing
and failed silently — no error, exit code 0.

**Blast radius.** None; a verification step, not a result. But it briefly suggested the
refactor had changed behaviour when it had not.

**Fix.** Dropped `\b`, and handled `getattr(cfg,` separately since it has a comma
rather than a dot.

**Prevention.** When a substitution is load-bearing for a correctness claim, assert it
changed something. A no-op `sed` and a successful one both exit 0.

---

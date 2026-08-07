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

**Closes when.** (1) Stage 01 refuses to hand a cache to stage 02 unless it validates, so a
v1 file cannot be graded at all; and (2) the Tier-3 semantic reading of the v2 survivors has
been done. Until (2), the site shows regenerated numbers no human has read — the claims were
withdrawn, and nothing has yet been put in their place.

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

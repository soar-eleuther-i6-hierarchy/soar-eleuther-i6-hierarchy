# Error log

Newest first. One entry per error that cost time or could have corrupted a result.

The entries worth writing are the ones that produced **no error message**. A crash
teaches you something once; a wrong number that looks right can survive into a paper.
Each entry therefore records *how it surfaced* — and when the honest answer is "it did
not, we went looking", that is the most useful line in the entry.

## The template

```markdown
## YYYY-MM-DD — <short title>

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

---
```

---

## 2026-08-06 — The validator inferred a dictionary size that does not exist

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

## 2026-08-06 — Default pad id was the document delimiter

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

## 2026-08-05 — Stage 02 accepted a non-gemma stats file and returned a full report

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

## 2026-08-05 — Contract validator rejected known-good data

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

## 2026-08-05 — Stub test dropped a quarter of its own tokens

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

## 2026-08-05 — `\b` word boundaries silently no-op in macOS sed

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

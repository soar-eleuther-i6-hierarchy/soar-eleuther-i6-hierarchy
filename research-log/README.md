# `research-log/` — the lab notebook

Not a changelog: commits record what changed in the code, these record what we ran,
what we learned, and what broke.

| File | Holds | Shape |
| --- | --- | --- |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | one entry per experiment run — question, method, numbers, reading, answer | append-only, newest first |
| [`ERROR_LOG.md`](ERROR_LOG.md) | one entry per error that cost time or could have corrupted a result | append-only, newest first |
| [`TEMPORAL_SAE.md`](TEMPORAL_SAE.md) | the T-SAE workstream as it currently stands: construction, numbers, reference comparison, what is still open | **rewritable**, kept current |
| [`PRIORS_IN_TIME.md`](PRIORS_IN_TIME.md) | what the Priors in Time method's hierarchy actually is, and why the metrics do not apply to it unchanged | **rewritable**, kept current |

Two further notes are kept in `local/`, which is not committed: one describes the GPU node,
and one records design decisions and the reasoning behind them.

The third file is a different kind of object from the first two and the difference is the
point. The logs are *evidence*: append-only, never edited, so a reading can be revisited
against what was actually observed. A workstream file is the *current picture* of one method —
it is meant to be rewritten as the picture changes, and it carries no history. Numbers in it
are pointers to the log entry that produced them, not independent claims. Keep it that way: if
a result in there ever disagrees with its log entry, the log entry wins.

Add one per method only when that method has enough moving parts that the log alone stops
being navigable. If these start accumulating for every small thing, they have stopped meaning
anything — the same failure mode `open` has in the error log.

Both are append-only: an entry is never edited after the fact. When a reading turns out
wrong, a new entry supersedes it and links back — that is what makes the log evidence
rather than a summary. The one exception is a personal name: these repositories are public, and
removing a name is a privacy correction, not a revision of the record. Write the role instead.

Project hygiene — stale numbers still circulating, dead links, what needs regenerating —
goes in the umbrella [`README.md`](../README.md) status table. It is not a result, so it
does not belong in any of these three.

## When to write

**An experiment entry** when a result would change what someone does next. Skip
refactors, renames and cleanups; those live in commit messages.

**An error entry** when something broke **quietly**. A crash teaches you once; a wrong
number that looks right can reach a paper.

## The experiment template

Copy this block, fill it, put it at the top of `EXPERIMENT_LOG.md`.

```markdown
## YYYY-MM-DD HH:MM TZ — <short title>

**Question.** The one thing we wanted to know, phrased so it can come out wrong.

**How it can be answered.** What evidence would settle it, and why that evidence
is decisive rather than merely suggestive.

**What we ran.** Exact commands and inputs — paths, run hashes, seeds, token
counts. Enough that someone else reproduces it without asking.

**Result.** The numbers. A table if there is more than one. No interpretation yet.

**Interpretation.** What the numbers mean, what they do *not* mean, and the
caveats that would change the reading. Say which alternative explanations survive.

**Answer.** One or two sentences answering the question directly — including
"we still cannot tell, because X" when that is the truth.

---
```

`ERROR_LOG.md` carries its own template, built around *how it surfaced*.

## Rules that keep this useful

**The question comes before the run.** If you cannot write the question first, you are
not running an experiment, you are looking around. That is fine — but it does not go
here until it produces a question.

**A question must be able to come out wrong.** "Does formatting density produce
frequency-driven edges?" can. "Is the pipeline working?" cannot.

**Separate the result from the interpretation.** They age differently: numbers stay
true, readings get revised. Someone re-reading in a month must be able to reinterpret
without re-running.

**Record what would have changed your mind.** A finding without stated caveats is not
reproducible reasoning, it is a claim.

**Negative and null results get entries too.** An entry whose Answer is "we still cannot
tell, because X" belongs here as much as one that settles something — a log that only
holds what worked lies by omission, and the project plan asks for negative results
explicitly. That is also why the file is an *experiment* log rather than a findings list.

**Paths and hashes, not descriptions.** `zipf_sweep/13df3dd54c16-s1` is checkable; "the
high-zipf run" is not.

**Never edit an old log entry's Result.** If it turns out wrong, add a new entry that
supersedes it and link back.

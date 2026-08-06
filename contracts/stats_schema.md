# The cached-statistics contract

Every hierarchy metric is a pure function over cached statistics. That is what lets one metric
implementation grade a gemma-2-2b SAE, a trained toy SAE and a PCFG SAE — and it is the reason a new
experiment needs an *adapter*, not a new metric.

This file is the normative spec. [`validate_stats.py`](validate_stats.py) enforces it.

```bash
python3 contracts/validate_stats.py metrics/outputs/layer_06/exp0_stats.pt
python3 contracts/validate_stats.py --self-test     # checks the spec against the synthetic toy
```

## Two shapes, both legitimate

The codebase carries the contract at two levels. Knowing which one you are producing is the first
decision an adapter makes.

| | `full` | `slice` |
| --- | --- | --- |
| Written by | `metrics/collect_statistics.py` → `exp0_stats.pt` | `metrics/validation/toy_world.py` |
| Read by | `metrics/run_metrics.py` | every metric function directly |
| Covers | all block pairs at once | exactly one parent→child block pair |
| `cofire` | `dict` keyed `"{p}->{c}"` → `[P, C]` | bare tensor `[P, C]` |
| `within_cofire` | `dict` keyed by block index → `[C, C]` | bare tensor `[C, C]` |
| `fire_count` | `[D_SAE]`, the whole dictionary | `[P + C]`, this pair only |
| Also carries | `schema_version`, `config`, `pairs` | `P`, `C`, `fire_p`, `fire_c` |

`run_metrics.analyse_pair()` cuts a `slice` out of a `full` file before calling the metrics, so
`slice` is the true minimal unit — but that cut is currently inlined rather than exposed as a
function.

**An adapter for a new SAE source should emit `full`**, so `run_metrics.py` runs against it with no
changes at all.

## `full` — required keys

### Corpus level

| Key | Type | Meaning |
| --- | --- | --- |
| `schema_version` | `int` | must be `2` — BOS excluded, energy/union accumulators present |
| `fire_count` | `[D_SAE]` | per-feature fire count over the corpus slice |
| `total_tokens` | `int > 0` | tokens entering the statistics, after BOS and padding are dropped |
| `token_counts` | `[vocab]` | per-token-id occurrence counts |
| `buckets` | `[vocab]` | frequency bucket per token id; same shape as `token_counts` |
| `pairs` | `list[(p, c)]` | which adjacent block pairs were accumulated |

### Per block pair, keyed `"{p}->{c}"`

| Key | Shape | Feeds |
| --- | --- | --- |
| `cofire` | `[P, C]` | coverage legs (metrics 1a/1b), independence null (6) |
| `cofire_by_bucket` | `[K, P, C]` | token-frequency control (5) |
| `g_parent_sum` | `[P, C]` | reconstruction condition (2a) — **signed** |
| `energy_cofire` | `[P, C]` | per-child energy share (1c) |
| `energy_total` | `[P]` | denominator for the energy share |
| `union_count` | `[P]` | exact joint-child support coverage `R_supp` |
| `union_energy` | `[P]` | exact joint-child mass coverage `R_mass` |

### Per child block, keyed by block index

| Key | Shape | Feeds |
| --- | --- | --- |
| `err_sum_c` | `[C]` | reconstruction condition (2a) |
| `g_child_sum` | `[C]` | reconstruction condition (2a) — **signed** |
| `fire_c_by_bucket` | `[K, C]` | token-frequency control (5) |
| `within_cofire` | `[C, C]` | sibling redundancy (3) and in-block edges (7) |

`within_cofire` is keyed over the union of `SIBLING_BLOCKS` and `IN_BLOCK_BLOCKS`, so it can contain
blocks that never appear as a child in `pairs`.

### How a block is declared

**A block is a set of feature indices.** Matryoshka's happen to be contiguous prefixes, and
nothing in the metrics requires that — every one of them is a matrix product over selected
columns. Two forms, and a file uses exactly one:

| Form | Field | For |
| --- | --- | --- |
| ranges | `block_ranges` — `[(0,128), (128,512), …]` | contiguous blocks: gemma, PCFG, TinyStories |
| indices | `block_indices` — `[[0,3,8], [1,2,5,7,9]]` | groups that are not contiguous |

`block_indices` **wins when both are present**, and readers must check for it first.

The trained toy is why the second form exists. It indexes by *which true feature each latent
recovered* — matched by decoder cosine — so its parent and child groups are scattered index
lists. The alternative was permuting the dictionary so they became contiguous, which would
have made `B0→B1` mean "true parents → true children" for the toy and "first 128 → next 384"
for gemma: the same field name carrying two meanings, with nothing in the file to say so.

Two rules the validator enforces for `block_indices`:

- **No feature in two blocks.** It would be counted as parent and child of itself somewhere
  and no metric could tell.
- **Every index within `fire_count`.** Note that index blocks need *not* cover the
  dictionary — the toy groups only the latents that matched a true feature — so `fire_count`
  is legitimately longer than the blocks, and `d_sae` is not derivable from them.

### `config`

| Field | Why it is required |
| --- | --- |
| `block_ranges` *or* `block_indices` | **the field that makes the object source-agnostic** — declares block membership, so the metrics never need to know which model produced the dictionary |
| `layer`, `sae_id` | provenance; goes into the report and the paper's figure captions |
| `matryoshka_steps` | the nesting schedule the ranges were derived from |
| `fire_threshold` | what counted as "firing"; changes every count downstream |
| `context_size`, `sibling_blocks`, `min_joint`, `bos_excluded` | read by the metrics or the report |

Also written and worth carrying: `sae_release`, `sae_source`, `n_docs`, `freq_high_mass`,
`freq_mid_mass`.

## Invariants the validator checks

Shape conformance is the easy half. These are the checks that actually catch a wrong adapter:

- **`cofire <= fire_count`** on both sides. A pair cannot co-fire more often than either feature
  fires alone; a violation means the masks are misaligned.
- **`cofire_by_bucket` sums over buckets back to `cofire`.** This is what catches an adapter that
  bucketed against the wrong vocabulary — the single most likely PCFG mistake, since a PCFG corpus
  has its own tiny vocabulary rather than gemma's.
- **No NaN or Inf** anywhere.
- **No negatives**, except `g_parent_sum` and `g_child_sum`. Those are reconstruction *gains*:
  ablating a feature can improve the reconstruction, and a negative gain is a real measurement.
- **`token_counts` and `buckets` are the same shape.**

## Writing an adapter

1. Emit `full`. Declare block membership from your SAE's own structure — not from
   `metrics/config.py`, whose values are gemma-specific (`[128, 512, 2048, 8192, 32768]`).
   Use `config.block_ranges` if the blocks are contiguous prefixes, `config.block_indices`
   if they are not.
2. Run `validate_stats.py` on the result. Shape errors are cheap to fix here and expensive to
   diagnose after the metrics have produced plausible-looking numbers.
3. **Reproduce a known number before trusting it.** The trained-toy tier reports precision 1.00 and
   recall 0.67 (6 of 9 edges, zero false positives). An adapter that routes the toy through the
   generic path must land on exactly those figures. Silent drift is the real risk: wrong statistics
   still produce numbers, just wrong ones.

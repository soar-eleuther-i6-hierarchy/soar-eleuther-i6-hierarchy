# Adapters — SAE source → cached statistics

Every hierarchy metric is a pure function over cached statistics. A new SAE source therefore needs an
**adapter**, not a new metric. This directory is where they live, because an adapter is by definition
about the relationship between two repos and cannot belong to either.

`sae-training`'s own README names this as its remaining open item:

> **Metrics Handoff (Exp 0)**: Ensure the saved SAE models can be seamlessly handed off to the
> `experiment_0` repository for calculating post-training graph metrics.

| Adapter | Source | Unblocks | State |
| --- | --- | --- | --- |
| `from_toy.py` | Matryoshka trained on Bussmann's tree | Tier 2 / Exp 1 | not written |
| `from_pcfg.py` | Matryoshka on a PCFG base transformer | **Exp 2** | not written |
| `from_tinystories.py` | Matryoshka / T-SAE / Priors-in-Time on TinyStories | Exp 3 | blocked upstream |

## Prerequisite: split `collect()` out of `main()`

`metrics/collect_statistics.py` currently mixes two concerns in `main()`. Only three things vary by
source:

| Line | What |
| --- | --- |
| `collect_statistics.py:130-131` | `U.load_model(device)` + `U.load_sae(device)` |
| `collect_statistics.py:136-141` | `load_dataset(C.DATASET)` → texts |
| `collect_statistics.py:142` | `tokenize_docs(...)` → `seqs` |

Everything after that — every accumulator, the main pass, `torch.save` — depends only on `model`,
`sae`, `seqs` and the config. Extracting it makes every adapter thin:

```python
def collect(model, sae, seqs, *, device, cfg=C, out_path=None, source_meta=None):
    """Everything from the frequency buckets through torch.save.
    Does not know where the model came from."""
```

The friction is `config.py`: `C.BLOCK_RANGES`, `C.D_SAE` and friends are module globals holding
gemma-specific values (`MATRYOSHKA_STEPS = [128, 512, 2048, 8192, 32768]`, `D_SAE = 32768`). A PCFG
SAE has neither. Hence the `cfg=C` parameter — the default behaviour stays byte-identical, and an
adapter passes a small object with the same attribute names.

**Acceptance test for the refactor:** `python3 collect_statistics.py --docs 16` must produce exactly
the same `exp0_stats.pt` as before. A refactor that changes any number has broken something.

## Writing an adapter

1. Emit the `full` shape — see [`../contracts/stats_schema.md`](../contracts/stats_schema.md) — so
   `run_metrics.py` runs against it unmodified.
2. Set `config.block_ranges` from the SAE's own nesting schedule, never from `metrics/config.py`.
3. Validate before trusting:
   ```bash
   python3 contracts/validate_stats.py <your_output>.pt
   ```
4. **Reproduce a known number.** Write `from_toy.py` first even though Exp 2 needs `from_pcfg.py`,
   because the toy has a published answer: precision 1.00, recall 0.67, 6 of 9 edges, zero false
   positives. An adapter that routes the toy through the generic path must land on exactly that.

Step 4 is the one that matters. Wrong statistics do not crash — they produce plausible numbers that
are simply wrong, and nothing downstream can tell. The published toy figures are the only tripwire
available.

## `from_pcfg.py` — the inputs

A PCFG run directory, written by the two pipelines:

```
$PCFG_OUTPUT_ROOT/<experiment>/<grammar_hash>/
├── model.pt        TransformerLens HookedTransformer — activations via run_with_cache
├── corpus.bin      the token stream; no tokenizer needed
└── sae/matryoshka_hook_resid_post_L{layer}/
    └── training_metrics.json   grammar + base-model config, saved to ease this handoff
```

Activations come from `blocks.{layer}.hook_resid_post`, the same hook the SAE was trained on.

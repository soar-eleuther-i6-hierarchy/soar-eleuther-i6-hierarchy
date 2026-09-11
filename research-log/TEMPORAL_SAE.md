# Temporal SAEs on the toy hierarchy

## Scope

This file describes the current state of the Temporal SAE (T-SAE) work. It covers what we
built, what we measured, and what remains open. It is not a log. The dated entries are in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), and the defect that prompted this work is in
[`ERROR_LOG.md`](ERROR_LOG.md). Those files are append-only. This file is revised as our
understanding changes.

**Status, 2026-09-11.** Training and standard evaluation are complete. A follow-up experiment
on a second tree shows that the first block selects features by firing rate rather than by
depth. No condition tested here recovers hierarchy into the first block. We have also not run
the hierarchy metrics on these checkpoints.

## Summary

We report three findings.

First, the original toy data could not train a T-SAE. The contrastive term requires sequences,
and the toy generates independent samples. The term stayed at exactly zero for every run. We
therefore built a toy generator with temporal structure.

Second, we detect no benefit from the contrastive term as the paper defines it. A Matryoshka
SAE trained on the same data, without the term, scores the same on the T-SAE objective. The
differences we measure are smaller than the variation across seeds.

Third, the term as implemented in the authors' released code does change the result. The paper
and the released code use different similarity functions. On the original tree, the released
version reaches all three parent features in the first block on every seed.

That third result does not mean what it appears to mean. In the original tree, parents are also
the most frequent features. A second tree with rare parents and frequent distractors separates
the two properties. There, the same condition places 0.667 parents in the first block instead of
three. The first block follows firing rate, not depth. Section 6.1 gives the numbers.

## 1. Why the toy required modification

T-SAE (arXiv:2511.05541) adds a contrastive term to a Matryoshka SAE. The term encourages
latents in the first block to take similar values on adjacent tokens. The motivation is that
semantic content changes slowly across a sequence while syntactic content changes quickly.

The term is computed only when the input has a temporal axis. `TemporalSAE.forward` tests
`x.dim() == 3`. The toy activation loader returns two-dimensional data, because the tree
generates independent samples. The contrastive term therefore evaluated to exactly `0.0` in
every step of every toy run.

No error was raised. The saved configuration still recorded `use_contrastive_loss: true`. The
resulting checkpoint was a Matryoshka SAE labelled as a T-SAE. See `ERROR_LOG.md`, 2026-09-10.

This also affects the project plan. The plan excludes Priors-in-Time from the toy because the
toy lacks temporal structure. It then specifies a T-SAE on the same toy. T-SAE has the same
requirement. Either both methods should be excluded, or the toy needs a temporal axis. We chose
the second option, which also makes Priors-in-Time available on the toy.

## 2. A toy generator with temporal structure

`toy_model.TemporalTreeSampler` extends the tree over discrete timesteps. At each step, every
node either retains its previous state or redraws it. The probability of retaining the state
depends on the node's depth. We use 0.95 at depth 1 and 0.6 at depth 2. Parents therefore
change slowly and children change quickly, which matches the assumption behind T-SAE.

The tree constraints hold at every timestep. A child can only be active while its parent is
active. Mutually exclusive siblings are never active together.

**Validity of the control condition.** We also generate data with no temporal correlation,
which we call the null condition. Persistence changes the correlation between timesteps. It
does not change how often each feature fires. We verified this: expected L0 remains 1.12 in
both conditions. Across 8 seeds we measured 1.1184 ± 0.0040 without persistence and
1.1194 ± 0.0201 with persistence, against a true value of 1.12.

This property matters for interpretation. The null condition uses the same marginal
distribution, with only the temporal correlation removed. If the marginal had also changed, we
could not attribute a difference to temporal structure alone.

It also means the standard evaluation script is valid here. That script samples independent toy
activations. Because the marginal is unchanged, it evaluates these checkpoints on their
training distribution.

Measured lag-1 autocorrelation at persistence (0.95, 0.6): parents 0.949, children 0.640,
distractors 0.950.

**One design choice.** The tree contains three parents and eight childless distractors. All of
them sit at depth 1, so all of them persist at the same rate. This is deliberate. Being slow
and being a parent are different properties. A first block filled with slow childless features
is a failure mode we want to be able to detect.

## 3. Experimental design

The dictionary is split into two blocks of 4 and 16 latents, a 20/80 split. This matches the
`group_fractions` of the released Gemma checkpoint. Remaining settings: `batch_topk`, `k=2`,
learning rate 0.03, 40,000 steps, batches of 200 sequences of length 16. We ran three seeds per
condition.

| condition | temporal data | contrastive term | purpose |
| --- | --- | --- | --- |
| `tsae_temporal` | yes | yes, cosine | the method as the paper defines it |
| `tsaeip_temporal` | yes | yes, inner product | the method as the released code defines it |
| `msae_temporal` | yes | no | tests whether the term contributes anything |
| `tsae_null`, `tsaeip_null` | no | yes | tests whether the term requires temporal structure |

`msae_temporal` is the control that matters. It is a Matryoshka SAE trained on identical data
without the contrastive term. If it matches a T-SAE on the T-SAE objective, the term
contributed nothing measurable.

## 4. Results

Three seeds per condition. Reported uncertainty is the standard deviation across seeds.

### 4.1 Contrastive objective

We evaluate the first block with InfoNCE on held-out data. Lower values are better. A model
that learned nothing scores ln(256) = 5.545.

We score every model under both similarity functions. A model can only be compared against a
baseline measured the same way.

| condition | InfoNCE, cosine | InfoNCE, inner product | distinct parents in block 0 |
| --- | --- | --- | --- |
| `tsae_temporal` | 5.217 ± 0.046 | 5.435 ± 0.016 | 2.667 ± 0.471 |
| `tsaeip_temporal` | 5.189 ± 0.047 | **4.579 ± 0.044** | **3.000 ± 0.000** |
| `msae_temporal` | 5.249 ± 0.031 | 5.475 ± 0.010 | 2.333 ± 0.471 |
| `tsae_null` | 5.575 ± 0.002 | 5.546 ± 0.001 | 2.667 ± 0.471 |
| `tsaeip_null` | 5.583 ± 0.006 | 5.547 ± 0.001 | 2.667 ± 0.471 |

Parent counts use distinct features. We match each latent to a true feature by decoder cosine
similarity, using a 0.4 threshold, and then count unique matches.

### 4.2 Dictionary quality

From `scripts/eval_metrics.py` on held-out toy activations.

| condition | FVE | block-0 FVE | L0 | dead | avg max cos |
| --- | --- | --- | --- | --- | --- |
| `tsae_temporal` | 0.9839 ± 0.0041 | 0.4264 ± 0.0510 | 1.781 ± 0.300 | 0.017 | 0.0773 ± 0.0884 |
| `tsaeip_temporal` | 0.9849 ± 0.0018 | **0.4584 ± 0.0012** | 1.517 ± 0.396 | **0.000** | **0.0390 ± 0.0014** |
| `msae_temporal` | 0.9848 ± 0.0046 | 0.3967 ± 0.0469 | 1.916 ± 0.098 | 0.017 | 0.1787 ± 0.1235 |
| `tsae_null` | 0.9868 ± 0.0027 | 0.4257 ± 0.0410 | 2.251 ± 0.328 | 0.000 | 0.1044 ± 0.0694 |
| `tsaeip_null` | 0.9865 ± 0.0028 | 0.4264 ± 0.0407 | 2.010 ± 0.034 | 0.000 | 0.0797 ± 0.0730 |

`avg max cos` is the mean over latents of the highest cosine similarity to any other decoder
direction. Lower values mean less redundant features. Nesting was monotonic in all conditions.

Share of total activation assigned to block 0: `tsaeip_temporal` 75.4% ± 3.1,
`tsae_temporal` 31.5% ± 4.4, `msae_temporal` 27.3% ± 4.3.

### 4.3 Duplicate latents in block 0

We also counted cases where two block-0 latents match the same true feature. This wastes
capacity in the block the contrastive term acts on.

`tsaeip_temporal` produced no duplicates on any seed. `tsae_temporal` and `msae_temporal` each
produced one duplicate on seed 0. On that seed both models used two of their four block-0
latents on feature 8.

## 5. Results and interpretation

### 5.1 The cosine version

`tsae_temporal` scores 5.217 ± 0.046 on the cosine objective. `msae_temporal`, which never
optimised that objective, scores 5.249 ± 0.031. The ranges overlap. Parent recovery is
2.667 ± 0.471 against 2.333 ± 0.471, and the gap of 0.33 is smaller than the seed spread of
0.47. FVE and L0 also overlap.

One measure does differ in the expected direction. `avg max cos` is 0.0773 for the cosine
T-SAE and 0.1787 for the Matryoshka control, a factor of 2.3. The seed spreads are large
(0.0884 and 0.1235), so this difference is not resolved by three seeds.

We therefore state the result as follows. At n=3, we detect no reliable benefit from the cosine
contrastive term. This is not the same as showing that the term has no effect. Three seeds with
a spread of 0.47 on parent count can only detect a large effect.

Temporal structure in the data does change the score. Cosine InfoNCE falls from about 5.58 to
about 5.22 when the data becomes temporally correlated. The same shift occurs for the model
without the contrastive term. The similarity between adjacent codes follows from the data, not
from optimising for it.

### 5.2 The inner-product version

`tsaeip_temporal` scores 4.579 ± 0.044 on the inner-product objective. The same baseline scores
5.475 ± 0.010. The gap is large relative to the seed spread.

It also reaches all three parents in block 0 on every seed, with no duplicates. Its features are
the least redundant (0.0390 ± 0.0014), it leaves no dead latents, and its block-0 FVE is the
highest with very little seed variation (± 0.0012). Full reconstruction is unchanged
(0.9849 against 0.9848).

The model assigns 75.4% of total activation to block 0, compared with 27.3% for the Matryoshka
control. We report this as an observation. We did not vary the mass share independently, so we
cannot say that it causes the improvement.

Section 6.1 shows that the parent recovery reported here does not survive when frequency and
depth are separated. The remaining measures in this subsection, which concern redundancy, dead
latents and reconstruction, are not affected by that result.

### 5.3 The effect depends on temporal structure

Under the null condition the two similarity functions give indistinguishable results.
`tsae_null` and `tsaeip_null` score 5.575 and 5.583 under cosine, and 5.546 and 5.547 under
inner product. The inner-product formulation is therefore not better in general. It is better
when temporal structure is present.

## 6. Limitations

### 6.1 The first block follows firing rate, not depth

In the original tree, parents fire at 0.15, distractors at 0.05 and children at 0.03. Parents
are therefore also the most frequent features, and the two properties cannot be separated.

We ran a second tree to separate them. `configs/tree_decorrelated.json` has the same shape, with
parents at 0.04 and distractors at 0.115. Expected L0 is 1.112 against the original 1.120, so
sparsity is held roughly constant. In this tree the four most frequent features are four
distractors, and no parent is among them.

| condition | parents in block 0, original tree | parents in block 0, decorrelated tree | distractors in block 0, decorrelated |
| --- | --- | --- | --- |
| `tsaeip_temporal` | 3.000 ± 0.000 | **0.667 ± 0.471** | 3.333 ± 0.471 |
| `tsae_temporal` | 2.667 ± 0.471 | 1.333 ± 0.943 | 2.667 ± 0.943 |
| `msae_temporal` | 2.333 ± 0.471 | 1.333 ± 0.943 | 2.667 ± 0.943 |

All three parents were still learned somewhere in the dictionary, in every condition and every
seed (3.000 ± 0.000). They are placed outside the first block rather than missing from the
model. This rules out the alternative that rare parents are never learned.

The inner-product condition changes most between the two trees. It falls from 3.000 ± 0.000 to
0.667 ± 0.471, and its first block fills with distractors. The condition that looked strongest
when frequency agreed with depth looks weakest when they disagree. This is the behaviour
expected from efficient selection by firing rate.

We do not claim that the inner-product version is worse than the others at parent recovery. In
the decorrelated tree the gap between 0.667 ± 0.471 and 1.333 ± 0.943 is smaller than the seed
spread. The supported claim concerns the change within each condition across the two trees.

The redundancy result moves the same way, which we did not anticipate. On the original tree the
inner-product condition had the least redundant dictionary by a wide margin. On the decorrelated
tree it has the most redundant one.

| condition | avg max cos, original tree | avg max cos, decorrelated tree |
| --- | --- | --- |
| `tsaeip_temporal` | **0.0390 ± 0.0014** | **0.2982 ± 0.0463** |
| `tsae_temporal` | 0.0773 ± 0.0884 | 0.2049 ± 0.1289 |
| `msae_temporal` | 0.1787 ± 0.1235 | 0.2560 ± 0.0458 |

Reconstruction is unaffected and similar in all conditions, between 0.988 and 0.991. Again the
strong claim is the change within the inner-product condition, from 0.0390 ± 0.0014 to
0.2982 ± 0.0463. Its advantage in Section 4.2 therefore also depended on frequency agreeing with
depth, and not only its parent recovery.

This converges with an independent result from the metrics validation work reported on
7 September, which found that only the frequency property was recovered well across toys.

Checkpoints: `sae-training/checkpoints/decorr_toy/seed{0,1,2}_*`. Full entry in
`EXPERIMENT_LOG.md`, 2026-09-11 12:30.

### 6.2 The objective favours the model trained on it

Inner-product InfoNCE is the objective `tsaeip_temporal` optimised. Comparing it against a model
that did not optimise it is not a neutral comparison. Parent count and `avg max cos` do not
depend on the objective, and both use decoder cosine similarity, which is scale-invariant. Those
two measures are therefore not affected by this concern, but they remain subject to Section 6.1.

### 6.3 Our implementation differs from the reference

We form one contrastive pair per sequence. The reference forms a pair at every token position,
roughly sixteen times more pairs per batch at our sequence length. This may reduce the
effective strength of the contrastive term, and it may affect the cosine condition specifically.
See Section 7.

### 6.4 Limited coverage

We tested one weight for the contrastive term (0.1), one block split, one sparsity level, one
tree, one persistence schedule, and three seeds. Full reconstruction sits between 0.984 and
0.987 in all conditions, so the task offers little headroom for methods to differ.

### 6.5 Block-0 occupancy varies

For the inner-product condition, block 0 was active on 95.2%, 42.7% and 43.5% of tokens across
the three seeds. A block that is active on almost all tokens is not obviously selective. We do
not know whether the first seed reflects a distinct solution or normal variation.

### 6.6 No hierarchy claim

We have not run the hierarchy metrics on these checkpoints. Recovering three parents into
block 0 is weaker than recovering a correct parent-to-child edge set. Until the metrics run,
this is not a Tier-2 result.

These results may also not transfer to Gemma. Our dictionary has 20 latents with `k=2`. The
released Gemma checkpoint has 16,384 latents with `k=20`.

## 7. Comparison with the reference implementation

The reference is `AI4LIFE-GROUP/temporal-saes`, a fork of `saprmarks/dictionary_learning`.

| | paper | released code | released Gemma checkpoint | this repository |
| --- | --- | --- | --- | --- |
| similarity | cosine | raw inner product | inner product | selectable |
| contrastive pair | `x_t`, `x_{t-1}` | previous or random earlier token | previous | `(t, t+1)`, equivalent |
| pairs per batch | every token | every token | | one per sequence |
| block split | | `group_fractions` | `[0.2, 0.8]` | `[4, 20]` |
| block L2 weights | | `group_weights` | `[0.2, 0.8]` | equal, 0.5/0.5 |
| term weight | α | `temp_alpha` 0.1 | | 0.1 |

The first row is the substantive difference. Section 3 of the paper defines the similarity as
cosine. The released trainer computes an unnormalised inner product, and its `encode` method
does not normalise. On this toy, the two choices lead to different outcomes.

The released Gemma checkpoint uses the inner-product version. Our results will be compared
against that checkpoint, so comparison runs should use `inner_product`. The cosine setting
should be retained as the version that matches the published text.

Two rows still differ from the reference: the per-block L2 weights, and the number of pairs per
batch. Both should be reconciled before any direct comparison. A test that evaluates the
reference loss and ours on identical inputs would confirm that the implementations agree. It
has not been written.

## 8. Released Gemma checkpoint

Source: `alex-oesterling/temporal-saes`, directory `trainer_0`. Local copy at
`data/temporal-saes-gemma-l12`, which is not tracked by git.

Configuration: `gemma-2-2b`, layer 12, 16,384 latents, `k=20`, `group_sizes [3276, 13108]`,
previous-token pairing, threshold 7.30089, 200,000 steps, learning rate 3e-4. The repository
also provides `explanations.json`.

The weights load into our `TemporalSAE` with no missing and no unexpected keys.

Two items remain. The checkpoint stores a bare state dictionary alongside a separate
`config.json`, and neither of our loaders reads that layout, so it needs an adapter. Its L0 is
also unverified, because verification requires real gemma-2-2b layer-12 activations and
therefore the GPU node.

## 9. Reproduction

```bash
cd sae-training
# paper version; replace cosine with inner_product for the released code's version
uv run scripts/train_toy.py --steps 40000 --batch_size 200 --seq_len 16 \
  --activation batch_topk --k 2 --lr 0.03 --latent_sizes 4 20 --seed 0 \
  --arch tsae --contrastive_similarity cosine --persistence 0.95 0.6 \
  --save_dir checkpoints/temporal_toy/seed0_tsae_temporal

# null condition: same marginal, no temporal correlation
#   ... --persistence 0
# architecture control: same data, no contrastive term
#   ... --arch matryoshka --temporal_data on --persistence 0.95 0.6

uv run scripts/eval_metrics.py --sae_path checkpoints/temporal_toy/seed0_tsae_temporal \
  --arch tsae --dataset toy --batch_size 1000 --steps 100 \
  --output_dir eval_results/temporal_toy/seed0_tsae_temporal
```

Checkpoints: `sae-training/checkpoints/temporal_toy/seed{0,1,2}_{tsae,tsaeip,msae}_*`.
Evaluation output: `sae-training/eval_results/temporal_toy/`, including `block_fve_curve.png`,
`decoder_heatmap.png`, `feature_density.png` and `eval_metrics.json`.

## 10. Mapping to the paper

Section 2 provides a new methodology subsection on the temporal toy. Section 3 extends the
Tier-2 paragraph in `3_Methodology.tex`, which currently describes only a ten-block Matryoshka
SAE. That paragraph should record the two-block 20/80 split used for T-SAE, and the reason:
it matches the released checkpoint. The `% TODO(Ruqiya)` marker in that file refers to this.

Sections 4 to 6 provide a results subsection for `4_Experimental_Results_v2.tex`. It belongs
next to "Metrics Validation on Synthetic Hierarchy". Section 6.1 should be stated in the main
text rather than an appendix, because it limits what the result can claim.

Section 7 belongs in related work or a methodology footnote. It concerns a discrepancy between
a published method and its released code, and is not a result of ours.

Figures available: `block_fve_curve.png` and `decoder_heatmap.png` per condition. Captions are
still to be written by a human. Diagnostic plots belong in the appendix.

## 11. Open items

1. Run the hierarchy metrics on these checkpoints. This requires an adapter that reads the
   T-SAE block structure. Until then there is no Tier-2 hierarchy result for T-SAE.
2. Extend the decorrelated-tree comparison. Seed 0 placed zero parents in the first block in
   all three conditions, which suggests shared variation we have not explained. More seeds, and
   a sweep over the parent and distractor rates, would show how sharply the effect depends on
   the gap between them.
3. Reconcile the two remaining differences with the reference, and add the equivalence test.
4. Train `priors_in_time` on the temporal toy, now available through `--temporal_data on`.
5. Write the adapter for the released Gemma checkpoint and run the metrics. Requires the GPU
   node.
6. Vary the contrastive weight. The cosine result is established only at 0.1.

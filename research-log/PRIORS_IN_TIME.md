# Priors in Time: what its hierarchy actually is

## Scope

This file records what we found when reading the Priors in Time method, also called Temporal
Feature Analysis (arXiv:2511.01836, repository `eslubana-goodfire/TemporalFeatureAnalysis`).
It is a workstream file, revised as understanding changes. Dated entries go in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

**Status, 2026-09-11.** No training runs yet. We stopped before training because of the finding
below, which changes what training would be for.

## Summary

The method does not produce a parent-to-child feature hierarchy. Its hierarchical clustering
groups **tokens within a sequence** into events, not dictionary features into parents and
children. The metrics we built grade edges between features, so they have no matching input
here.

This affects the three-way comparison in Exp 3 and Exp 4. It should be settled before more work
goes into this arm.

## 1. What the architecture does

The model splits each token's activation into two parts. A **predictive** code is computed by
attention over the codes of earlier tokens. A **novel** code is computed from whatever residual
the predictive part fails to explain. Reconstruction uses the sum of both.

There is no nesting, no block structure, and no capacity bottleneck. The dictionary is flat.
This matches how the project plan already describes it, as the non-architectural method.

## 2. Where the hierarchy comes from

The paper applies hierarchical clustering from SciPy to latent codes, using cosine similarity
and a colouring threshold of 0.2. This is described in Appendices A.5 and A.6.

What gets clustered is the point that matters. From Figure 5 of the paper:

> "We confirm the alignment of predictive codes with event boundaries by running an
> off-the-shelf hierarchical clustering algorithm, finding **the token clusters** indeed
> correspond to (sub)events occurring in the story as the narrative proceeds."

Appendix A.6 clusters phrase-averaged codes within garden path sentences. Each item is a phrase
in one sentence, computed as the mean latent code over the tokens of that phrase.

So the rows being clustered are token positions or phrases, taken from one text. The result is a
segmentation of that text over time. It is not a statement about which dictionary feature is a
parent of which other dictionary feature.

## 3. The paper never claims a feature hierarchy

We searched the full text, 4,077 lines extracted from the PDF.

| term | occurrences |
| --- | --- |
| `parent` | 0 |
| `child feature` | 0 |
| `feature hierarchy` | 0 |
| `hierarchical feature` | 0 |
| `sub-feature` | 0 |

The word `hierarch` appears 31 times. Every instance we checked refers to hierarchical
clustering of token codes, or cites other work.

## 4. The released code contains no clustering

We downloaded 15 of the 16 Python files in their repository, plus the demonstration notebook.
None contains `linkage`, `dendrogram`, `fcluster`, `AgglomerativeClustering`, `KMeans`, or any
import from `scipy.cluster`.

This is consistent with Section 2. The clustering is an analysis step applied to a chosen text,
described in an appendix, and not part of the released training or analysis code.

Our repository's README records the clustering as "unimplemented". The reason is now clearer.
There is no reference implementation to port, and the thing it would produce is not a feature
hierarchy.

## 5. Our implementation against theirs

`src/sae_training/architectures/priors_in_time.py` matches their `sae/saeTemporal.py` on the
parts we checked. These are the attention loop over context codes, and the shifted context
`z_ctx` with a zero first position. They also include the projection scale computed against the
residual input, the accumulation into `z_pred`, and the subtraction of that projection from the
input. The scaling constant `lam = 1/(4*d_in)`, the tied encoder, and the N(0,1) dictionary
initialisation agree as well.

We have not checked every line, and we have not run both implementations on the same input.

## 6. What this means for the comparison

Matryoshka and T-SAE define hierarchy between features. A parent feature and a child feature are
different entries in the dictionary, and our metrics grade the edge between them. Both methods
impose this through nested blocks in the loss.

Priors in Time defines structure between **positions in a sequence**. Two tokens belong to the
same event. That is a different object, measured on different things.

Comparing them directly needs a decision about what is being compared. Three options, in
decreasing order of how much they preserve the original method:

1. **Compare only on the shared object.** Both methods produce a dictionary. We could compare
   reconstruction, sparsity, redundancy and dead features, and report no hierarchy comparison
   for Priors in Time.
2. **Define a feature hierarchy for Priors in Time ourselves**, for example by clustering
   decoder directions, which is what `scripts/eval_hierarchy.py` currently does. This would be
   our construction, not the paper's, and should be labelled that way.
3. **Test the method's own claim on its own terms**, by checking whether its predictive codes
   segment a sequence into the events we planted. The temporal toy makes this possible, because
   parent states persist across timesteps and therefore define segments with known boundaries.

Option 3 is the only one that tests what the paper actually claims. It also fits the project,
because the toy gives ground truth for segment boundaries in a way real text does not.

## 7. Open questions for the team

The project description says all three methods "claim to recover hierarchical feature
dictionaries". For Priors in Time this does not hold, on the evidence above. The framing needs
revising, or a reason we have missed.

A mentor raised a related question on 6 September: whether Priors in Time needs a separate
description in the methodology, since it derives hierarchy by clustering. The answer is stronger
than a separate description. It derives a different kind of structure, and the metrics do not
apply to it unchanged.

## 8. Training status

Training on the temporal toy works. A short run completed with
`--arch priors_in_time --temporal_data on`. Nothing is blocking the training itself. What is
blocking is the question of what to measure afterwards.

# Two-Tower Recommender: Findings

This document summarises the problems found during evaluation of the two-tower retrieval model, the fixes applied, and their practical implications.

## 1. Inflated evaluation metrics

**Problem.** The original evaluation did not mask items the session had already interacted with. A model that simply memorised recently-viewed products scored well because those products appeared as both input features and ground-truth targets. Repeat predictions — recommending an item the user just clicked — inflated MRR and Recall without providing useful discovery.

**Practical implication.** Any A/B test or product decision based on the unmasked numbers would have overstated the model's ability to surface *new* products. The model appeared far stronger than it was.

**Fix.** Split the evaluation into three slices:

| Slice | Definition | Purpose |
| --- | --- | --- |
| Has-history novel | Session has prior events; target is not in recent history | Discovery quality — the metric that matters for recommendation |
| Repeat | Target is in the session's recent history | Measures memorisation, not generalisation |
| First-event | No prior events in the session | Cold-start quality |

All reported metrics now use the has-history novel slice with seen items masked to negative infinity before ranking.

## 2. Missing baselines

**Problem.** Without non-learned baselines, there was no way to tell whether the model was learning anything beyond simple co-occurrence or popularity patterns.

**Practical implication.** A model that merely re-ranks by global popularity adds serving cost without adding value. Baselines set the floor a learned model must clear.

**Fix.** Three baselines were implemented and evaluated on the same masked, has-history novel slice:

| Baseline | MRR | Recall@10 | Recall@100 |
| --- | --- | --- | --- |
| Global popularity | 0.016 | 0.034 | 0.123 |
| Category popularity | 0.031 | 0.066 | 0.225 |
| Co-occurrence (GPU, tie-break corrected) | 0.128 | 0.278 | 0.636 |

Co-occurrence — sum the co-occurrence counts of the session's recent products, rank candidates by that sum — is the strongest non-learned baseline and the bar the model must beat.

## 3. Persistent MLP gap: user tower underperformed item-mean

**Problem.** The two-tower model produces a user embedding and an item embedding. "Item-mean" evaluation averages the item embeddings of the session's recent products and uses that average as the query — bypassing the user tower MLP entirely. Across every configuration tested, item-mean MRR exceeded user-tower MRR:

| Configuration | Item-Mean MRR | User-Tower MRR | Gap |
| --- | --- | --- | --- |
| v2 baseline (w=1.0) | 0.106 | 0.071 | 0.035 |
| v2 + duplicate masking | 0.103 | 0.075 | 0.028 |
| v3 residual + dup mask | 0.108 | 0.075 | 0.033 |
| v4 item-pool + dup mask | 0.107 | 0.075 | 0.032 |

The user tower's MLP, which has access to 8 session-level scalar features (event count, average price, cart value, etc.) and 4 sequence features, consistently made predictions *worse* than a simple average of item embeddings.

**Practical implication.** In production, the user tower generates the query embedding — item-mean is not available at serving time because it requires knowing which item to score. If the user tower is worse than item-mean, the session features are actively hurting retrieval quality. The model would serve worse results than a system that simply averaged recent item embeddings.

## 4. Hypotheses ruled out

Several explanations for the MLP gap were investigated and eliminated:

### 4a. Duplicate items in training batches

In-batch negative sampling treats every other row in the batch as a negative. When the same product appears multiple times in a batch, the model receives contradictory gradients: the item is simultaneously a positive (for one row) and a negative (for another). Masking duplicate item IDs within each batch was tested.

**Result.** MRR changed by 0.002–0.003 — within the seed-to-seed noise band of 0.002–0.006. Duplicate masking is good hygiene but did not explain the gap.

### 4b. Residual alignment (v3 architecture)

A residual connection was added so the user tower output = raw item-mean + MLP correction. At initialisation this guarantees the user tower matches item-mean exactly; the MLP then learns a correction. After training, however, the cosine similarity between the raw item-mean and the full item-tower output dropped to 0.54 (median), meaning training destroyed the alignment the residual was supposed to preserve.

**Result.** v3 MRR = 0.108 / 0.075, indistinguishable from v2. The residual did not help because the loss itself was biased (see §5).

### 4c. Gradient through pooling path (v4 architecture)

v4 replaced the user tower's learned pooling with a direct call to the item tower, so gradients from the user tower's loss flow back through the item embeddings. When this gradient path was blocked (`no_grad`), item-mean MRR dropped from 0.106 to 0.093 — confirming the gradient is important for embedding quality. But with gradients enabled, v4 matched v2 within noise.

**Result.** All architectural changes (v2, v3, v4) fell within a 0.002–0.006 MRR band. Architecture was not the problem.

### 4d. Train vs eval gap (overfitting)

The MLP gap was measured on both the training set (first 8 files, 1.36M rows) and the eval set (full, 1.97M rows):

- Train: item-mean 0.111, user-tower 0.084, gap 0.027
- Eval: item-mean 0.104, user-tower 0.077, gap 0.028

**Result.** The gap is nearly identical on train and eval. The user tower underperforms item-mean on *training data it has seen*. This is not overfitting — the objective itself is biased.

## 5. Root cause: biased in-batch softmax

**Problem.** In-batch negative sampling draws negatives from the same mini-batch. Popular items appear as negatives far more often than rare items, in proportion to their training-set frequency. The softmax loss therefore over-penalises the model for assigning high scores to popular items, because popular items appear as negatives in almost every batch.

The item-mean bypass is unaffected: it uses a fixed average of embeddings and never sees the loss. But the user tower MLP is trained end-to-end against this biased loss. It learns to suppress scores for popular items — even when the session context clearly indicates the user wants a popular product. This is why the MLP makes things *worse*: it is learning the sampling bias, not the user's intent.

**Evidence.** The over-correction check measured the median training-set count of the top-10 recommended items:

| Configuration | User-Tower Median | Item-Mean Median |
| --- | --- | --- |
| Without logQ | 223 | 1,008 |
| With logQ | 1,652 | 1,202 |

Without logQ, the user tower's top-10 recommendations had a median popularity of 223 training occurrences — far below even the item-mean baseline (1,008). The user tower was actively avoiding popular items because the biased loss taught it that popular items are bad negatives.

## 6. The fix: logQ frequency correction

**Method.** Subtract `log(frequency)` from the logits before the softmax, where frequency is the item's count in the training set. This is the standard logQ correction from [Yi et al., 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/): it cancels the over-representation of popular negatives by down-weighting their contribution to the denominator.

Critically, logQ is applied only during training. At inference time, the model scores items with plain dot products — no correction needed. The popularity estimates come exclusively from the training set.

**Results (has-history novel, masked, seed=42):**

| Metric | Without logQ | With logQ | Co-occurrence |
| --- | --- | --- | --- |
| User-tower MRR | 0.074 | **0.122** | 0.128 |
| Item-mean MRR | 0.104 | 0.110 | — |
| User-tower R@10 | 0.163 | **0.262** | 0.278 |
| User-tower R@100 | 0.450 | **0.613** | 0.636 |

The user tower now *beats* item-mean (0.122 vs 0.110), confirming that session features genuinely improve retrieval once the sampling bias is removed. The gap to co-occurrence narrowed from 0.053 to 0.006.

**All slices improved:**

| Slice | Without logQ | With logQ |
| --- | --- | --- |
| Has-history novel MRR | 0.074 | 0.122 |
| Repeat MRR | 0.521 | 0.577 |
| First-event MRR | 0.005 | 0.049 |

First-event sessions (cold start, no history) improved from near-zero (without logQ) to 0.049 (with logQ). However, global popularity — the only valid baseline for this slice (category popularity is undefined with no history) — achieves MRR=0.049 on the same rows. The model is indistinguishable from popularity on first-event rows (Δ MRR = −0.0002). The logQ fix restored the model to parity with popularity (from well below it), but the user tower adds no value beyond popularity for cold-start sessions.

**Seed consistency.** Two seeds (42 and 123) produced MRR 0.1218 and 0.1213 (delta 0.0004), far tighter than the 0.002–0.006 noise band observed across earlier experiments.

## 7. Reciprocal rank fusion with co-occurrence

The model and co-occurrence capture different signals: the model uses session-level features and learned embeddings; co-occurrence uses raw item pair counts. Their errors are partially independent, making a blend worthwhile.

**Method.** Reciprocal Rank Fusion (RRF) combines ranks rather than raw scores, avoiding the scaling problem between cosine similarities and unbounded co-occurrence sums:

```
rrf(item) = 1/(60 + rank_model) + w/(60 + rank_cooc)
```

w was tuned on the Nov 1–3 evaluation slice. First-event sessions fall back to the model alone since co-occurrence has no signal there.

**Results (has-history novel, 1,967,768 rows):**

| w | MRR | R@10 | R@100 | Δ MRR vs model |
| --- | --- | --- | --- | --- |
| Model alone | 0.122 | 0.262 | 0.613 | — |
| Co-occ alone | 0.128 | 0.278 | 0.636 | — |
| **Blend w=1.0** | **0.137** | **0.288** | **0.658** | **+0.015** |

The blend at w=1.0 (equal weighting) beats both standalone methods. The gain of +0.015 MRR is 3× the pre-set adoption threshold of 0.005.

**Serving cost.** The blend requires a co-occurrence lookup table (e.g., a Lakebase online table keyed by product ID) and a rank-merge step inside the serving pyfunc. This adds latency and infrastructure. The 0.015 MRR gain on has-history sessions justifies this cost, but the decision should account for the production serving budget.

## 8. Summary of final numbers

**Configuration: v2 architecture + duplicate masking + logQ correction + RRF blend (w=1.0)**

| Slice | Method | MRR | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| Has-history novel | Blend (model + co-occ) | 0.137 | 0.288 | 0.658 |
| First-event | Model alone | 0.049 | 0.106 | 0.277 |
| Repeat | Model alone | 0.577 | 0.783 | 0.925 |

For comparison, the strongest single baselines:

| Slice | Baseline | MRR |
| --- | --- | --- |
| Has-history novel | Co-occurrence | 0.128 |
| First-event | Global popularity | 0.0489 |

## 9. Practical implications

1. **Any two-tower model using in-batch negatives needs logQ correction.** Without it, the user tower learns the sampling distribution, not user intent. This applies to all two-tower retrieval models trained with in-batch softmax, not just this one.

2. **Item-mean is a diagnostic, not just a baseline.** When item-mean beats the user tower, the user tower's additional features are making things worse — a signal that the objective is biased or the features are noisy. Monitoring this gap during training catches the problem early.

3. **Architecture changes cannot fix a biased loss.** Residual connections, gradient routing, and pooling strategies all fell within a 0.002–0.006 noise band because they do not address the root cause. Fix the loss first; then test architectures.

4. **Train-eval gap analysis distinguishes overfitting from objective bias.** If the MLP gap exists on training data, the model is not overfitting — the loss itself is teaching the wrong thing.

5. **The blend is worth the serving cost, but only for sessions with history.** First-event sessions should use the model alone. The co-occurrence table adds infrastructure; the 0.015 MRR gain needs to be weighed against the serving latency and operational complexity.

6. **These numbers are tuned on Nov 1–3.** A held-out Nov 4–6 test pass (single run, no iteration) would confirm the numbers are not overfit to the evaluation window.
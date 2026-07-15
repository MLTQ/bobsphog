# A3 results: counterfactual page selection

## Question

A3 asks whether a query-relevant page set exists and whether a small learned
estimator can recover it without evaluation labels.

Every comparison uses the same count of equal-sized rank-4 pages. The policies
are:

- **Random:** average of 12 uniformly sampled page sets.
- **Static SVD:** pages in original singular-value priority across layers.
- **Oracle:** greedy direct-loss selection on a separate labeled calibration
  batch from the same domain.
- **Learned:** greedy predicted utility from the domain-token hidden state,
  candidate page embedding, and current resident-set embedding.

The oracle is expensive and label-aware. It establishes whether useful sparse
working sets exist; it is not deployable.

## Counterfactual supervision

For random resident sets $S$ and omitted candidates $i$, the collector measures
per-example utility:

$$
u_i(x,S)=L(f_S(x))-L(f_{S\cup\{i\}}(x)).
$$

Positive utility means fetching the page reduced answer loss. The seed-29 run
collected 9,216 training and 3,072 validation labels. About 75% were positive.

The first estimator achieved:

- validation utility correlation: 0.581;
- validation RMSE: 0.361 loss units; and
- utility-sign accuracy: 74.7%.

This is predictive signal, but substantial counterfactual error remains.

## Seed-29 equal-budget results

### Addition

| Pages | Logical residency | Base | Random | Static SVD | Oracle | Learned | Full |
|------:|------------------:|-----:|-------:|-----------:|-------:|--------:|-----:|
| 2 | 55.1% | 47.7% | 71.7% | 75.2% | 77.1% | 76.4% | 100% |
| 4 | 58.6% | 47.7% | 83.5% | 94.5% | 96.3% | 92.6% | 100% |
| 8 | 65.5% | 47.7% | 96.1% | 99.8% | 99.8% | 99.4% | 100% |

### Multiplication

| Pages | Logical residency | Base | Random | Static SVD | Oracle | Learned | Full |
|------:|------------------:|-----:|-------:|-----------:|-------:|--------:|-----:|
| 2 | 55.1% | 70.1% | 81.8% | 79.9% | 88.3% | 87.7% | 100% |
| 4 | 58.6% | 70.1% | 90.9% | 94.9% | 98.2% | 92.0% | 100% |
| 8 | 65.5% | 70.1% | 97.8% | 99.4% | 99.6% | 98.0% | 100% |

At two pages, the learned selector beats both random and static SVD on both
domains and nearly matches the oracle. At larger budgets, static SVD is already
near saturation and usually beats this first learned estimator.

The learned two-page sets are also domain dependent. Addition selects pages in
block 0 expansion and block 1 projection, while multiplication selects two
different block 1 expansion pages. There is no two-page overlap for this seed.

## Three-seed check

Across seeds 29, 30, and 31:

- oracle mean accuracy gain over random: 7.3 points;
- learned mean accuracy gain over random: 4.0 points;
- learned mean regret to oracle: 3.4 points;
- oracle beat random in 18/18 rows;
- learned beat random in 17/18 rows;
- mean validation utility correlation: 0.571;
- oracle mean gain over static SVD: 4.3 points; and
- learned mean gain over static SVD: 0.9 points.

The learned selector beat static SVD in 6/18 rows: two rows in each seed,
corresponding to the tightest query-conditioned regimes. Its overall gain over
static SVD is not yet robust enough to claim a generally superior policy.

## Interpretation

A3 establishes three useful facts:

1. **Sparse useful working sets exist.** The calibration oracle consistently
   beats random and static selection at equal page count.
2. **Counterfactual utility is learnable.** A label-free estimator recovers a
   meaningful portion of the oracle advantage on held-out queries.
3. **Query conditioning matters most when memory is tight.** At larger budgets,
   simply loading the strongest SVD components is difficult to beat because the
   toy task saturates quickly.

A3 does not establish physical memory savings, globally optimal bundles,
natural-language generalization, or a retriever that dominates static selection
at all budgets.

## Next gate

The most informative next step is an interaction-aware selector:

- collect pair or bundle utility in addition to marginal page utility;
- build a sparse co-utility graph;
- compare independent ranking with graph-expanded bundles;
- use a ranking objective aligned with top-$k$ selection rather than only
  pointwise regression; and
- make the synthetic task harder enough that four- and eight-page budgets do not
  immediately saturate.

The current experiment still runs comfortably on the Mac. The 4090 becomes
worthwhile for wider multi-seed/hyperparameter sweeps, a larger non-saturating
toy model, or the first physical CUDA pager.

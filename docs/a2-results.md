# A2 results: variable structured page dropout

## Experiment

The first A2 run used seed 19 on CPU with:

- a two-layer, 32-channel dense teacher;
- addition and multiplication modulo ten over a shared vocabulary;
- 400 dense-teacher steps and 400 paged-student steps;
- batch size 128;
- FFN base rank 4 and page rank 4;
- sampled page dropout rates of 0.25, 0.5, 0.75, and 0.9; and
- partial-path task plus distillation loss and full-path retention KL.

The dense teacher reached 100% held-out answer accuracy on both domains. Exact
SVD conversion preserved its full-residency function.

## Anti-collapse control

Allowing every student parameter to train made the base-only path reach 100% on
both tasks. The resident skeleton had absorbed the complete toy task, so this
was capacity collapse rather than evidence for paging.

The corrected default freezes the converted resident skeleton and optimizes only
the residual page factors. The base-only row must therefore remain unchanged,
and improvements at partial budgets must come from pages.

## Sampled-mask results

These rows average fresh independent page masks. Residency is logical: all page
tensors are still physically allocated in this phase.

| Dropout | Mean residency | Addition before | Addition after | Multiplication before | Multiplication after |
|--------:|---------------:|----------------:|---------------:|----------------------:|---------------------:|
| 1.00 | 51.7% | 43.4% | 43.4% | 67.2% | 67.2% |
| 0.90 | 59.4% | 49.5% | 74.4% | 71.3% | 88.2% |
| 0.75 | 64.6% | 58.8% | 83.9% | 80.0% | 95.3% |
| 0.50 | 75.8% | 82.1% | 95.6% | 84.2% | 97.9% |
| 0.25 | 88.8% | 90.5% | 98.7% | 88.8% | 99.3% |
| 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% |

Variable structured page dropout therefore improves the expected
quality/residency curve without moving the frozen base-only point. The 0.2-point
full-path multiplication regression is small but nonzero and should be tracked
across seeds.

## Static-prefix results

Ordered SVD prefixes also improved at the most useful intermediate budgets. At
65.5% residency, addition rose from 92.3% to 96.2% and multiplication from
96.8% to 99.4%. Prefixes are secondary after page training because learned
factors need not retain their original singular-value priority order.

## Page specialization

Full-model single-page ablations produced:

- top-quartile page Jaccard overlap: 0.556;
- utility-vector cosine similarity: 0.709; and
- mean absolute cross-domain utility gap: 0.00544 loss units.

The domains use measurably different page mixtures, but the separation is not
strong enough to call the pages semantic experts. Random dropout rewards
redundancy and robustness; it does not train a query-conditioned selector.

## Conclusion and next gate

A2 passes its mechanical objective: residual pages can be trained to improve
multiple partial-residency budgets while a frozen skeleton prevents trivial
capacity collapse. It does not yet prove query-specific working sets or physical
memory savings.

The next stage should build oracle page/bundle selection from causal ablation
labels, compare it against random and static prefixes at equal bytes, and then
train the first counterfactual utility estimator. The 4090 is not required for
that implementation; it becomes useful for multi-seed sweeps, larger task/model
variants, or physical CUDA paging measurements.

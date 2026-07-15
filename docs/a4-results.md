# A4 results: sparse relationship graphs

## Question

A4 asks whether pairwise relationships among pages improve bundle construction
beyond independent page scores.

The experiment uses a harder compositional task over shared modulus-8 tokens:

$$
(a+b)c\bmod 8
\qquad\text{versus}\qquad
ab+c\bmod 8.
$$

The dense teacher reaches about 99.6% on both domains. After page-dropout
training, the full paged path retains roughly 98% accuracy while four- and
eight-page budgets remain below saturation.

## Relationship definition

For pages $i$ and $j$, measured from the base-only skeleton:

$$
\operatorname{interaction}(i,j)
=u(\{i,j\})-u(\{i\})-u(\{j\}).
$$

Positive values indicate superadditive complementarity. Negative values mean
the pages are redundant or substitutable: loading both provides less benefit
than their singleton utilities suggest.

The graph evaluates pairs among the twelve strongest singleton candidates and
retains each page's four strongest absolute interactions. The resulting domain
graphs contain about 34 edges among 44 pages, or roughly 3.6% of all possible
undirected edges.

## Why signed edges matter

The first implementation retained only positive synergy. One domain produced no
edges, and graph guidance slightly hurt the learned selector.

After retaining signed interactions, every strongest retained edge across three
seeds and both domains was negative. In this experiment the useful relationship
index is therefore a **redundancy map**, not a collaboration map. It helps avoid
spending a limited budget on pages that repair the same residual computation.

## Seed-41 equal-budget results

The columns are answer accuracy. All non-control policies use the same count of
equal-sized pages.

### Add then multiply

| Pages | Residency | Random | Static SVD | Singleton | Graph | Learned | Learned + graph | Oracle | Full |
|------:|----------:|-------:|-----------:|----------:|------:|--------:|----------------:|-------:|-----:|
| 2 | 50.2% | 62.9% | 71.1% | 68.2% | 74.2% | 70.6% | 70.6% | 74.2% | 98.4% |
| 4 | 52.6% | 67.4% | 79.9% | 78.6% | 80.7% | 78.9% | 78.9% | 85.2% | 98.4% |
| 8 | 57.3% | 79.5% | 93.0% | 69.3% | 85.4% | 85.4% | 85.9% | 92.7% | 98.4% |

### Multiply then add

| Pages | Residency | Random | Static SVD | Singleton | Graph | Learned | Learned + graph | Oracle | Full |
|------:|----------:|-------:|-----------:|----------:|------:|--------:|----------------:|-------:|-----:|
| 2 | 50.2% | 51.6% | 61.5% | 56.0% | 61.2% | 61.5% | 61.5% | 61.2% | 97.9% |
| 4 | 52.6% | 56.6% | 74.0% | 61.2% | 69.5% | 75.3% | 77.9% | 75.8% | 97.9% |
| 8 | 57.3% | 68.6% | 90.4% | 63.5% | 82.6% | 81.2% | 85.7% | 92.4% | 97.9% |

For this seed, the calibrated graph beats singleton ranking in all six rows by
9.5 points on average. Adding it to the learned policy helps primarily after a
resident set contains enough pages for redundancy edges to become informative.

## Three-seed check

Across seeds 41, 42, and 43:

- calibrated graph mean gain over singleton ranking: 5.6 points;
- calibrated graph beat singleton ranking in 15/18 rows;
- learned-plus-graph mean gain over learned selection: 0.8 points;
- learned-plus-graph beat learned selection in 7/18 rows; and
- every retained strongest edge in all six domain graphs was negative.

The relationship signal is therefore robust for repairing naive independent
ranking. Its incremental value to the A3 estimator is positive but small because
that estimator already sees a resident-set embedding and can learn some
substitution implicitly.

## Static SVD remains important

In seed 41, the graph policy averages 2.7 points below static SVD, and the
learned-plus-graph policy averages 1.6 points below it. The graph helps most at
tight budgets and when singleton ranking selects redundant pages; it does not
yet dominate a strong static low-rank layout at larger budgets.

This prevents the A4 result from being overstated. We have evidence that sparse
page relationships are useful, not that the current graph is the best selector.

## Conclusion

A4 validates the relationship-index concept in a concrete form:

> A small sparse graph of signed page interactions can improve equal-budget
> bundle construction by steering selection away from redundant computation.

The result remains logical and calibration-driven. It does not yet include
physical transfers, an automatically chosen graph for arbitrary prompts, or
learned bundle utilities beyond pairs.

## Next gate: physical paging

The logical stack now contains:

- graceful partial-residency training;
- query-conditioned marginal utility prediction; and
- a sparse relationship index for bundle correction.

The next highest-value experiment is a real CUDA page cache with CPU-resident
page factors, asynchronous transfer, fixed GPU cache capacity, and measurements
of peak VRAM, transferred bytes, prefill latency, and cache hit rate. This is the
point where the 4090 Linux box becomes materially useful.

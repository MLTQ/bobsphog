# `test_glm_b22_benchmark.py`

## Purpose

Checks the pure, CPU-safe parts of the GLM B2.2 benchmark without loading the
1.5 TB checkpoint.

## Coverage

- deterministic one-per-subject MMLU sampling from official-style CSV files;
- seeded subject subsampling when a bounded cross-section is requested;
- direct-choice prompt formatting;
- deterministic equal-layer prefill-frequency bundle selection; and
- Wilson confidence intervals for the small stratified accuracy sample.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Benchmark reproducibility | Same seed and data select identical rows | Process-random hashing |
| B2.2 policy | Quotas differ by at most one page | Unconstrained global ranking |
| Accuracy report | Small-sample uncertainty is explicit | Point estimate without interval |

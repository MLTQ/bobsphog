# `test_b1_throughput.py`

## Purpose

Validates the pure latency aggregation used by the pretrained token-rate
benchmark.

## Components

### Summary test

- **Does**: Checks total time, token rate, median, and nearest-rank p95.

### Validation test

- **Does**: Rejects empty and nonpositive latency samples.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Throughput report | Decode samples are positive synchronized durations | Accepting zero or missing samples |
| Mac CI | Tests require no pretrained dependencies | Eager model loading |

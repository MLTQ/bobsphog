# `b1_throughput.py`

## Purpose

Measures the user-visible speed cost of exact expert paging. It separates model
load, prompt prefill/time-to-first-token, synchronized steady decode, and total
output throughput for full-resident and paged Qwen paths.

## Components

### `B1ThroughputConfig`

- **Does**: Selects reference, paged, or same-device comparison mode plus the
  prompt, output length, device, and expert-cache budget.

### `summarize_decode_latencies`

- **Does**: Reports aggregate token rate and mean, median, p95, minimum, and
  maximum synchronized decode latency.

### `_benchmark_path`

- **Does**: Times one prefill and a fixed-length manual greedy decode using the
  real KV/recurrent state.
- **Rationale**: Manual decoding separates TTFT from steady decode and ignores
  EOS so both paths execute exactly the requested number of steps.

### `run_b1_throughput`

- **Does**: Loads the requested model paths, forces the paged path along the
  reference token sequence in comparison mode, and computes slowdown factors.
- **Interacts with**: `load_reference_qwen`, `load_paged_qwen`, and
  `CudaExpertCache`.

### `main`

- **Does**: Exposes the benchmark as `bobsphog-b1-throughput`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| TTFT metric | First forward contains the whole formatted prompt | Pre-cached prompt state |
| Decode rate | Every step synchronizes before timing completes | Asynchronous timing without events |
| Same-device ratio | Paged path consumes the reference token path | Independent divergent generations |
| Cache accounting | Prefill and decode deltas are reported separately | Snapshot only after the full run |

## Notes

The benchmark excludes model-load time from token-rate calculations but reports
load time independently. A single prompt is a controlled systems measurement,
not a workload-wide throughput distribution.

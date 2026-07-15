# `glm_b15.py`

## Purpose

Runs the smallest meaningful GLM-5.2 demand-paging feasibility experiment on a
Strix Halo host. It loads the resident scaffold, executes one exact token through
file-backed routed experts, and reports measured allocation, I/O, latency, and
next-token candidates.

## Components

### `GlmB15Config`

- **Does**: Selects checkpoint, HIP/CUDA device, exact expert-cache budget,
  one-token seed text, and candidate count.

### `run_glm_b15`

- **Does**: Builds the file-backed source and bounded cache, loads the GLM
  scaffold, validates a single-token input, runs one synchronized forward, and
  emits JSON-compatible feasibility evidence.
- **Interacts with**: `load_paged_glm` in `glm_model.py`,
  `MappedGlmExpertSource` in `glm_checkpoint.py`, and `CudaExpertCache`.

### `main`

- **Does**: Exposes the experiment as `bobsphog-glm-b15`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Memory probe | Cache holds at least one layer's eight selected experts | Smaller cache |
| Routing measurement | Input encodes to exactly one token | Multi-token prefill |
| Output | Standard output is one JSON document | Logging to stdout |
| Claim scope | Result is paged feasibility, not full-reference parity | Presenting it as a same-device control |

## Notes

One GLM-5.2 expert is 72 MiB, and the 75 sparse layers select 600 expert pages
for a one-token forward. A full-resident B1.5 reference is impossible on the
128 GiB host, so validation starts with end-to-end execution and coherent output.

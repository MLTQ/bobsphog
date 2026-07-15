# `b1_capability.py`

## Purpose

Measures whether demand paging changes Qwen3.6 behavior relative to a complete
same-device reference model. It compares memory peaks, generated capability
probes, teacher-forced token decisions, and full next-token distributions.

## Components

### `CapabilityProbe` / `DEFAULT_PROBES`

- **Does**: Defines compact instruction, factual, arithmetic, translation,
  science, and code probes with simple expected-answer checks.

### `B1CapabilityConfig`

- **Does**: Selects the checkpoint, CUDA/HIP device, expert-cache budget,
  generation length, probe count, optional true autoregressive verification,
  and top-k comparison width.

### `compare_teacher_forced_logits`

- **Does**: Computes teacher KL, absolute logit error, top-1 agreement,
  raw-reference top-1 agreement, actual generated-token agreement, and top-k
  overlap for aligned traces.

### `run_b1_capability`

- **Does**: Generates traces with a full-resident text model, unloads it,
  replays those sequences through the paged model, optionally regenerates them
  through the real paged KV/decode path, and reports capability and peak-memory
  parity on the same GPU/runtime.
- **Interacts with**: `load_reference_qwen` in `reference_model.py`,
  `load_paged_qwen` in `moe_model.py`, and `CudaExpertCache`.

### `main`

- **Does**: Exposes the experiment as `bobsphog-b1-capability`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Same-device control | Reference and paged paths use one runtime and dtype | Cross-device comparison |
| Greedy trace | Reference generation is deterministic | Sampling or stochastic processors |
| Teacher forcing | Last `G` logits predict the `G` generated tokens | Different `logits_to_keep` semantics |
| Decode verification | Requested paged generations start from identical prompt tokens | Changing chat formatting between paths |
| Cache budget | A single layer can protect at most 256 expert pages | Capacity below one-layer union |
| Capability claim | Expected-answer checks accompany distribution parity | Reporting memory alone |

## Notes

Teacher-forced agreement is a capability-preservation checksum, not a broad
benchmark. True paged autoregressive verification additionally exercises KV
and recurrent state plus decode-time cache churn.

The raw-logit top-1 metric can differ from the token emitted by `generate` when
the model's generation configuration applies a logits processor. Generated-
token agreement and true autoregressive sequence agreement are therefore the
authoritative behavioral checks.

# `glm_model.py`

## Purpose

Constructs the GLM-5.2 causal language model without allocating its routed
expert reservoir. The text scaffold is built on the meta device, exact expert
collections are replaced by cache-backed shells, and only resident tensors are
streamed to the accelerator.

## Components

### `checkpoint_key_to_glm_key`

- **Does**: Retains resident language-model tensors while excluding routed
  experts and the checkpoint's extra MTP layer.

### `PagedGlmExperts`

- **Does**: Schedules the unique experts selected by GLM's native router and
  delegates exact weighted execution to `CudaExpertCache`.

### `load_paged_glm`

- **Does**: Builds `GlmMoeDsaForCausalLM` on meta, replaces all sparse expert
  modules, streams exact scaffold tensors to CUDA/HIP, and rejects incomplete
  materialization.
- **Interacts with**: `MappedGlmExpertSource` in `glm_checkpoint.py` and the
  Transformers 5.12 GLM-MoE-DSA implementation.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| GLM decoder | Replacement experts keep the three-argument forward contract | Transformers expert API changes |
| Memory bound | No `.mlp.experts.*` checkpoint tensor becomes a model parameter | Loading routed experts into the scaffold |
| Completeness | Every remaining model parameter leaves meta | Silently accepting missing resident tensors |
| Checkpoint | Layer 78 is MTP-only and excluded from the 78-layer causal model | Different MTP layout |

## Notes

The adapter intentionally uses the eager expert accumulation order for initial
correctness. Fused expert kernels and prefetch are later performance work.

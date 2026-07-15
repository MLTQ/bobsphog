# `moe_model.py`

## Purpose

Constructs the Qwen3.6 text model without allocating routed expert reservoirs.
The architecture is created on the meta device, packed expert modules are
replaced by cache-backed shells, and only language-model scaffold tensors are
loaded from the sharded checkpoint.

## Components

### `checkpoint_key_to_text_key`

- **Does**: Maps full vision-language checkpoint names into the text-only
  `Qwen3_5MoeForCausalLM` namespace and rejects vision, MTP, and routed-expert
  tensors.

### `PagedMoeExperts`

- **Does**: Receives the original router's top-k indices, schedules the unique
  `(layer, expert)` pages, and delegates exact weighted execution to
  `CudaExpertCache`.

### `load_paged_qwen`

- **Does**: Instantiates Qwen on the meta device, installs provider-backed
  expert shells, streams non-expert tensors from each safetensor shard directly
  into CUDA parameters while preserving each checkpoint tensor's dtype, and
  verifies no meta parameters remain.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Qwen decoder | Replacement experts retain the original three-argument forward | Different Transformers MoE API |
| Memory bound | Packed `.mlp.experts.*` tensors are never loaded | Passing expert tensors to the scaffold loader |
| Text-only model | Vision and MTP keys are ignored | Loading the conditional-generation wrapper |
| Completeness | Every remaining parameter leaves the meta device | Silently accepting missing scaffold weights |

## Notes

Transformers, Accelerate, and safetensors are imported lazily so Mac/CPU unit
tests can validate key mapping without installing the pretrained stack.

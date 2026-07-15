# `test_glm_model.py`

## Purpose

Checks the GLM checkpoint-to-runtime key filter without importing Transformers
or allocating the model.

## Components

### `test_checkpoint_key_to_glm_key_filters_nonresident_tensors`

- **Does**: Retains scaffold and output-head keys while excluding routed experts,
  MTP-only layer 78, and unrelated entries.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| GLM loader | Routed experts never enter scaffold loading | Filter behavior |
| Causal model | MTP-only layer 78 is not a target | Checkpoint architecture changes |

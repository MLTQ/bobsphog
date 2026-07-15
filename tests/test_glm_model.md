# `test_glm_model.py`

## Purpose

Checks the GLM checkpoint-to-runtime key filter without importing Transformers
or allocating the model.

## Components

### `test_checkpoint_key_to_glm_key_filters_nonresident_tensors`

- **Does**: Retains scaffold and output-head keys while excluding routed experts,
  MTP-only layer 78, and unrelated entries.

### Optional route observer test

- **Does**: Verifies the paged GLM expert adapter passes native router indices
  to a cache observer before ordinary scheduling and execution.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| GLM loader | Routed experts never enter scaffold loading | Filter behavior |
| Causal model | MTP-only layer 78 is not a target | Checkpoint architecture changes |
| B2.2 recorder | Observer sees the unmodified route tensor | Reordered or copied indices |

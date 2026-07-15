# `test_moe_model.py`

## Purpose

Validates full-checkpoint to text-scaffold name mapping without importing the
optional pretrained runtime.

## Components

### Checkpoint key mapping test

- **Does**: Accepts language scaffold and output-head tensors while rejecting
  routed experts, vision tensors, MTP tensors, and unknown keys.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Mac CI | Mapping test has no Transformers dependency | Eager pretrained imports |
| Scaffold loader | Expert reservoirs can never enter the accepted key set | Mapping `.mlp.experts.*` |


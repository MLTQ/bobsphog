# `test_reference_model.py`

## Purpose

Validates the full-reference checkpoint mapping without importing the optional
pretrained runtime or allocating model weights.

## Components

### Reference key-mapping test

- **Does**: Retains language scaffold, routed experts, and the output head while
  rejecting vision, MTP, and unknown tensors.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Mac CI | Mapping remains dependency-light | Eager Transformers imports |
| Full control | Expert tensors map into the text model | Filtering routed experts |

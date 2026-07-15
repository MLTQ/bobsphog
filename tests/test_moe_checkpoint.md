# `test_moe_checkpoint.py`

## Purpose

Validates Qwen3.6 metadata extraction and packed expert-to-shard resolution
without downloading or importing the pretrained checkpoint.

## Components

### Metadata and index test

- **Does**: Creates a miniature Hugging Face-style config/index, verifies the
  derived 6 MiB expert size, and resolves both layer expert tensors.

### Bounds test

- **Does**: Checks invalid layer naming and source requests fail before any
  safetensor access.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CPU/Mac CI | Metadata tests need no Transformers or safetensors package | Eager optional imports |
| Stage B loader | Official packed tensor names remain explicit | Silent fallback naming |


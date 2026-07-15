# `test_glm_checkpoint.py`

## Purpose

Validates GLM-5.2 metadata parsing, exact expert naming, 72 MiB page accounting,
and gate/up packing without requiring the large checkpoint or an accelerator.

## Components

### GLM metadata tests

- **Does**: Confirms sparse-layer discovery, expert byte counts, and shard paths.

### Expert packing test

- **Does**: Replaces safetensor reads with tiny tensors and verifies the cache's
  required gate-then-up layout plus source metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| GLM source | Gate precedes up in `ExpertWeights.gate_up` | Packing order |
| Hardware sizing | BF16 page bytes follow `3 * H * I * 2` | Accounting formula |
| Checkpoint adapter | Expert tensor names match the official index | Naming changes |

# `test_expert_cache.py`

## Purpose

Checks that the CUDA expert cache reproduces a direct SwiGLU expert computation,
handles router-weighted token accumulation, and reuses warm expert pages.

## Components

### CUDA expert cache test

- **Does**: Uses a tiny in-memory source, schedules two expert keys, compares
  cached and direct routed outputs, and verifies the next request is all hits.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CPU-only CI | Test skips cleanly when CUDA is absent | Removing CUDA guard |
| Qwen adapter | Routed accumulation matches reference semantics | Different expert weighting order |


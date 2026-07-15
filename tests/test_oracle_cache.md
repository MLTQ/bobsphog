# `test_oracle_cache.py`

## Purpose

Protects the deterministic future-use policy independently of CUDA and the
large Qwen checkpoint.

## Coverage

- Furthest-next-use victim selection.
- Immediate failure when live routing diverges from the oracle trace.
- Trace completion and overrun detection.


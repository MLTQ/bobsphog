# `test_b21.py`

## Purpose

Verifies B2.1 trace reconstruction and pipeline arithmetic without requiring an
accelerator or pretrained checkpoint.

## Coverage

- Layer-aware expert-key reconstruction from B2 JSON.
- Deduplicated per-layer decode-union construction.
- Rejection of token/trace misalignment.
- Serial, compute-only, and ideal one-token-overlap timing.

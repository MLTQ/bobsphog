# `test_b2.py`

## Purpose

Verifies B2's trace-only working-set analysis without requiring an accelerator
or pretrained checkpoint.

## Coverage

- Cumulative union and new-page growth.
- Whole-model and per-layer previous-token overlap.
- Malformed layer traces and the one-forward edge case.


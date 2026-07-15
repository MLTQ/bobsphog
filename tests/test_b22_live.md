# `test_b22_live.py`

## Purpose

Protects the CPU-testable corpus-to-live-route boundary for B2.2.

## Coverage

- reconstructs exact layer/expert route keys and forced token IDs;
- converts a backend-native recorded route into predictor evaluation tensors;
- preserves the decode-forward count; and
- rejects train prompts so live evidence cannot be reported as held-out.

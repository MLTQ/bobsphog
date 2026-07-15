# `test_b23_policy.py`

## Purpose

Protects the target-isolation, route conversion, physical-cache replay, and
calibration primitives used by the B2.3 adaptive policy.

## Coverage

- preserves ordered prefill and decode groups;
- verifies confidence features are unchanged when query decode counts change;
- recovers a simple linear target with the ridge calibrator; and
- confirms a correctly selected pinned page can reduce warm-cache decode misses.

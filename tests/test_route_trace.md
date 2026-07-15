# `test_route_trace.py`

## Purpose

Verifies route-hook ordering and expert-set normalization on CPU without Qwen or
an accelerator.

## Coverage

- One hook call per layer.
- Sorted unique expert IDs with layer-qualified keys.
- Explicit hook cleanup.


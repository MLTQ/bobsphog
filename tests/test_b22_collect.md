# `test_b22_collect.py`

## Purpose

Protects the prompt-corpus schema and split labels used by the expensive B2.2
trace collection run.

## Coverage

- Valid stratified records retain ordering and split labels.
- Duplicate prompt IDs fail before model loading.


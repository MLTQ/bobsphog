# `test_physical_cache.py`

## Purpose

Validates exact CUDA execution through asynchronously scheduled pinned CPU
sources and confirms that a second preparation reuses cached pages.

## Components

### CUDA cache integration test

- **Does**: Schedules a cold plan without a host wait, compares resident and
  physical logits after device-event synchronization, then verifies warm hits,
  exact cache bytes, and CPU source placement.
- **Interacts with**: Factorized toy model, catalog, and `PhysicalPageCache`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A5 correctness | Physical factor copies reproduce resident output | Approximate cache representation |
| CPU-only CI | Test skips cleanly without CUDA | Removing CUDA guard |

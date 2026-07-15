# `test_paging.py`

## Purpose

Verifies static budget and structured-dropout page-plan semantics.

## Components

### Plan tests

- **Does**: Checks prefix capping, deterministic random masks, and default full
  versus base behavior.
- **Interacts with**: `PagePlan` in `src/bobsphog/paging.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Training experiments | Seeded masks are reproducible | Random-number strategy changes |
| Model execution | Full/base defaults resolve every unlisted layer | Default policy changes |

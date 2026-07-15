# `__init__.py`

## Purpose

Defines the small public API for toy experiments without exposing internal
module layout to callers.

## Components

### Public exports

- **Does**: Exposes the dense teacher, paged student, conversion function, page
  controls, and synthetic A2 task.
- **Interacts with**: `conversion.py`, `dense_model.py`, `decomposition.py`,
  `paging.py`, `model.py`, and `synthetic.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Experiment code | Named imports remain stable | Renaming or removing an export |
| Tests | Package imports work from `src` | Changing package structure |

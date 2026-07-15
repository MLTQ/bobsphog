# `test_objectives.py`

## Purpose

Protects masking and KL-direction behavior for A2 losses and metrics.

## Components

### Objective tests

- **Does**: Confirms ignored positions do not affect metrics and identical
  teacher/student distributions have zero KL.
- **Interacts with**: Functions in `objectives.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A2 optimization | Mask reduction averages selected answers | Reduction changes |
| Distillation | KL is teacher-to-student | Argument/direction changes |

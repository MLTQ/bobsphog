# `training.py`

## Purpose

Contains reusable A2 optimization loops for a dense task teacher and a paged
student trained under variable structured page dropout.

## Components

### `OptimizationConfig`

- **Does**: Defines optimizer steps, batch size, learning rate, decay, and
  clipping.

### `TrainingSummary`

- **Does**: Reports initial, trailing-window final, overall mean loss, and the
  number of optimized scalar parameters.

### `train_dense_teacher`

- **Does**: Optimizes the dense model on masked arithmetic answer targets.
- **Interacts with**: `DenseToyTransformer`, `TwoDomainArithmetic`, and masked CE.

### `train_multi_budget_student`

- **Does**: Samples a page-dropout budget each batch, matches ground truth and
  teacher logits on that partial path, and anchors the full path to the teacher.
- **Interacts with**: `PagePlan.random_dropout` and masked KL in `objectives.py`.
- **Rationale**: By default only residual pages are trainable, preventing the
  resident skeleton from absorbing the complete toy task.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `a2.py` | Both loops return `TrainingSummary` | Return schema changes |
| Student training | Teacher stays frozen | Enabling teacher gradients |
| Multi-budget objective | Partial path receives task + KL; full path receives retention KL | Loss composition changes |
| A2 anti-collapse control | `freeze_resident=True` trains page factors only | Unfreezing skeleton parameters |

## Notes

Only one random partial budget is sampled per batch to keep the CPU experiment
small. Explicit multiple partial passes can be added when CUDA sweeps begin.

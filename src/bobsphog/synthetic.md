# `synthetic.py`

## Purpose

Generates the controlled A2 language-model task: addition and multiplication
modulo ten using the same number vocabulary and distinct prompt domain markers.

## Components

### `SyntheticBatch`

- **Does**: Carries causal inputs, shifted targets, answer-position masks, and
  per-example domain IDs.
- **Interacts with**: Objectives and A2 training in `a2.py`.

### `TwoDomainArithmetic`

- **Does**: Samples balanced or domain-specific arithmetic clauses.
- **Rationale**: Shared surface form makes page specialization about operation,
  while exact generated answers give an unambiguous evaluation signal.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `a2.py` | Domain ID 0 is addition and 1 is multiplication | Domain remapping |
| `objectives.py` callers | `answer_mask` marks shifted result targets | Sequence layout changes |
| Model config | `VOCAB_SIZE` covers every generated token | Token allocation changes |

## Notes

Only result positions contribute to task losses and metrics. Random operands are
context, not prediction targets.

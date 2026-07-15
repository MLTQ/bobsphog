# `test_synthetic.py`

## Purpose

Checks the exact token and answer-mask semantics of the two-domain task.

## Components

### Arithmetic generation tests

- **Does**: Recomputes addition answers, verifies result positions, and checks
  multiplication token bounds/domain IDs.
- **Interacts with**: `TwoDomainArithmetic` in `synthetic.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A2 training | Masks select result tokens after causal shifting | Clause layout changes |
| Model config | Generated IDs stay below `VOCAB_SIZE` | Vocabulary changes |

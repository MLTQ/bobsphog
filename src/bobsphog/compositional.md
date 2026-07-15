# `compositional.py`

## Purpose

Defines the harder A4 task with two order-sensitive arithmetic compositions over
the same vocabulary and modulus.

## Components

### `CompositionalArithmetic`

- **Does**: Generates $(a+b)c\bmod n$ and $ab+c\bmod n$ clauses with explicit
  domain markers and answer masks.
- **Interacts with**: Existing teacher/student training and A4 evaluation.
- **Rationale**: Three operands and operation order reduce early budget
  saturation while keeping exact, cheap supervision.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A4 model config | `vocab_size` covers all generated IDs | Token allocation changes |
| A4 graph routing | Domain ID 0 is add-then-multiply, 1 is multiply-then-add | Domain remapping |
| Objective code | `answer_mask` marks shifted result targets | Clause layout changes |

## Notes

The task tests learned computation over familiar operand combinations, not
out-of-distribution mathematical generalization.

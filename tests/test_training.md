# `test_training.py`

## Purpose

Provides a fast integration check that both A2 optimization phases execute and
return finite summaries.

## Components

### Training-loop integration test

- **Does**: Runs two dense-teacher steps, conversion, and two multi-budget
  student steps on a minimal model.
- **Interacts with**: Training, conversion, synthetic task, and both models.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| A2 CLI | Core training paths execute on CPU | Training signatures or device behavior |

# `utility_data.py`

## Purpose

Builds A3 supervision by executing the model before and after adding a candidate
page to a sampled resident set.

## Components

### `UtilityExamples`

- **Does**: Stores query states, candidate IDs, resident masks, and measured
  per-example marginal utilities.

### `collect_utility_examples`

- **Does**: Samples domains/resident sets and labels omitted pages with
  $L(S)-L(S\cup\{i\})$.
- **Interacts with**: `ToyTransformer.hidden_states`, `PageCatalog`, and
  per-example masked loss.
- **Rationale**: The hidden state at position one contains the causal domain
  prefix without using answer labels as retriever input.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `retriever.py` | Positive targets mean adding the page reduced loss | Utility sign changes |
| A3 split | Examples are returned on CPU in aligned row order | Device or ordering changes |
| Query policy | Query is final normalized hidden state at the domain token | Query position/representation changes |
| A3/A4 tasks | Caller may provide exactly two distinct domain names | Domain cycling changes |

## Notes

Labels are page-marginal, not bundle utilities. Random resident sets expose some
interactions, but explicit pair/bundle labels remain future work.

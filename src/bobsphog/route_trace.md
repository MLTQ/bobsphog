# `route_trace.py`

## Purpose

Captures the exact per-layer routed-expert sets selected by a full-resident Qwen
forward without altering router inputs, outputs, or expert execution.

## Components

### `ExpertRouteRecorder`

- **Does**: Registers a forward-pre-hook on each layer's expert collection,
  records the top-k index tensors for one model forward, and returns sorted
  `(layer, expert)` groups.
- **Interacts with**: Full-resident Qwen models loaded by `reference_model.py`
  and the B2.2 corpus collector.
- **Rationale**: Raw GPU indices are retained until the forward ends so capture
  does not introduce one device synchronization per layer.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| B2.2 collector | Exactly one expert-module call per layer per forward | Expert batching across layers |
| Trace consumers | Groups are ordered by layer and experts are sorted | Unstable group ordering |
| Capability control | Hooks never replace arguments or outputs | Mutating hook return values |


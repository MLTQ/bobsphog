# `test_cache_simulation.py`

## Purpose

Protects the cache-policy analysis used by B2 from optimistic or structurally
invalid miss counts.

## Coverage

- Demonstrates a trace where future-aware Belady replacement improves on LRU.
- Rejects capacities that cannot hold one atomic layer request.
- Confirms duplicate keys within a layer count as one physical request.
- Replays the post-schedule expert-touch order that determines later LRU victims.
- Verifies pinned pages survive prefill and that prefill warmth contributes to
  decode hits.
- Rejects pinned bundles that leave insufficient atomic-group capacity.
- Distinguishes eager bundle prefetch from lazy first-demand retention.
- Models pin admission either before prefill or only at the decode boundary.

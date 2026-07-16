# Fixed-offset expert page store

## Intent

`page_store.py` replaces repeated safetensors slice discovery with one exact,
contiguous BF16 expert file. It changes storage layout only: every loaded expert
has the same tensors and numerical behavior as the checkpoint source.

The B2.4 hypothesis is that stable offsets and larger sequential storage locality
will reduce source overhead enough for B2.3's lower fault count to become a real
throughput improvement.

## On-disk contract

A store directory contains:

- `metadata.json`: versioned model geometry and source-checkpoint hashes.
- `experts.bf16`: complete data, published only after a successful build.
- `experts.bf16.partial`: resumable build output, always page aligned.

Pages are layer-major and then expert-major:

`page_index = layer * num_experts + expert`

Each page is raw BF16 containing flattened `gate_up_proj` followed by flattened
`down_proj`. Its size is

`3 * hidden_size * moe_intermediate_size * 2 bytes`.

For Qwen3.6-35B-A3B this is 6 MiB per page, 10,240 pages, and 60 GiB total.

## Builder contract

`build_page_store()` reads one packed layer at a time and appends complete expert
pages. It resumes from a page-aligned partial file and fsyncs after every layer.
Metadata and the final filename are published only after exact-size validation.
An already complete data file is validated and reused.

CLI:

```bash
uv run bobsphog-page-store build --checkpoint CHECKPOINT --store STORE
uv run bobsphog-page-store inspect --store STORE
uv run bobsphog-page-store evict --store STORE
```

## Runtime contract

`ContiguousExpertSource` maps the complete file into virtual address space without
materializing it in RAM. `load(layer, expert)` computes one stable offset, clones
one page, optionally pins that allocation, and exposes two tensor views matching
`ExpertWeights`. The complete mapped file does not count against CUDA residency;
the existing expert cache still bounds GPU memory.

Initialization rejects unsupported versions, inconsistent metadata, truncated
data, checkpoint-geometry mismatches, and mismatched checkpoint config/index
hashes when the caller supplies its checkpoint root. Runtime coordinates are
bounds checked.

On platforms exposing `POSIX_FADV_DONTNEED`, the `evict` command requests that
the kernel discard clean cached pages for this file. B2.4 invokes it immediately
before each controlled cold-source trial. This is best-effort kernel advice, not
a claim that unrelated checkpoint files or system caches were globally flushed.

## Deliberate limitations

- Version 1 stores BF16 only and performs no compression or quantization.
- The file duplicates the checkpoint's expert weights and requires 60 GiB for the
  target Qwen model.
- It optimizes exact random page reads. Superpage prefetch and direct I/O are
  separate experiments after B2.4 establishes the storage-layout effect.

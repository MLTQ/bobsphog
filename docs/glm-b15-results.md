# GLM-5.2 B1.5 feasibility results

## Result

GLM-5.2 completed a real one-token forward on a 128 GiB Framework Desktop
with a Radeon 8060S. The 1.506 TB BF16 checkpoint remained file-backed while
the B1.5 loader kept the causal scaffold and a bounded set of exact routed
experts in GPU-addressable memory.

The measured PyTorch peak was **38,501,770,240 bytes (35.86 GiB)**, or 2.56% of
checkpoint bytes. This establishes the requested feasibility result: GLM-5.2
can execute on this Strix Halo with exact demand-paged experts even though the
full checkpoint is more than twelve times larger than host RAM.

## Model geometry

The official checkpoint contains 78 causal layers. Layers 0–2 are dense and
layers 3–77 are sparse, with 256 routed experts and eight selected experts per
token. The checkpoint also includes one MTP layer that is not part of the
causal forward and was excluded from the scaffold.

One exact BF16 expert page contains gate, up, and down matrices:

$$
3(6144)(2048)(2\text{ bytes})
=75{,}497{,}472\text{ bytes}
=72\text{ MiB}.
$$

The 16-page cache therefore has a fixed capacity of 1,207,959,552 bytes
(1.125 GiB). A one-token forward selects eight pages in each of 75 sparse
layers, for 600 exact expert requests.

## Measured residency and latency

| Measurement | Value |
|-------------|------:|
| Checkpoint tensor bytes | 1,506,659,919,872 |
| Resident causal tensors | 1,194 |
| Resident causal scaffold | 37,202,615,808 bytes |
| Expert-cache capacity | 1,207,959,552 bytes |
| Measured peak allocation | 38,501,770,240 bytes |
| Peak/checkpoint fraction | 2.555% |
| Model construction and load | 29.012 s |
| One-token forward | 42.491 s |

The loader ignored 58,391 routed-expert and MTP tensors and read the resident
scaffold from 85 checkpoint shards. Peak allocation closely tracks scaffold
plus cache capacity and transient attention/output storage; there is no hidden
materialization of the routed-expert reservoir.

## Exact page traffic

The empty cache produced the expected cold behavior:

| Metric | Value |
|--------|------:|
| Requests | 600 |
| Hits | 0 |
| Misses | 600 |
| Evictions | 584 |
| Exact expert bytes transferred | 45,298,483,200 |
| Source-load time | 39.032 s |
| Source plus scheduling time | 39.152 s |
| Host wait time | 2.16 ms |

Reactive source loading dominates the forward. The run proves bounded exact
execution, not acceptable generation throughput; prediction, prefetch, and
larger prompt-level working sets remain necessary.

## Validation

Before the full forward, one real 72 MiB GLM expert was loaded from its three
individual safetensor entries, packed into the generic B1.5 page layout, and
executed through the ROCm cache. Its BF16 output was bit-identical to direct
resident execution (`max_abs_error = 0`).

The complete checkpoint was also validated against its official index:

- 282/282 shard files present;
- 59,585/59,585 indexed tensor keys present;
- every safetensors header readable; and
- no missing or extra indexed keys.

The end-to-end probe used the raw one-token input `Hello` (token ID 9703) and
completed all 75 sparse layers. A full-resident reference is physically
impossible on this host, so this experiment does not claim full-model output
parity or language-quality validation. It establishes execution feasibility
and exact individual expert-page behavior.

## Environment and reproducibility

- Hardware: AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 122 GiB RAM
- GPU-addressable GTT aperture: 65,945,616,384 bytes
- Runtime: ROCm 7.2.4, PyTorch 2.10.0, Transformers 5.12.0
- Checkpoint revision: `b4734de4facf877f85769a911abafc5283eab3d9`
- Storage: shards 1–141 on the system NVMe and 142–282 on a second NVMe,
  presented through one symlinked checkpoint directory
- Raw result: `outputs/glm-b15-strix.json`

Run with:

```bash
bobsphog-glm-b15 \
  --checkpoint /srv/models/GLM-5.2 \
  --cache-pages 16 \
  --seed-text Hello
```

The initial four-transfer download plus a simultaneous Docker image pull
hard-locked this host. The successful setup used one checkpoint worker per SSD,
forced Docker Hub's stalled CDN connections to IPv4, and unloaded the unused
MediaTek Wi-Fi modules. The final download and inference run then remained
stable for more than an hour.

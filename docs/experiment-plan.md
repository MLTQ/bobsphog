# Experiment plan

## Recommendation: toy first, pretrained second

Use a staged plan rather than choosing one starting point permanently.

### Stage A: nanoGPT-like toy model

Start with a small decoder-only transformer trained on a compact corpus with at
least two separable domains. This stage is for causal and systems validation,
not impressive language quality.

Advantages:

- full-model, partial-model, and oracle-ablation runs are cheap;
- every page, transfer, activation, and loss change can be logged;
- page size, rank, budgets, and dropout distributions can be swept quickly;
- failures in replay or retrieval are easier to localize; and
- synthetic domains can test whether known latent structure is recoverable.

Limitations:

- small models may localize or memorize differently from useful LLMs;
- toy corpora understate general-language and reasoning dependencies;
- simulated paging does not establish a real hardware advantage; and
- success does not prove useful pretrained knowledge can be decomposed.

### Stage B: pretrained 1–3B checkpoint

After the mechanism passes the Stage A gates, adapt a permissively licensed,
well-supported 1–3B decoder checkpoint. Prefer a model with ordinary transformer
blocks, accessible weights and tokenizer, strong evaluation support, and no
architectural feature that confounds the first paging experiment.

Keep attention resident initially. Decompose FFN matrices into a base
approximation plus residual low-rank pages, train the pages and controller to
reconstruct teacher behavior, then fine-tune with multi-budget objectives.

Advantages:

- tests whether the method preserves real pretrained knowledge;
- avoids the cost of pretraining useful language capability;
- yields credible VRAM and latency measurements; and
- allows teacher-logit distillation from the unchanged checkpoint.

Costs:

- oracle utility labels and sweeps are much more expensive;
- framework and kernel overhead can obscure the mechanism; and
- decomposition damage must be separated from retrieval failure.

**Recommendation:** do not begin by training a 1–3B model from scratch. Do not
stop after the toy result either. The toy model proves mechanics; the pretrained
model establishes relevance.

## 1. Research questions

1. Does a useful prompt-conditioned parameter working set exist at the chosen
   page granularity?
2. Does structured multi-budget training improve the quality/working-set curve
   over post-hoc decomposition or pruning?
3. Can a learned retriever approach an oracle page selector using only resident
   state?
4. Do relationship-graph bundles outperform independent page ranking?
5. Can page transfers be predicted early and amortized over a prompt?
6. Do additive pages enable cheaper refinement than rerunning the full model?
7. Does coded redundancy improve graceful degradation enough to justify its
   storage, decode, and interference costs?

## 2. Stage A implementation phases

### A0. Build measurement infrastructure

Implement a normal small transformer and collect full-model reference outputs.
Add exact accounting for:

- resident and peak bytes;
- logical page bytes and physical allocations;
- simulated and real bytes transferred;
- per-layer compute and wall time;
- page residency, cache hits, evictions, and churn; and
- per-example quality at every budget.

This prevents a parameter-count result from being mistaken for a memory or
latency result.

### A1. Static decomposition, no learned retrieval

For each FFN matrix, construct a resident low-rank approximation and partition
the residual into independently executable low-rank pages. Verify:

$$
W\approx W_{\mathrm{base}}+\sum_i U_iV_i^\top.
$$

Measure reconstruction error and language-model loss as pages are added by
oracle residual magnitude. This establishes the best easy quality/density curve
before routing is involved.

### A2. Variable page dropout and multi-budget training

Train or fine-tune with whole-page masks, sampling both dropout rate and explicit
memory budget. Compare independent masks, nested budget masks, and bundle masks.
Check whether quality improves smoothly and whether capacity collapses into a
few universal pages.

### A3. Oracle and learned retrieval

For a manageable candidate pool, measure true marginal loss improvement from
adding each omitted page. Train a ranking/regression controller on those values.
Evaluate:

- random selection;
- activation or router-frequency heuristics;
- gradient/Fisher proxies;
- nearest-neighbor page keys;
- learned counterfactual utility; and
- oracle greedy selection.

The gap between learned and oracle selection distinguishes “no useful sparse
working set” from “useful set exists, but retrieval is poor.”

### A4. Relationship graph and bundle selection

Measure co-activation and pairwise complementarity for sampled page pairs. Build
a top-$k$ graph and compare independent top-$k$ retrieval with graph-expanded
bundles. Track whether bundles improve quality enough to offset extra bytes.

### A5. Paging and refinement runtime

Use CPU RAM as the cold tier. Introduce bounded GPU cache, asynchronous
transfers, contiguous superpages, prefetch, and eviction. Compare:

1. one skeleton prefill plus full rerun;
2. rerun only from the earliest changed layer;
3. cached suffix replay; and
4. approximate additive state correction where valid.

Report time to first token and end-to-end latency, not only kernel time.

## 3. Stage A data design

Use at least two visibly different domains plus shared language. For example,
one structured/symbolic domain and one natural-language factual domain. Include:

- training examples from each domain;
- held-out compositions requiring both domains;
- rephrasings and adversarial surface variation;
- out-of-domain general text; and
- explicit topic-shift prompts for cache churn tests.

A synthetic task with known independent and interacting latent rules is useful
for testing whether the relation graph recovers true dependencies. It should
supplement, not replace, a language corpus.

## 4. Stage A go/no-go gates

Advance to a pretrained model only if:

1. The oracle selector beats random, magnitude-only, and static equal-size
   subsets at the same resident bytes.
2. Learned retrieval closes a meaningful fraction of the gap between a cheap
   heuristic and the oracle on held-out examples.
3. Multi-budget training improves at least part of the quality/bytes frontier
   without catastrophic full-budget regression.
4. Prompt-level page sets are materially more stable than token-level sets.
5. Transfer-aware selection retains an advantage after measured or realistically
   simulated I/O costs.
6. Added pages improve expected quality often enough for an anytime claim; all
   non-monotonic cases are reported.

If gate 1 fails, the representation or granularity lacks useful locality. If
gate 1 passes but gate 2 fails, focus on the index and counterfactual labels. If
quality gates pass but gate 5 fails, the idea may still be useful for model
extraction but not demand paging.

## 5. Stage B: pretrained 1–3B validation

### B1. Select and freeze a teacher

Choose the checkpoint based on license, architecture simplicity, ecosystem
support, and available domain/general benchmarks. Record exact version, dtype,
tokenizer, context length, and reference scores.

### B2. Establish the resident floor

Measure the fixed memory cost of embeddings, attention, normalization, output
head, KV cache, runtime workspace, and the smallest acceptable FFN base. This
determines whether paging residual capacity can matter at all on the target
device.

### B3. Decompose and reconstruct

Compare:

- singular-value or other low-rank factorization;
- neuron/block partitioning;
- learned uncoded residual atoms; and
- only later, sparse coded atoms.

First train to match hidden states and teacher logits; then add domain and
general instruction data. Preserve a general-data mixture to avoid creating a
specialist that recognizes vocabulary but loses instruction following.

### B4. Train the controller

Use sampled page ablations, teacher disagreement, and downstream loss to train
counterfactual utility. Restrict expensive candidate evaluation with the
associative index, but periodically sample outside its top results to prevent
blind spots.

### B5. Run hardware experiments

Test fixed VRAM budgets with CPU-resident exact pages. Only after prompt-level
RAM paging works should NVMe be introduced. Use page-aligned storage, pinned
buffers, asynchronous reads, and superpage layouts when testing NVMe.

## 6. Comparison baselines

All comparisons must use the same task data and report both nominal parameter
count and measured peak memory.

- Full model at native precision.
- Full model with equal-memory quantization.
- Static magnitude pruning.
- Structured neuron/channel/head pruning where applicable.
- Post-hoc sparse pruning with an appropriate sparse runtime.
- Ordinary low-rank decomposition at the same resident bytes.
- Static task-specific page subset.
- Random page selection.
- Frequency- or activation-based routing.
- Learned uncoded atom dictionary.
- Random fountain-coded dictionary.
- Learned coded dictionary with erasure training.
- Conventional MoE or expert-offload analogue at comparable granularity.
- Generic CPU/GPU layer offloading.
- A separately distilled small model at similar VRAM and latency.
- Oracle greedy page selection as an upper bound, not a deployable baseline.

## 7. Metrics

### Quality

- Language-model loss/perplexity.
- Domain task accuracy or exact match.
- General-language and instruction-retention scores.
- Teacher KL divergence and top-token agreement.
- Calibration and abstention/uncertainty quality.
- Robustness to rephrasing, composition, and topic shifts.
- Quality at every page budget and rate of non-monotonic refinements.

### Memory and storage

- Fixed skeleton bytes.
- Exact resident page bytes.
- Measured peak GPU allocation, including workspaces and staging.
- CPU and disk footprint.
- KV-cache use as context grows.
- Index and relationship-graph overhead.

### Transfer and cache behavior

- Host/disk bytes read per prompt and per generated token.
- Useful versus wasted prefetched bytes.
- Cache hit rate and late page-fault rate.
- Page-set Jaccard similarity across adjacent tokens.
- Evictions, reloads, and working-set churn.
- Sequential versus random I/O characteristics.

### Latency and throughput

- Time to first token.
- Prefill latency.
- Inter-token latency and tokens per second.
- Refinement latency and number of passes.
- Overlap between transfer and compute.
- End-to-end energy if measurable.

### Retrieval quality

- Recall of oracle top-$k$ pages and bundles.
- Ranking correlation with measured marginal utility.
- Regret versus oracle under the same byte/latency budget.
- Utility per transferred byte.
- Performance on pages or combinations not frequently sampled during training.

### Specialization and structure

- Page overlap between domains.
- Fraction of universal versus domain-selective pages.
- Relation-graph sparsity and bundle stability.
- Capability gained or lost when a page neighborhood is removed.
- Graceful degradation under random and adversarial erasures.

## 8. Required ablations

Vary one element at a time:

- no page dropout versus fixed-rate versus variable-rate dropout;
- task loss only versus multi-budget distillation;
- independent ranking versus relationship-graph expansion;
- static graph versus context-conditioned relationships;
- page rank, byte size, and superpage packing;
- prompt-level fixed set versus periodic decode epochs versus per-token routing;
- skeleton capacity versus pageable capacity;
- rerun versus suffix replay versus approximate corrections;
- no churn penalty versus several strengths;
- uncoded atoms versus random coding versus learned coding; and
- CPU-resident versus NVMe-resident pages.

## 9. Decisive plots

The main plots should show task quality against:

- peak VRAM;
- total bytes transferred;
- time to first token;
- end-to-end latency; and
- steady decoding throughput.

Also plot quality as pages arrive over time. A strong result would show a fixed
resident skeleton plus query-selected pages approaching the full teacher on the
current domain while occupying less peak GPU memory and outperforming static
equal-memory baselines after transfer costs are included.

## 10. Suggested implementation order

1. Instrumented toy transformer.
2. Static low-rank base plus exact residual pages.
3. Oracle quality-versus-page curves.
4. Variable structured page dropout and multi-budget distillation.
5. Counterfactual retriever.
6. Sparse relationship graph and bundle retrieval.
7. CPU-to-GPU cache, prefetch, and prompt-level paging.
8. Pretrained 1–3B decomposition and recovery training.
9. Real hardware comparisons against quantization and offloading.
10. Only then: NVMe, coded redundancy, attention paging, and sophisticated
    incremental correction.

This order keeps each failure interpretable and prevents the expensive systems
pieces from masking whether a query-specific working set exists at all.

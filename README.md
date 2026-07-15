# bobsphog

**bobsphog** is a research prototype for a *spongiform neural network*: a
budget-conditioned, demand-paged model that keeps a small approximate scaffold
resident and retrieves exact, query-relevant parameter pages only when they are
likely to improve the answer.

The intended loop is:

```text
resident skeleton -> approximate prefill -> predict useful omitted pages
                  -> fetch a coherent page bundle -> apply corrections
                  -> refine or decode within a fixed memory budget
```

The goal is not to reconstruct every parameter of a large model. It is to
maximize answer quality under limits on peak memory, transfer volume, and
latency:

$$
\max_S Q(f_S(x))
\quad\text{subject to}\quad
\operatorname{VRAM}(S)\le B,\qquad
\operatorname{I/O}(S)\le T.
$$

## Core idea

Split each pageable layer into:

1. A small, permanently resident approximation that always produces a usable
   result.
2. Independently executable low-rank residual pages stored in GPU memory, CPU
   memory, or eventually NVMe according to temperature.
3. A compact associative index and retrieval controller that estimate which
   omitted pages—or interacting bundles of pages—would most improve the current
   computation.

For layer $\ell$:

$$
h_{\ell+1}
=F_{\ell,\mathrm{base}}(h_\ell)
+\sum_{i\in S_\ell}g_i(h_\ell)
U_i^{(\ell)}V_i^{(\ell)\top}h_\ell.
$$

The low-rank terms are additive corrections. This structure is important: it
makes a page executable on its own and creates a path toward refining cached
states without rerunning the entire transformer.

The model is trained at many page budgets with structured page dropout, so the
base path remains coherent, additional pages improve fidelity, and the
retriever learns the counterfactual value of fetching a page that is not yet
resident.

## What “holographic” means here

A conventional fountain code redundantly mixes all source data so that almost
any sufficiently large packet set can recover the whole. That provides erasure
resilience, but it does **not** localize a capability or reduce the number of
degrees of freedom needed to reconstruct arbitrary weights.

This project uses “holographic” more narrowly: a compact, distributed,
associative *index over parameter pages*. It supports approximate questions such
as “which omitted page or computational neighborhood is valuable for this
hidden state?” The exact pages remain separately executable. Sparse task support
must be learned through routing, multi-budget training, and causal utility
objectives; random invertible mixing alone will not provide it.

## Research hypothesis

A model trained for graceful partial residency can have a much smaller
query-specific working set than its full parameter set. A resident approximation
plus a learned counterfactual retriever may select that working set early enough
to hide transfers, hold peak accelerator memory fixed, and approach the full
model's quality on the current query.

The decisive result is not merely “the sparse model still runs.” It is a better
quality/VRAM/I/O/latency frontier than equal-memory quantization, static pruning,
low-rank compression, and ordinary paging policies.

## Where to start

Start with a **toy nanoGPT-like transformer**, then move to a **small pretrained
1–3B model**.

The toy stage is the right first experiment because it makes page construction,
structured dropout, oracle ablations, retrieval supervision, correction replay,
and cache accounting cheap enough to debug exhaustively. It can establish that
the mechanism works, but not that it transfers useful knowledge or improves a
real deployment frontier.

The pretrained stage is therefore mandatory for relevance. Once the toy model
passes explicit gates, decompose or adapt the FFN residuals of a permissively
licensed 1–3B checkpoint while initially keeping attention and other universal
components resident. This isolates the paging hypothesis from the cost and
confounds of training a language model from scratch.

See:

- [Architecture](docs/architecture.md) for the model, index, objectives, and
  inference system.
- [Experiment plan](docs/experiment-plan.md) for staged tests, baselines,
  metrics, ablations, and go/no-go criteria.

## Scope boundaries

The first prototype should use CPU RAM as the cold tier, page FFN residuals
rather than every tensor, choose most pages once per prompt, and tolerate only
occasional decode-time page faults. NVMe, attention paging, coded redundancy,
and fully incremental state repair should be added only after the fundamental
quality-versus-working-set result is demonstrated.

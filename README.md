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
- [A2 results](docs/a2-results.md) for the first learned multi-budget run and
  the resident-capacity-collapse control.
- [A3 results](docs/a3-results.md) for oracle and learned counterfactual page
  selection across equal logical budgets.
- [A4 results](docs/a4-results.md) for the compositional task and sparse signed
  page-relationship graph.

## Current prototype

The first toy milestone implements a nanoGPT-like causal transformer whose FFN
matrices are decomposed into a resident low-rank base plus ordered executable
low-rank pages. It includes full, base-only, uniform-prefix, and reproducible
structured-dropout page plans; logical byte accounting; and page execution
traces.

Run it locally with:

```bash
uv sync
uv run pytest
uv run bobsphog-smoke
uv run bobsphog-a2
uv run bobsphog-a3
uv run bobsphog-a4
```

The smoke command reports output divergence from the full model as logical
resident parameter bytes increase. All page tensors are still physically
allocated in one PyTorch model; real CPU/GPU demand paging comes after the
logical mechanics and training objectives are validated.

The Mac is sufficient for decomposition tests, controller plumbing, and small
debug training runs. Move to the 4090 when A2 multi-budget training needs broad
hyperparameter sweeps, when CUDA transfer/cache measurements begin, or when the
project advances to a pretrained 1–3B checkpoint.

The A2 command trains a dense teacher on addition and multiplication modulo ten,
converts it exactly into the paged representation, trains the student with
variable structured page dropout and teacher distillation, and emits static SVD
and sampled-mask budget curves plus per-domain page-ablation utilities as JSON.
The resident skeleton is frozen during student training by default to prevent
the toy task from collapsing entirely into always-resident weights; pass
`--train-resident` to reproduce that control. For a fast wiring check, reduce
`--teacher-steps`, `--student-steps`, and `--batch-size`.

The A3 command retrains the A2 fixture, collects direct counterfactual utility
labels for omitted pages, fits a query-and-resident-set utility estimator, and
compares learned selection against random pages, static SVD order, and a
label-aware greedy oracle at identical page counts. These are still logical
selection experiments; no physical page transfer occurs yet.

The A4 command moves to order-sensitive compositional arithmetic, builds a
sparse signed pair-interaction graph from calibration examples, and compares
independent page scores against graph-expanded bundles for both calibrated and
learned selectors.

## Scope boundaries

The first prototype should use CPU RAM as the cold tier, page FFN residuals
rather than every tensor, choose most pages once per prompt, and tolerate only
occasional decode-time page faults. NVMe, attention paging, coded redundancy,
and fully incremental state repair should be added only after the fundamental
quality-versus-working-set result is demonstrated.

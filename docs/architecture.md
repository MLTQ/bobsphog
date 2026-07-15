# Architecture

## 1. Problem framing

Sparse mixture-of-experts models show that compute can be conditional, but most
implementations still need all expert weights somewhere readily accessible.
Dense models have no clean expert boundaries, and their useful features may be
distributed across neurons and layers. Conventional offloading lowers VRAM by
moving fixed tensors, but transfer latency can dominate if the system discovers
its working set too late or changes it every token.

The proposed system treats parameters as a demand-paged associative memory. It
should:

- produce a coherent approximation from a small resident scaffold;
- predict the value of exact parameter regions that were not executed;
- retrieve a query-specific, interacting subgraph of executable pages;
- improve gracefully at larger memory budgets;
- keep peak accelerator memory bounded; and
- amortize retrieval over a prompt or a stable decoding epoch.

Let the full parameter set be partitioned into pages
$\mathcal B=\{B_1,\ldots,B_N\}$. At refinement step $r$, only $S_r\subseteq
\mathcal B$ is resident. The optimization target is:

$$
S_x^*=arg\max_{S}
Q(f_S(x))
\quad\text{s.t.}\quad
\operatorname{bytes}(S)\le B,
\quad \operatorname{transfer}(S)\le T.
$$

This is an **anytime model in parameter space**. For budgets $b$, it produces
$p_b(y\mid x)$, and training should make expected quality non-decreasing:

$$
\mathbb E[Q(p_{b+\Delta b})]\ge \mathbb E[Q(p_b)].
$$

Strict per-example monotonicity is unlikely without explicit safeguards, so it
must be measured rather than assumed.

## 2. Three representations of the model

### 2.1 Resident skeleton

The skeleton $W_{\mathrm{base}}$ is always in accelerator memory and must be able
to run independently. Initially it should contain:

- embeddings and output head;
- attention, normalization, and residual plumbing;
- a low-rank or otherwise compressed FFN approximation at every layer;
- the retrieval controller and compact index; and
- optionally, a few universally valuable residual pages.

This deliberately preserves shared language and reasoning machinery instead of
assuming that “unrelated knowledge” can be cleanly removed. The first prototype
should page FFN residual capacity only. Attention paging can be studied later if
the fixed skeleton becomes the dominant memory cost.

### 2.2 Exact executable weight pages

Write a pageable matrix as a base plus residual atoms:

$$
W^{(\ell)} = W_{\mathrm{base}}^{(\ell)}
+\sum_{i=1}^{N_\ell} B_i^{(\ell)},
\qquad
B_i^{(\ell)}=U_i^{(\ell)}V_i^{(\ell)\top}.
$$

With active set $S_\ell$:

$$
h_{\ell+1}=F_{\ell,\mathrm{base}}(h_\ell)
+\sum_{i\in S_\ell}g_i(h_\ell)
U_i^{(\ell)}(V_i^{(\ell)\top}h_\ell).
$$

Low-rank pages are attractive because they are independently executable,
compact, additive, and compatible with ordinary dense GPU operations. Other
possible page units include neuron blocks, quantized tiles, or small experts,
but tiny pages create random I/O and kernel-launch overhead while large pages
waste memory. Logical atoms should therefore be packable into contiguous
multi-megabyte superpages for transfer.

Pages occupy a hierarchy:

```text
GPU VRAM: hot pages and fixed skeleton
CPU RAM:  warm pages and prefetch buffer
NVMe:     cold pages (later-stage experiment)
```

### 2.3 Holographic or associative index

An optional conceptual form for a resident distributed sketch is:

$$
H=\sum_{i=1}^{N} k_i\circledast\phi(B_i),
$$

where $k_i$ is a page key, $\phi(B_i)$ a compact descriptor, and
$\circledast$ a binding operation. In practice, learned key/value embeddings
and approximate nearest-neighbor retrieval are a simpler first implementation:

$$
\operatorname{score}(i\mid h,S,u)
=q(h,S,u)^\top a_i,
$$

where $u$ captures uncertainty and $a_i$ is the page key. Multiple embeddings
may describe semantic affinity, causal repair value, and sequential use.

The index is a semantic page table, not a replacement checkpoint. A generic
fountain transform $z=Aw$ is invertible redundancy: reconstructing arbitrary
$w$ still requires roughly the original number of independent degrees of
freedom, and a transformer cannot execute $z$ without decoding unless its layers
are specifically redesigned. Useful sparse support must therefore be learned.

Fountain-like coding may later add erasure tolerance or progressive loading by
forming sparse coded atoms

$$
C_i=\sum_{j\in\mathcal N(i)}s_{ij}B_j,
$$

but it should be evaluated after the uncoded learned dictionary. Otherwise
coding gains and retrieval gains cannot be separated.

## 3. Sparse relationship graph

A dense $N\times N$ “holographicity tensor” is too expensive. Store only the
top-$k$ directed neighbors of each page:

$$
\mathcal N(i)=\operatorname{TopK}_j G_{ij}.
$$

Edges may combine:

$$
G_{ij}=\alpha G_{ij}^{\mathrm{coactivation}}
+\beta G_{ij}^{\mathrm{causal}}
+\gamma G_{ij}^{\mathrm{gradient}}
+\delta G_{ij}^{\mathrm{sequential}}
+\eta G_{ij}^{\mathrm{repair}}.
$$

- **Coactivation:** pages often useful on the same examples.
- **Causal complementarity:** joint inclusion matters more than either page
  alone.
- **Gradient affinity:** pages receive aligned task gradients.
- **Sequential prediction:** use of one page predicts later-layer pages.
- **Residual repair:** one page repairs errors left by another subset.

Use the graph to expand a high-scoring seed into a coherent computation bundle.
This addresses interaction effects such as

$$
\Delta\mathcal L_{i,j}
\gg \Delta\mathcal L_i+\Delta\mathcal L_j.
$$

Because relationships depend on context, a static graph is only a prior. A
factorized conditional relation can refine it:

$$
G_{ij}(h)\approx a_i^\top C(h)b_j,
\qquad a_i,b_j\in\mathbb R^d,\ d\ll N.
$$

The extracted working model is the induced, query-conditioned subgraph
$M_x=G[S_x]$, not a single human-legible “physics expert.”

## 4. Counterfactual utility estimator

A missing page cannot emit an activation or an ordinary gradient because it was
not executed. Retrieval must estimate the counterfactual:

> How much would fetching exact page $B_i$ improve the answer given the current
> state and resident set?

For loss $\mathcal L$, the true marginal value is:

$$
\Delta_i(S,x)=
\mathcal L(f_S(x))-\mathcal L(f_{S\cup\{i\}}(x)).
$$

Train a controller $R$ to predict it:

$$
\widehat{\Delta}_i=R(q(x,h),\phi(B_i),S).
$$

Labels come from sampled inclusions, exclusions, and bundle ablations during
training. Useful attribution signals can combine activation-gradient products,
Fisher-style scores, and direct ablation:

$$
I_i=\mathbb E_{x\sim D}
\left[
\lambda_1\left|z_i\frac{\partial\mathcal L}{\partial z_i}\right|
+\lambda_2\Delta_i^{\mathrm{ablation}}
+\lambda_3\operatorname{Fisher}_i
\right].
$$

Direct ablation is the target of record; cheaper signals are approximations.
Single-prompt traces are insufficient. Calibration must cover varied phrasing,
edge cases, and general capabilities the specialist must retain.

Rank candidates by expected gain per systems cost:

$$
U(B_i)=
\frac{\mathbb E[\widehat{\Delta}_i\mid q,S]}
{\operatorname{bytes}(B_i)+
\lambda\operatorname{latency}(B_i)+
\mu\operatorname{churn}(B_i)}.
$$

Bundle scores should also be learned because independent top-$k$ selection
misses page synergy.

## 5. Training objectives

### 5.1 Variable structured page dropout

Sample a dropout rate and masks over whole pages, not individual scalar weights:

$$
p\sim P(p),\qquad m_i\sim\operatorname{Bernoulli}(1-p),
$$

$$
W_{\mathrm{effective}}
=W_{\mathrm{base}}+\sum_i m_i B_i.
$$

Training across mild to extreme dropout teaches robustness across residency
budgets. However, dropout alone teaches survival under damage, not intelligent
retrieval.

### 5.2 Multi-budget distillation

Match the full teacher at several explicit budgets $\mathcal B$:

$$
\mathcal L_{\mathrm{budget}}
=\sum_{b\in\mathcal B}\lambda_b
D_{\mathrm{KL}}\left(p_{\mathrm{full}}\,\|\,p_b\right).
$$

Combine this with the language-model task loss at every budget. Sampling nested
sets $S_b\subset S_{b+\Delta b}$ and adding a monotonicity penalty can discourage
additional pages from degrading predictions:

$$
\mathcal L_{\mathrm{mono}}
=\sum_b\max(0,\mathcal L_{b+\Delta b}-\mathcal L_b+m).
$$

### 5.3 Retrieval supervision

For sampled omitted candidates:

$$
i^*=\arg\max_{i\notin S}\Delta_i(S,x),
\qquad
\mathcal L_{\mathrm{router}}=-\log P_R(i^*\mid h,S).
$$

Regression or ranking losses over measured $\Delta_i$ retain more information
than only the best-page label. Include bundle-level targets for interaction
effects.

### 5.4 Locality, repairability, and redundancy

Useful auxiliary pressures are:

- **Churn:** penalize $|S_t\triangle S_{t-1}|$ so sequences use stable working
  sets.
- **Predictability:** make early hidden states predict later page needs, enabling
  prefetch.
- **Repairability:** encourage pages to act as additive residual corrections
  instead of relying on brittle all-to-all interactions.
- **Diversity/load balance:** prevent every page from learning the same function
  or a few universal pages from absorbing all capacity.
- **Erasure loss (later):** randomly remove selected or coded pages and train for
  graceful degradation.

A schematic total objective is:

$$
\mathcal L=
\mathcal L_{\mathrm{LM}}
+\lambda_b\mathcal L_{\mathrm{budget}}
+\lambda_r\mathcal L_{\mathrm{router}}
+\lambda_m\mathcal L_{\mathrm{mono}}
+\lambda_c\mathcal L_{\mathrm{churn}}
+\lambda_d\mathcal L_{\mathrm{diversity}}
+\lambda_e\mathcal L_{\mathrm{erasure}}.
$$

## 6. Inference, retrieval, and caching

### 6.1 Prompt prefill

Run an approximate pass with the skeleton:

$$
h^{(0)}=f_{S_0}(x).
$$

Layer or prompt controllers emit retrieval queries. The index returns seeds,
the relation graph expands them into bundles, and the utility estimator ranks
bundles under a byte and latency budget:

$$
C_\ell=\operatorname{Retrieve}(q_\ell,H,G,S_r),
\qquad
S_{r+1}=S_r\cup\operatorname{TopK}(C_\ell).
$$

Prefetch likely pages while earlier layers execute. Most of the sequence-level
working set should be loaded during prefill.

### 6.2 Refinement

The simplest correct prototype reruns affected suffixes after pages arrive. The
target architecture applies additive corrections to cached intermediate states:

$$
h_\ell^{(r+1)}=h_\ell^{(r)}
+\Delta h_\ell(B_{S_{r+1}\setminus S_r}).
$$

Because nonlinear downstream layers depend on the changed state, exact local
correction may still require recomputing the suffix. The project must measure
when approximate replay is stable instead of assuming corrections commute
through nonlinearities.

Stop refinement when the budget is exhausted, the answer distribution
stabilizes,

$$
D_{\mathrm{KL}}(p_{r+1}\,\|\,p_r)<\epsilon,
$$

or marginal predicted utility falls below transfer cost.

### 6.3 Decode epochs, not per-token storage faults

SSD or host-to-device fetches on every token will usually erase the benefit.
Keep $S_{\mathrm{prompt}}$ stable during decoding and permit page faults only
when uncertainty, topic, or hidden-state routing changes materially. A new
epoch can retrieve and prefetch another bundle. Speculative hidden states can
predict future needs, but this is a later optimization.

### 6.4 Cache policy

Reserve a bounded GPU page cache plus asynchronous staging space. Eviction
priority can combine:

$$
\operatorname{priority}(B_i)=
\alpha\operatorname{recentUse}
+\beta\operatorname{predictedUse}
+\gamma\operatorname{reloadCost}
+\delta\operatorname{universality}
+\eta\operatorname{bundleAffinity}.
$$

Log useful bytes, wasted prefetched bytes, cache hits, late faults, evictions,
and page-set churn. The controller must optimize end-to-end latency, not merely
oracle accuracy.

## 7. Principal failure modes

- **No semantic locality:** useful computation may be too distributed for a
  small working set.
- **Fountain-code illusion:** random coded packets spread corruption rather than
  isolate capabilities and require decoding before normal execution.
- **Retriever circularity:** omitted pages provide no direct evidence; poor
  counterfactual estimates may never fetch the pages that would correct them.
- **Combinatorial interactions:** individually weak pages can be valuable only as
  a bundle.
- **Approximation drift:** an early skeleton error changes later routing and
  creates self-reinforcing divergence.
- **Non-monotonic refinement:** more pages can perturb representations and reduce
  quality.
- **Transfer granularity:** small pages cause random reads and launch overhead;
  large pages waste memory and bandwidth.
- **Bandwidth domination:** PCIe or storage transfers may cost more than saved
  computation.
- **Page churn:** token-level routing instability destroys cache locality.
- **Universal-capacity floor:** embeddings, attention, normalization, and common
  reasoning may leave a large irreducible skeleton.
- **Training expense:** counterfactual labels require many partial executions and
  ablations.
- **Capacity collapse:** excessive dropout can push all knowledge into the base
  or a few super-pages, defeating paging.
- **Benchmark overfitting:** a narrow calibration set may produce a brittle
  specialist that fails on rephrasing or prerequisite reasoning.
- **Runtime mismatch:** theoretical parameter savings may not map to faster
  kernels or lower peak allocations.

Each is an empirical question and should have a corresponding metric or ablation
in the experiment plan.

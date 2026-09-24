# Operation Design: gated_delta_net_backward

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + data_movement, multi-phase, single dispatch) |
| Goal | Vector–Jacobian product of the chunked gated delta rule (Gated DeltaNet linear attention). Consumes the six forward inputs plus `do` (and optionally `dht`) and returns six gradients in one native device-program dispatch. |
| Math | See **Reference forward** and **Backward derivation** below. |
| Mode | Derivative |
| References | `eval/golden_tests/gated_delta_net_backward/helpers.py::_chunk_gated_delta_rule_fwd` (the definition); `models/experimental/gated_attention_gated_deltanet/torch_functional/delta_rule_ops.py:114-245` (in-tree torch reference); `models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py` (forward adapter, I/O contract); `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp:16-49` (forward public contract); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md` |

### Reference forward (per batch element `b`, head `h`, chunk index `i`, chunk length `C`)

With `q̃ = q·scale`, `kβ = k⊙β`, `vβ = v⊙β`:

```
decay[t]   = Σ_{u≤t} g[u]                     (cumsum within the chunk)
γ[t]       = exp(decay[t]),   Γ = γ[C-1],     w[t] = exp(decay[C-1] − decay[t]) = Γ/γ[t]
L[t,s]     = exp(decay[t] − decay[s])         for s ≤ t, else 0
A          = −((kβ @ kᵀ) ⊙ L) ⊙ strict_tril   (strictly lower)
Tinv       = (I − A)^{-1}                     ("attn" in the reference; unit lower-triangular)
v_corr     = Tinv @ vβ                        [C,V]
kcd        = Tinv @ (kβ ⊙ γ)                  [C,K]     (k_cumdecay)
Q          = q̃ ⊙ γ                            [C,K]
P          = k ⊙ w                            [C,K]
intra      = (q̃ @ kᵀ ⊙ L) ⊙ tril_incl_diag    [C,C]
v_new_i    = v_corr_i − kcd_i @ S_i           [C,V]
o_i        = Q_i @ S_i + intra_i @ v_new_i    [C,V]
S_{i+1}    = Γ_i·S_i + P_iᵀ @ v_new_i         [K,V]
```

### Backward derivation (what this op computes)

Reverse recurrence, `dS_N = dht` (or 0), walking `i = N−1 … 0`, with
`u_i = intra_iᵀ @ do_i` and `c_i = Q_iᵀ @ do_i` (both **state-independent**, hence chunk-parallel):

```
dv_new_i = u_i + P_i @ dS_{i+1}                                [C,V]
dS_i     = Γ_i·dS_{i+1} + c_i − kcd_iᵀ @ dv_new_i              [K,V]
dh0      = dS_0                                                (only when initial_state is given)
```

Per-chunk gradient assembly (needs `S_i`, `dS_{i+1}`, `v_new_i`, `dv_new_i`, all from the scans):

```
dQ    = do_i @ S_iᵀ                                            [C,K]
Mraw  = do_i @ v_new_iᵀ ;  Mt = Mraw ⊙ tril_incl_diag ; M = Mt ⊙ L
dP    = v_new_i @ dS_{i+1}ᵀ                                    [C,K]
d_kcd = −dv_new_i @ S_iᵀ                                       [C,K]
dΓ    = Σ_{K,V}( dS_{i+1} ⊙ S_i )                              scalar
dvβ   = Tinvᵀ @ dv_new_i ;  dU = Tinvᵀ @ d_kcd
d_attn = dv_new_i @ vβᵀ + d_kcd @ Uᵀ           (U = kβ ⊙ γ)     [C,C]
dA    = (Tinvᵀ @ d_attn @ Tinvᵀ) ⊙ strict_tril                 [C,C]
W     = −dA ⊙ L                                                [C,C]
dkβ   = W @ k + dU ⊙ γ
dq    = scale · ( dQ ⊙ γ + M @ k )                             [C,K]
dk    = Mᵀ @ q̃ + dP ⊙ w + Wᵀ @ kβ + dkβ ⊙ β                    [C,K]
dv    = dvβ ⊙ β                                                [C,V]
dβ    = rowsum_V( dvβ ⊙ v ) + rowsum_K( dkβ ⊙ k )              [C,1]
dγ    = rowsum_K( dQ ⊙ q̃ ) + rowsum_K( dU ⊙ kβ )               [C,1]
dw    = rowsum_K( dP ⊙ k )                                     [C,1]
dL    = Mt ⊙ (q̃ @ kᵀ)  −  dA ⊙ (kβ @ kᵀ) ;   R = dL ⊙ L
d_decay      = rowsum(R) − rowsum(Rᵀ) + dγ⊙γ − dw⊙w            [C,1]
d_decay[C-1] += Σ_t( dw[t]·w[t] ) + dΓ·Γ
dg    = reverse_cumsum(d_decay)  =  ut_onesᵀ-free form: UT_ones @ d_decay   [C,1]
```

Two properties this derivation makes structural, and which the golden suite pins:

* `dq` depends on `do` only (via `dQ` and `M`). With `do = 0`, `dq` is **exactly** zero — the `dht`
  path cannot reach `q`. No code path adds a `dS`-derived term to `dq`.
* The backward **never re-runs the forward-substitution loop**. It uses the stored `Tinv` and the
  algebraic inverse VJP `dA = Tinvᵀ · d_attn · Tinvᵀ` — two `[C,C]` matmuls.

### Padded-tail semantics (`T % chunk_size ≠ 0`)

The last chunk is padded with `q=k=v=do=0`, `g=β=0`. Consequences that the design relies on and the
implementer must preserve:

| Quantity at a padded row | Value | Why it is correct |
|---|---|---|
| `A`, `intra`, `P`, `vβ`, `kβ` rows | 0 | `k=β=0` |
| `Tinv` rows | identity row | `A` row is 0 |
| `decay[t]` | `decay[last real]` | `g_pad = 0` |
| `Γ_i = exp(decay[C-1])` | `exp(decay[last real])` | correct chunk decay |
| `d_decay[C-1]` | **non-zero** (carries `dΓ·Γ` and `Σ dw·w`) | must **not** be zeroed — it flows into real `dg` through the reverse cumsum |
| `dq/dk/dv/dg/dβ` at padded rows | never written | outputs have exactly `T` rows |

The reader zero-fills the padded region of every gathered block; the writer writes only real rows.

---

### Dispatch contract (read this before the CB layout)

**Exactly one native TTNN device-program dispatch per public invocation.** All of stage P, the
scan, and stage G live inside one `ProgramDescriptor`, sequenced by two per-group semaphore
rendezvous. Host-side validation, output pre-allocation and the `block_val_tiles` extent solve are
not dispatches; `ttnn.from_torch`, `ttnn.permute`, `ttnn.zeros` and every other TTNN op are, and
none of them appears in the entry point.

Two places in the surrounding code read the other way and are **wrong for this op as designed**:

* `eval/golden_tests/gated_delta_net_backward/axes.py`'s `observed()` docstring says
  "`device_num_programs` counts the programs — for this op that is expected to be more than one".
  It is not. The observe-only wrapper still works (it sums whatever it sees); the comment is a
  stale expectation, not a contract.
* The task brief's "prefer multiple `generic_op` dispatches over one kernel that does not fit L1"
  is a *fit* argument, and this design answers the fit argument a different way: the
  `block_val_tiles` extent knob plus the `l1_ledger.md` closed form guarantee the working set fits
  at every INPUTS shape (worst case 742 KB), so the multi-dispatch escape hatch is never needed.
  The barrier, not a second dispatch, is what separates the stages.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `q` | `ttnn.Tensor [B,T,H,K]` | yes | TILE, DRAM interleaved, L2-normalized along K | — | tensor |
| `k` | `ttnn.Tensor [B,T,H,K]` | yes | TILE, DRAM interleaved, L2-normalized along K | — | tensor |
| `v` | `ttnn.Tensor [B,T,H,V]` | yes | TILE, DRAM interleaved | — | tensor |
| `g` | `ttnn.Tensor [B,T,H]` | yes | TILE, log-space decay (≤ 0) | — | tensor |
| `beta` | `ttnn.Tensor [B,T,H]` | yes | TILE, in (0,1) | — | tensor |
| `do` | `ttnn.Tensor [B,T,H,V]` | yes | TILE | — | tensor |
| `dht` | `ttnn.Tensor [B,H,K,V]` or `None` | no | TILE | `None` | tensor (+ `has_dht` CT flag) |
| `initial_state` | `ttnn.Tensor [B,H,K,V]` or `None` | no | TILE | `None` | tensor (+ `has_h0` CT flag) |
| `chunk_size` | `int` | no | multiple of 32; `{32, 64}` at Phase 0 | 64 | CT (`chunk_size_tiles`) |
| `scale` | `float` | no | any finite | `K ** -0.5` | RT (bit-pattern) |
| `compute_kernel_config` | `DeviceComputeKernelConfig` | no | — | `HiFi4, math_approx=False, fp32_dest_acc_en=True` | program config |
| `memory_config` | `ttnn.MemoryConfig` | no | interleaved | `q.memory_config()` | host |

`scale` is applied on device (the gathered `q` block is multiplied by `scale`), never on host — a
host multiply would be a second dispatch.

Argument order is positional `(q, k, v, g, beta, do)` — **`g` before `beta`**, matching
`delta_rule_ops.py:114-124` and the shipping forward binding.

---

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `q,k [B,T,H,K]`; `v,do [B,T,H,V]`; `g,beta [B,T,H]`; `initial_state,dht [B,H,K,V]` |
| Dtype | `float32` or `bfloat16`, uniform across all inputs. `bfloat8_b` is INVALID at TARGET (`feature_spec.py`) |
| Layout | TILE (the whole layout universe for this op) |
| Memory | DRAM, interleaved |

**Tiling consequence that dominates the dataflow.** `q` is rank-4 `[B,T,H,K]`, so TILE layout tiles
the **last two** dims `(H,K)`. A page is therefore `[32 heads × 32 key-dims]` for **one token**, and a
single head occupies **one row** of every page. `H ∈ [1,8]` on every INPUTS entry, so the stored
tensor is `32/H` × its logical size and a per-head read touches `1/32` of every page it reads. This is
the single most important fact in this design; the Dataflow Strategy section below is built around it.
The shipping forward op sidesteps it with a host-side `permute([0,2,1,3])` — unavailable here, because
a permute is a second dispatch.

`g`/`beta` are rank-3 `[B,T,H]`, tiled over `(T,H)`: a page is `[32 tokens × 32 heads]`, so one head
is one **column** — a stride-32 access.

### Output

| Property | Value |
|----------|-------|
| Shape | `dq,dk [B,T,H,K]`; `dv [B,T,H,V]`; `dg,dbeta [B,T,H]`; `dh0 [B,H,K,V]` or `None` |
| Dtype | same as `q` |
| Layout | TILE |
| Memory | `memory_config` if given, else `q.memory_config()` (DRAM interleaved) |

`dh0` is `None` — not a zero tensor — when `initial_state is None`. The host allocates five outputs in
that case and six otherwise; the kernel's `dh0` write is gated on the `has_h0` compile-time flag so the
program hash differs between the two shapes.

`generic_op` returns only its last `io_tensors` entry, so the entry point pre-allocates all outputs,
passes them in `io_tensors`, discards the return handle and returns its own tuple.

### Phase 0 support intent

The implementer owns `SUPPORTED`; this table states what the design is built to cover, so the
rectangle is a decision rather than an accident. Every value below is served by the **same** code
path — there is no second regime hiding behind any of these axes.

| Axis (name as `feature_spec.py` TARGET uses it) | TARGET | Design covers | Note |
|---|---|---|---|
| `dtype` | `float32, bfloat16, bfloat8_b` | `float32, bfloat16` | `bfloat8_b` is **INVALID** at TARGET (block-quantized gate sequences), not a refinement candidate. Internal precision is `Float32` for the state / `Tinv` / decay regardless of input dtype, so `bfloat16` is a CB-format change at the boundaries only. |
| `layout` | `TILE` | `TILE` | the whole layout universe for this op |
| `state_mode` | `do_only, with_h0, with_h0_and_dht` | all three | `has_h0` and `has_dht` are compile-time flags; `do_only` produces five outputs, the others six |
| `chunk_size` | `32, 64` | both | `Ct = chunk_size/32`; `neumann_steps = ceil(log2(chunk_size))` is derived from it |
| `seq_alignment` | `chunk_aligned, chunk_ragged` | both | the reader zero-fills the padded tail; the writer emits exactly `T` rows |
| `head_dims` | `square, wide_v` | both | `K` and `V` are independent extents throughout; nothing in the design assumes `K == V` |
| (not a TARGET axis) | — | `H ≤ 32` | a **mechanism cap**, not a support axis: the source page index formula assumes `ceil(H/32) == 1`. `validate()` must raise for `H > 32`; every INPUTS entry has `H ≤ 8`. |

If the implementer needs to narrow Phase 0 further, `dtype = [float32]` is the one axis whose
removal costs nothing structural — `bfloat16` then becomes a boundary-format refinement.

---

## Blocking Model

### Axes

`Ct = chunk_size/32`, `Kt = ceil(K/32)`, `Vt = ceil(V/32)`, `NC = ceil(T/chunk_size)`, `BH = B·H`.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (B) | **independent** — no term of the recurrence couples batch elements | `block_batch = 1` | 1 | derived from the flattened `bh` work-item index (host) | flattened with `head` into `bh`, spread across the grid in stages P and G; one core per `bh` in stage S | — (already split) |
| `head` (H) | **independent** — each head has its own state `S [K,V]`; no cross-head term | `block_head = 1` | 1 | same | same as `batch` | — (already split) |
| `chunk` (NC) | **dependent** — `S_{i+1}` depends on `S_i` and `dS_i` on `dS_{i+1}`; a result spans the axis in both directions | `block_chunks = 1` | 1 | `CHUNKS_PER_BLOCK` host constant → CT arg | **split across cores in stages P and G** (the state-independent work); **not split** in stage S, which walks it sequentially on the `bh`-owning core | scheme-change (parallel prefix scan over `[K,K]` transition matrices — regime R5) |
| `token-in-chunk` (C) | **dependent** — the UT transform `(I−A)^{-1}` couples every token in the chunk; the chunk *is* the algorithm's coupling unit | `block_chunk_tiles = Ct` | `chunk_size/32` (1 or 2) — the **whole** chunk, never a sub-chunk | `chunk_size` parameter → CT `chunk_size_tiles` | never split across cores; splitting it would require inverting a block-triangular system across cores | scheme-change (no candidate) |
| `key_dim` (K) | **dependent** — contracted in `kβ@kᵀ`, `kcd@S`, `Pᵀ@v_new`; a result spans it | `block_key_tiles = Kt` | `ceil(K/32)` — the whole K extent, resident | `Kt` derived from `q.shape[-1]` (host) | not assigned across cores | scheme-change (K-split needs a cross-core sum on every `[C,V]` product) |
| `value_dim` (V) | **independent for the scans and `dv`; dependent for `dq/dk/dg/dbeta`** — the state factorizes exactly by V-column, but four of the six gradients reduce over V | `block_val_tiles` | largest divisor of `Vt` whose closed-form footprint fits L1 (`Vt` itself on every INPUTS entry except the `K=128,V=256,C=64` cell) | `block_val_tiles` solved once on host from the `l1_ledger.md` closed form → CT arg; `num_v_blocks = Vt / block_val_tiles` derived from it | **not** split across cores in Phase 0 — looped *within* the core, so the V-reduced gradients accumulate in L1 for free | scheme-change across cores (regime R4: needs a cross-core sum of `[C,C]`+2·`[C,K]`+scalars per chunk); knob-turn within the core |
| `gather token window` | implementation axis of the face-row gather (see Dataflow) | `gather_stage_tokens` | 32 (one tile row) | `GATHER_STAGE_TOKENS` host constant → CT arg | n/a (reader-local) | knob-turn |

Every axis has a row; the three "not split across cores" answers are decisions, with their reason in
the Character column.

**Intermediate-stage axes.** Stage S produces `S_i`, `dS_i` (`[K,V]` per chunk) and `v_new_i`,
`dv_new_i` (`[C,V]` per chunk). Their axes are the same `(bh, chunk, K, V)` set already tabled above:
`bh` stays spread across cores, `chunk` is the sequential axis by construction, `K` is whole, `V`
follows `block_val_tiles`. Stage S is the one stage that leaves grid cores idle: while `BH` cores walk
the scan, `min(grid, BH·NC) − BH` cores wait at the group barrier. That is a **work-assignment**
consequence of the axis characters, not a tunable extent — the chunk axis genuinely has no
parallelism available at Phase 0, and R5 is the regime row that would change it.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_gather_stage` | `gather_depth` | 2 | Overlaps the next token window's DRAM face-row reads with the RISC-V re-pack of the current one. This is the op's dominant data-movement term, so it is the one place depth is unconditionally worth its L1. |
| `cb_egress` | `egress_depth` | 2 | Lets the writer drain block `n` to DRAM scratch while compute produces block `n+1`. Load-bearing in stage S, which is the low-parallelism critical path. |
| `cb_grad_egress` | `egress_depth` | 2 | Same, for the six gradient outputs in stage G. |
| every block CB (`cb_*_block`, `cb_kmat_*`, `cb_vmat_*`, `cb_cc_*`, `cb_state_*`) | `block_depth` | 1 | These CBs **are** the block — the working set of one `(bh,chunk)` item held resident across ~20 phases. Depth 2 would double the op's L1 for pipelining that only pays when `items_per_core > 1`. Recorded as the **overlap** perf lamp below. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST capacity (`dest_helpers.hpp:103 DEST_AUTO_LIMIT`; 4 tiles at `fp32_dest_acc_en=true` half-sync, 8 at full-sync) | matmul **output subblock** `rt_dim·ct_dim`, *not* the block | implementer walks each block matmul in subblocks of ≤ `DEST_AUTO_LIMIT` tiles; `matmul_block` validates `ct_dim`/`rt_dim` ≤ 8/16 (`tt_metal/hw/inc/api/compute/matmul.h:189-193`) | silent DEST overrun / corrupted output tiles |
| Reduce accumulation depth in DEST | number of tiles folded into one `reduce()` call | use `ReduceInputPolicy::WaitAndPopPerTile` (the default) so the library chunks at `DEST_AUTO_LIMIT` | as above |
| L1 residency (see `l1_ledger.md` closed form) | `block_val_tiles` | largest divisor of `Vt` with `footprint_bytes(block_val_tiles) ≤ ttnn.get_max_worker_l1_unreserved_size()` | L1 allocation failure at program build (loud), or CB overlap (silent corruption) if the check is skipped |
| TILE granularity of `chunk_size` | `block_chunk_tiles` | `chunk_size % 32 == 0` enforced in `validate()`; matches the forward op's `TT_FATAL(chunk_size % 32 == 0)` | the `[C,C]` UT matrix would straddle a partial tile and the triangular masks would be wrong |
| Neumann doubling depth | number of squaring steps for `(I−A)^{-1}` | `neumann_steps = ceil(log2(chunk_size))` (5 for C=32, 6 for C=64), a CT constant derived from `chunk_size` | **wrong results**, not an error: too few steps truncates the Neumann series and silently drops the highest-order intra-chunk dependencies |
| Face-row span read | `gather_stage_tokens` | ≤ 32 (a source page holds one token; a destination tile holds 32) | the staging buffer would over- or under-fill a destination tile row |
| Semaphore count | per-core semaphores | 2 (`sem_prep_done`, `sem_scan_done`) | exceeding the per-core semaphore budget fails program build |
| `H ≤ 32` | head-tile count of the input layout | Phase 0 assumes `ceil(H/32) == 1` so the source page index is `((b·T + t)·Kt + kt)`; `validate()` rejects `H > 32` | wrong page indices → silently wrong gradients |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 — phased three-stage** (P: prep, grid-parallel over `(bh,chunk)` → per-group barrier → S: scan, one core per `bh` → per-group barrier → G: gradient assembly, grid-parallel over `(bh,chunk)`) | **built** | always (the default and only Phase 0 path) | `(block_chunks=1) × (Ct) × (Kt) × (block_val_tiles)`; state block `Kt × block_val_tiles` | **Minimum** = each of `q,k,v,do` crosses DRAM once (face-row granularity, 100% useful bytes) and each of the six outputs crosses once. R1 achieves exactly one face-row crossing per input and one per output. Above the minimum: compact head-major copies (`+1` write `+1` read per input, at full-page granularity) and the derived scratch (`Tinv, kcd, v_corr, u, c, S, dS, v_new, dv_new`), which is **structurally unreachable-minimum** — stages P/S/G have different core assignments, so every value crossing a stage boundary must cross DRAM. | Fixed cost amortized per block: the matmul/SFPU init + data-format reconfig at each of ~20 phase boundaries (measured ≈110–150 ns per reconfig, ≈320 ns per phase pass — `compute_block_size` perf entry), the ~25 CB handshakes, and the reader's per-block pipeline fill. Intended frequency: **once per `(bh,chunk)` block** for every init and reconfig; **once per chunk** for the scan-stage state matmuls; **twice per program** for the barrier. Raising `block_chunks` above 1 would amortize the inits over several chunks — blocked by L1 at the largest shape (see the perf lamp). |
| **R2 — fused single-stage** (one core per `bh`; each core does prep + both scans + assembly for all its chunks, no barriers, no prep scratch) | **deferred** — R1 already covers every shape R2 would serve, correctly and at ≥ R1's parallelism. R2's only advantage is on `NC == 1` shapes (5 of 22 INPUTS entries), where it saves the scratch round-trip of a *single* chunk; R1 there is already one core per `bh`, so the saving is bounded by one chunk's scratch bytes. Reachable because R1's per-chunk compute phases are identical — R2 is R1 with the two barriers removed and the scratch CBs kept in L1. | `NC == 1` | same block shape as R1 | Saves the derived-scratch round trip (≈ 9 blocks written + read per chunk). Still one face-row crossing per input. | Same fixed costs as R1, minus the two barriers. |
| **R3 — de-interleave stage 0** (a leading grid-parallel pass that reads whole source pages, harvests **all `H` heads** per page, and writes head-major compact copies; stages P/S/G then read only compact pages) | **deferred** — R1 covers every shape correctly; R3 is a pure data-movement optimization whose payoff is a factor of `H` on the gather term only. Reachable because R1 already materializes head-major compact copies (`sc_q/sc_k/sc_v/sc_do`) with exactly the layout R3 would produce; R3 moves *who* writes them and adds one more group of barriers. | always applicable; worth it when `H ≥ 4` | page-window block `gather_stage_tokens × Dt` | Replaces `H` separate 272-element span reads per (token, d-tile) with **one** `(256 + 16·H)`-element span read serving all `H` heads: gather bytes ÷ `H`, gather transactions ÷ `H`. Measured-analogue: `double_buffer` entry — a single core retires ~8–9 M NoC transactions/s regardless of depth, so transaction count is the governing term here. | Nothing new; it moves the same fixed costs to a stage with `B·Tt` work items instead of `BH·NC`. |
| **R4 — V-split across cores** (stage S factorizes exactly by V-column, as the shipping forward op's `distribute_scan` does; each core owns `(bh, v_block)`) | **deferred** — R1 covers every shape; R4's unique domain is shapes where `BH·NC` under-fills the grid *and* `Vt > 1`, and on those shapes the four V-reducing gradients (`dq, dk, dg, dbeta`) would need a per-chunk cross-core sum of `[C,C] + 2·[C,K] +` scalars. Reachable because `block_val_tiles` **already exists** as an extent knob and the V-loop is already the outer loop of stages S and G — R4 is a core-assignment change on an existing extent plus a combine. | `BH·NC < grid_area` and `Vt > 1` | `Kt × block_val_tiles` state per core | Adds cross-core traffic: per `(bh,chunk)`, `(Ct² + 2·Ct·Kt + 2·Ct)` tiles summed across `num_v_blocks` cores. Saves nothing on DRAM. | Fewer combine rounds per core. |
| **R5 — parallel prefix scan over chunks** (the reverse recurrence is linear: `dS_i = (Γ_i·I − kcd_iᵀ P_i)·dS_{i+1} + (c_i − kcd_iᵀ u_i)`, so a Blelloch scan over `[K,K]` transition matrices is legal) | **deferred** — R1 covers every shape; the scan is 4 matmuls out of the ~25 per chunk, so the Amdahl ceiling is small, and the composition cost is `K³` per merge versus `K²V` per sequential step (comparable at `K=128, V=256`, worse at `V ≫ K`). The task brief also pins this out of Phase 0 explicitly. Reachable because stage S is already a separate stage with its own core assignment and its own scratch contract. | `NC ≫ BH` and `K ≤ V` | `[K,K]` transition block | Adds `O(NC·log NC)` `[K,K]` blocks of cross-core traffic. | Fewer merge levels. |
| **R6 — per-core re-stream of the padded inputs** (each `bh` core reads whole `[32 heads × 32 dims]` pages once per pass, using `1/32` of each) | **rejected** — superseded by R1's face-row gather + compact copies. This is the **dead end**: it moves `32 ×` the logical input bytes on *every* pass, no good scheme passes through it, and nothing written for it survives when the compact-copy structure arrives. Costed here so the choice is visible: at `(1,256,4,128,256)` it moves 84 MB where R1 moves 2.6 MB of useful gather traffic. | — | — | `32 ×` logical per pass | — |

**Selection predicate (Phase 0).** Exactly one regime is built, so the program-descriptor builder has
no regime branch. The only host-side solve is the extent solve for `block_val_tiles` (closed form in
`l1_ledger.md`). Because the *extent* varies with shape, the design requires **extent-pinned tests**:
the acceptance suite includes `(1,256,4,128,256)`-class geometry so the `num_v_blocks > 1` path is
exercised, not only the `num_v_blocks == 1` fast path.

### Traffic ranking

All candidate splits, ranked by aggregate movement cost, tier by tier. `L` denotes the total logical
bytes of `q+k+v+do`; the expensive tier here is **DRAM at face-row granularity** (~64–128 useful bytes
per NoC transaction), not DRAM bandwidth.

| # | Candidate split | DRAM crossings of `q,k,v,do` | Gather transactions | Cross-core traffic | Occupancy | Verdict |
|---|---|---|---|---|---|---|
| 1 | **(bh, chunk) for P and G, (bh) for S** — R1 | 1 face-row read + 1 compact write + 1 compact read | `C·Dt` span reads per `(bh,chunk)` item, **once** | 2 semaphore round trips per `bh` group (no payload) | `min(grid, BH·NC)` in P and G, `BH` in S | **chosen** |
| 2 | (bh) for everything, no compact copies — R2 at `NC>1` | 2 face-row reads (prep pass and assembly pass) | `2 ×` #1 | none | `BH` | loses on both traffic *and* occupancy |
| 3 | (b, chunk) covering all `H` heads | 1 whole-page read, `32/H ×` bytes | `C·Dt` page reads (÷`H` vs #1) | none | `B·NC` | cheaper in transactions, but `H ×` the compute per core and `H ×` less occupancy; only reachable as a *separate* stage → R3 |
| 4 | (bh, v_block), combine over V — R4 | same as #1 | same as #1 | `(Ct² + 2·Ct·Kt + 2·Ct)` tiles per chunk, `×` fan-in | `BH·NV` | adds real cross-core payload for parallelism R1 already has via `chunk` |
| 5 | parallel scan over chunks — R5 | same as #1 | same as #1 | `O(NC·log NC)` `[K,K]` blocks | `min(grid, BH·NC)` in S too | largest cross-core payload; smallest Amdahl share |
| 6 | (bh), whole-page per-head reads — R6 | `32 ×` logical, **per pass** | `C·Dt` per pass | none | `BH` | dead end |

The ranking is decided by the **gather term**, which dominates every other term by an order of
magnitude at every INPUTS shape. #1 is the unique candidate that pays it **once**, at 100%-useful-byte
granularity, with the widest work-item space. Occupancy is a tiebreaker here, not the argument: #1
beats #2 on bytes before occupancy is considered at all.

Full per-tensor crossing counts are in `l1_ledger.md`.

### Block schedule

A **logical** schedule — reader, compute and writer are three asynchronous kernels, and the three
stages are three sections of the same three binaries selected by runtime-arg item counts.

```cpp
// ---- Stage P : prep.  block = one (bh, chunk) item.  spread over min(grid, BH*NC) cores ----
build_constant_tiles();                                   // once per core, at boot
for (uint32_t p = 0; p < core_num_prep_items; ++p) {
    gather_chunk_inputs(p);            // reader: face-row gather + tail zero-fill + compact copy-out
    chunk_decay_block(p);              // decay = LT_ones @ g ; gamma, Gamma, w
    decay_mask_block(p);               // L = tril( exp(decay (x) 1 - 1 (x) decay) )
    ut_matrix_block(p);                // A = -((k_beta @ k^T) . L) . strict_tril
    ut_inverse_block(p);               // Tinv = Neumann doubling, neumann_steps squarings
    chunk_prep_products_block(p);      // kcd, v_corr, Q, P, U
    intra_attn_block(p);               // intra = (q~ @ k^T . L) . tril_incl_diag
    scan_seed_block(p);                // u = intra^T @ do ;  c = Q^T @ do
    store_prep_block(p);               // writer: 6 scratch blocks -> DRAM
    signal_scan_owner(p);              // writer: sem_prep_done += 1 on this item's scan core
}

// ---- Barrier 1 : per-(bh) group fan-in.  scan core waits for NC increments. ----

// ---- Stage S : scan.  block = one chunk step of a [Kt x block_val_tiles] state.  BH cores ----
if (core_is_scan_owner) {
  for (uint32_t vb = 0; vb < num_v_blocks; ++vb) {
    load_initial_state(vb);                               // S = h0[:,vb] or 0
    for (uint32_t i = 0; i < tensor_chunks; ++i)
        state_forward_step(i, vb);                        // v_new_i ; store S_i, v_new_i ; S <- Gamma*S + P^T v_new
    load_final_state_grad(vb);                            // dS = dht[:,vb] or 0
    for (int32_t i = tensor_chunks - 1; i >= 0; --i)
        state_reverse_step(i, vb);                        // dv_new_i ; store dS_{i+1}, dv_new_i ; dS <- ...
    store_dh0_block(vb);                                  // if has_h0
  }
  release_group();                                        // writer: sem_scan_done += 1 on each G core of this group
}

// ---- Barrier 2 : per-group release.  each core waits for its group count. ----

// ---- Stage G : gradient assembly.  block = one (bh, chunk) item, V-accumulated ----
for (uint32_t p = 0; p < core_num_grad_items; ++p) {
    load_grad_block_invariants(p);     // reader: q, k, Tinv, decay-vector  (V-independent)
    decay_mask_block(p);               // L recomputed from decay (no scratch tensor for L)
    zero_grad_accumulators(p);         // dq, dk, M, d_attn, dU, dgamma, dw, dGamma
    for (uint32_t vb = 0; vb < num_v_blocks; ++vb) {
        load_grad_block_vslice(p, vb); // reader: v, do, v_new, dv_new, S_i, dS_{i+1} for this V block
        state_grad_products_block(p, vb);   // dQ, dP, d_kcd, dGamma
        output_attn_grad_block(p, vb);      // Mraw, Mt, M
        ut_backward_block(p, vb);           // dv_beta, dU, d_attn, dA
        value_grad_block(p, vb);            // dv  -> writer   (the only V-local output)
    }
    key_grad_block(p);                 // dk  -> writer
    query_grad_block(p);               // dq  -> writer
    beta_grad_block(p);                // dbeta -> writer
    decay_grad_block(p);               // dL, R, d_decay (+ the C-1 correction)
    gate_grad_scan_block(p);           // dg = UT_ones @ d_decay  (reverse cumsum)
    scatter_grad_block(p);             // writer: face-row scatter into the six outputs
}
```

Per-operation contract:

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `build_constant_tiles` | `4 × Ct²` tiles | the four constant tiles, for the whole program | **once per core, at boot** |
| `gather_chunk_inputs` | `Ct·Kt` (×2) + `Ct·Vb` (×2) + `2·Ct` | gathered blocks stay resident for all of stage P | one reader init + `gather_depth` pipeline fill per item |
| `chunk_decay_block` … `scan_seed_block` | see Block Operation Realization | `decay/γ/w/β`, `L`, `Tinv`, `k`, `q̃` stay resident across all of stage P | **one** matmul init + one reconfig per operation per block (not per tile) |
| `ut_inverse_block` | `Ct × Ct`, `neumann_steps` squaring pairs | `Tinv` accumulator and the running `A^{2^j}` | one matmul init for the whole doubling loop; `2·neumann_steps` block matmuls |
| `state_forward_step` / `state_reverse_step` | `Kt × Vb` state, `Ct × Vb` and `Ct × Kt` operands | the state block `S` (resp. `dS`) stays resident across **all** `tensor_chunks` steps — it is never packed out to DRAM between steps | one matmul init per *scan direction*, not per chunk; 2 block matmuls per chunk |
| stage-G operations | as tabled below | `dq/dk/M/d_attn/dU/dγ/dw/dΓ` accumulators stay resident across the `num_v_blocks` loop | one init per operation per block; the V-loop re-enters the same inits |
| barriers | — | — | **exactly twice per program**, not per chunk and not per V block |

### Provenance of the perf figures quoted below

`ttnn/ttnn/operations/examples/master.md` — the catalog of measured on-device kernel-performance
patterns — **is not present in this working tree** (`ttnn/ttnn/operations/examples/` does not exist on
this branch). Every knob in this design is therefore argued from structure: transaction counts, byte
counts, tier crossings and L1 residency, all derived above from this op's own shapes. Where a catalog
entry name appears below (`double_buffer`, `compute_block_size`, `matmul_output_subblock`,
`noc_placement`, `eltwise_l1_vs_dest_accumulate`) it is a **pointer for whoever has the branch**, and
the numbers attached to it are catalog-reported rather than measured here. Treat them the way a perf
lamp should be treated: as a reason to measure the named alternative, never as a measured fact about
this op.

Which entry informed which knob:

| Knob | Catalog pointer | What the pointer suggests |
|---|---|---|
| `gather_depth = 2`, `egress_depth = 2` | `double_buffer` | 4–8 outstanding reads per barrier with CB depth 2 is the reported sweet spot; a single core is transaction-rate-bound at ~8–9 M transactions/s, which is the term that dominates this op's gather |
| `block_chunks`, reconfig elision | `compute_block_size` | reported ~1.6× from 1→8 rows per block on a 5-phase chain, and ~110–150 ns per data-format reconfig; this op has ~20 phase boundaries |
| matmul subblock walk, `fp32_dest_acc_en` | `matmul_output_subblock` | the win tracks subblock *size*, not shape; `fp32_dest_acc_en` halves the DEST budget to 4 tiles and the reported win from 1.46× to 1.40× — decide the DEST width first, then derive every subblock |
| `row_wise=True` in `split_work_to_cores` | `noc_placement` | spreading along the DRAM-facing axis rather than down a column |
| state residency across scan steps | `eltwise_l1_vs_dest_accumulate` | the realistic pattern when DEST is busy (it is — every scan step is a matmul) is a packer-L1-accumulated L1-resident accumulator, not a DEST-camped one; this design keeps `cb_state_a`/`cb_state_b` L1-resident across all `tensor_chunks` steps for exactly that reason |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** | Every block CB is depth 1 because the block *is* the per-chunk working set and the op is already L1-tight at the largest shape. When `items_per_core > 1` (`BH·NC > grid_area`, reachable at `(1,512,2,64,64)` on a small grid) the reader cannot run ahead into item `n+1`. | Depth 2 on `cb_q_block`/`cb_k_block`/`cb_v_block`/`cb_do_block` only, funded by dropping `block_val_tiles` one step. `double_buffer` measured 1.74 → 2.78× for exactly this shape of change on a DRAM-fed reader. |
| **Overlap (block_chunks)** | `block_chunks = 1` is the coarsest that fits at the largest shape, but at `K=V=32, C=32` the per-chunk working set is ~1/8 of L1 and the ~20 per-phase inits are paid once per chunk anyway. | `block_chunks = 2` or `4` at small `K·V`. `compute_block_size` measured 1.00 → 1.64× going from 1 to 8 rows per block on a 5-phase chain; this op has ~20 phases, which is the regime where the lever is largest. |
| **Grid synchronization** | Stage P and G spread over `min(grid, BH·NC)` cores, which is the maximum-participation split. At `BH·NC = 1` (`(1,32,1,32,32)`) the two barriers are pure overhead on a single core. | Compare against a build with the barriers compiled out for `BH·NC == 1` — i.e. measure regime R2's predicate. |
| **Gather granularity** | The reader's default is a **272-element span read** per (token, d-tile) (one NoC transaction covering both faces of the wanted row, ~12% useful bytes for fp32) plus two 64-byte L1 re-packs. The alternatives are two 64-byte DRAM reads (100% useful, 2 transactions) and a whole-page read (4096 B, 1 transaction, 3% useful). | Sweep the three. Analytically: span ≈ 1.16 GB/s useful, two-run ≈ 0.58 GB/s, whole-page ≈ 0.66 GB/s per core, all transaction-rate-bound at ~110 ns/transaction (`double_buffer` measured 8–9 M transactions/s/core). This is the op's dominant cost, so it is the first thing to measure. |
| **fp32 DEST** | `fp32_dest_acc_en = True` is the default, inherited from the forward op (`chunk_gated_delta_rule.cpp:267-273`) and motivated by `dg` being the weakest gradient. It halves the DEST budget to 4 tiles, which caps every matmul subblock at 4 and (per `matmul_output_subblock`) cuts the subblock win from 1.46× to 1.40×. | Measure `fp32_dest_acc_en = False` against the `dg` PCC gate at bfloat16 only. Never override a caller-supplied `compute_kernel_config`. |
| **Reconfig elision** | ~20 phase boundaries per block, many of them fp32→fp32. `compute_block_size`'s reconfig ablation measured 110–150 ns per reconfig and 1.19× total on a 40-call chain. | Elide `reconfig_data_format` on the fp32→fp32 boundaries only; keep it on every boundary that touches an input-dtype CB (`cb_q_block` etc. are `bfloat16` when the inputs are). Silent corruption if elided across a real format change. |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `q,k,v,do` DRAM → L1 (stage P) | TILE, input dtype, `[B,T,32,D]` padded | **face-row span gather.** For head `h` and token `t`, the wanted row of source page `(b,t,0,dt)` spans both faces: `start_elems = (h/16)·512 + (h%16)·16`, `span_elems = 272`. One `noc_async_read` per `(token, d_tile)` into `cb_gather_stage`; after the barrier the reader re-packs the two 16-element runs per token into the destination tile's face rows with plain RISC-V stores. Page index `= ((b·T + t)·Kt + dt)` (`ceil(H/32)==1`). | The dominant cost of the op. `gather_stage_tokens = 32` tokens per staging window = exactly one destination tile row. `TensorAccessor::get_noc_addr(page_id, offset)` supplies the address. |
| `g,beta` DRAM → L1 (stage P) | TILE, input dtype, `[B,32·Tt,32]` padded | **column gather.** Head `h` is column `h` of page `(b, t/32, 0)`. One 4-byte (fp32) / 2-byte (bf16) read per token into the destination tile's column 0. `C` reads per chunk — small, and parallel over `BH·NC` cores. | The two vectors land in `cb_gate_block` as two column-tiles. |
| Padded tail | — | The reader zero-fills destination rows `t ≥ T` in every gathered block, including `g` and `beta`. | Makes every padded-row property in the table above hold by construction. |
| compact copy-out (stage P reader) | TILE, input dtype | The reader `noc_async_write`s the freshly assembled block from its own (still unpushed) `cb_*_block` reservation to `sc_q/sc_k/sc_v/sc_do`, then pushes to compute. One producer, one consumer — the DRAM write is the producer reading back its own reservation, not a second CB consumer. | Turns stage G's input read into a full-page compact read. |
| derived scratch (stage P/S compute → DRAM) | TILE, **Float32 always** | Compute packs each block into `cb_egress`; the writer pops and routes to `sc_attn / sc_kcd / sc_vcorr / sc_u / sc_c / sc_S / sc_dS / sc_vnew / sc_dvnew / sc_vec` in a compile-time-fixed order. | `fp32` regardless of input dtype: `Tinv`, the decay exponentials and the state carry the op's numerical risk, and `dg` is the weakest gradient. Matches the forward op's format plan (`chunk_gdn_phased_program_factory.cpp` `namespace pcb`). |
| Barrier 1 (P → S) | — | **Per-`(bh)` group fan-in.** Each core, after finishing *all* its prep items, sends `noc_semaphore_inc(+1)` to the `sem_prep_done` semaphore of the scan core of each group it produced for, once per item. The scan core waits for `sem_prep_done == tensor_chunks`. No global rendezvous, no multicast. | Deadlock-free by construction: **all** prep is issued before **any** wait. A core that owns items in several groups has already sent every increment before it blocks. |
| Barrier 2 (S → G) | — | **Per-group release.** After its scan, the scan core sends `noc_semaphore_inc(+1)` to `sem_scan_done` on every core that has stage-G work in its group (unicast list, `≤ grid_area` targets, computed on host and passed as runtime args). Each core then waits for `sem_scan_done == core_num_release_groups` before starting **any** stage-G item. | `core_num_release_groups` = number of distinct `bh` groups this core has stage-G items in; a host-computed runtime arg. Cores with no stage-G work wait for 0. |
| stage G inputs DRAM → L1 | TILE, mixed | Full-page reads from the compact scratch and from `sc_S/sc_dS`. No face-row gather in stage G. | This is what the compact copies bought. |
| outputs L1 → DRAM (stage G) | TILE, input dtype | **face-row scatter**, the mirror of the gather: two 16-element `noc_async_write`s per `(token, d_tile)` for `dq/dk/dv`; one scalar write per token for `dg/dbeta` into column `h`. `dh0` is written by stage S as whole pages. | Only rows `t < T` are written; padded rows of the output tensors are never touched and are sliced off by `to_torch`. |
| Sharded placement | — | Not in TARGET (`memory_layout` is INTERLEAVED-only for this op). If it is ever added, the logical shards are already named: a `(bh)`-flavoured shard is an **independent** cut (knob-turn — bind `cb_state_*` to the shard with `ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, no NoC re-read); a `V`-flavoured shard is the **R4** cut and is a scheme-change because four gradients reduce over V. | Stated per scheme so the classification does not have to be rediscovered. |

---

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = one `(bh, chunk)` item in stages P and G; one `(bh)` scan in stage S |
| Grid | `grid = device.compute_with_storage_grid_size()`; `num_cores = min(grid.x·grid.y, BH·NC)` |
| Per-core work | `ttnn.split_work_to_cores(grid, BH·NC, row_wise=True)` over the flattened item index `wi = bh·NC + i`. `row_wise=True` spreads along the DRAM-facing axis (`noc_placement` measured ~2.9× against the column line). |
| Scan-core assignment | the scan owner of group `bh` is the core that holds item `wi = bh·NC + 0`. Guarantees `BH ≤ num_cores` whenever `NC ≥ 1`, and puts the scan on a core that already holds chunk 0's prep. |
| Remainder | `split_work_to_cores` returns two groups; the first `rem` cores get one extra item. The host asserts `Σ items == BH·NC`. |
| Tile geometry | `Kt = ceil_div(K, 32)`, `Vt = ceil_div(V, 32)`, `Ct = chunk_size // 32` (exact — `chunk_size % 32 == 0` is validated), `NC = ceil_div(T, chunk_size)`, `Tt = ceil_div(T, 32)`, `Htile = ceil_div(H, 32)`. Every count uses `ceil`, per image — `T` is not tile-aligned on 6 of 22 INPUTS entries, and `H` is never tile-aligned. |
| Runtime args per core | `[base addresses…, prep_item_start, core_num_prep_items, is_scan_owner, scan_bh, grad_item_start, core_num_grad_items, core_num_release_groups, release_target_count, release_target_coords…]` |

Only one regime is built, so there is no regime-selection predicate. The **extent** (`block_val_tiles`)
does vary with shape; see the extent-pinned-test requirement in the Regimes section.

---

## Circular Buffers

`Ct = block_chunk_tiles`, `Kt = block_key_tiles`, `Vb = block_val_tiles`. `in_dtype` = the input dtype;
`f32` = `Float32`. `D_e = egress_depth = 2`, `D_g = gather_depth = 2`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_const_lt_ones` | 0 | `tile_size(f32)` | `Ct·Ct` | live set spans the `(C,C)` block; constant in every other axis | f32 | reader | compute | whole program |
| `cb_const_strict_tril` | 1 | `tile_size(f32)` | `Ct·Ct` | as above | f32 | reader | compute | whole program |
| `cb_const_ut_ones` | 2 | `tile_size(f32)` | `Ct·Ct` | as above; used only by `gate_grad_scan_block` | f32 | reader | compute | whole program |
| `cb_const_eye` | 3 | `tile_size(f32)` | `Ct·Ct` | as above; used only by `ut_inverse_block` | f32 | reader | compute | whole program |
| `cb_reduce_scaler` | 4 | `tile_size(bf16)` | 1 | a packed scaler tile — constant in every axis | bf16 | reader | compute | whole program |
| `cb_gather_stage` | 5 | `gather_stage_tokens · row_span_bytes` | `D_g` | reader-local staging for the face-row span read; live set is one token window, streams over every other axis | raw bytes | reader | reader | stage P |
| `cb_q_block` | 6 | `tile_size(in_dtype)` | `Ct·Kt` | spans `(C,K)`; streams over chunk | in_dtype | reader | compute | stages P, G |
| `cb_k_block` | 7 | `tile_size(in_dtype)` | `Ct·Kt` | as above | in_dtype | reader | compute | stages P, G |
| `cb_v_block` | 8 | `tile_size(in_dtype)` | `Ct·Vb` | spans `(C, block_val_tiles)`; streams over chunk and over the V-block loop | in_dtype | reader | compute | stages P, G |
| `cb_do_block` | 9 | `tile_size(in_dtype)` | `Ct·Vb` | as above | in_dtype | reader | compute | stages P, G |
| `cb_gate_block` | 10 | `tile_size(in_dtype)` | `2·Ct` | two column-tiles (`g`, `β`); spans `C`, constant in `K`/`V` | in_dtype | reader | compute | stages P, G |
| `cb_decay_block` | 11 | `tile_size(f32)` | `4·Ct` | four column-tiles (`decay`, `γ`, `w`, `Γ`-broadcast); spans `C` | f32 | compute | compute | stages P, G (in place) |
| `cb_kmat_a` | 12 | `tile_size(f32)` | `Ct·Kt` | `kβ` → `U` → `P`; spans `(C,K)` | f32 | compute | compute | stages P, S, G |
| `cb_kmat_b` | 13 | `tile_size(f32)` | `Ct·Kt` | `kcd`; spans `(C,K)` | f32 | compute | compute | stages P, S, G |
| `cb_kmat_c` | 14 | `tile_size(f32)` | `Ct·Kt` | `Q` / `dQ` / `dP` / `d_kcd` scratch; spans `(C,K)` | f32 | compute | compute | stages P, G |
| `cb_kmat_d` | 15 | `tile_size(f32)` | `Ct·Kt` | `dU` and the `dk` V-accumulator; spans `(C,K)` | f32 | compute | compute | stage G |
| `cb_vmat_a` | 16 | `tile_size(f32)` | `Ct·Vb` | `vβ` → `v_corr` → `v_new`; spans `(C, Vb)` | f32 | compute | compute | stages P, S, G |
| `cb_vmat_b` | 17 | `tile_size(f32)` | `Ct·Vb` | `u` → `dv_new`; spans `(C, Vb)` | f32 | compute | compute | stages P, S, G |
| `cb_vmat_c` | 18 | `tile_size(f32)` | `Ct·Vb` | `dvβ`; spans `(C, Vb)` | f32 | compute | compute | stage G |
| `cb_cc_a` | 19 | `tile_size(f32)` | `Ct·Ct` | `Tinv`; spans `(C,C)` | f32 | compute | compute | stages P, G |
| `cb_cc_b` | 20 | `tile_size(f32)` | `Ct·Ct` | `L`; spans `(C,C)` | f32 | compute | compute | stages P, G |
| `cb_cc_c` | 21 | `tile_size(f32)` | `Ct·Ct` | `A` → `intra` → `M` → `d_attn` → `dA`; spans `(C,C)` | f32 | compute | compute | stages P, G |
| `cb_cc_d` | 22 | `tile_size(f32)` | `Ct·Ct` | Neumann running power `A^{2^j}`; then `q̃kᵀ` / `kβkᵀ` in stage G; spans `(C,C)` | f32 | compute | compute | stages P, G |
| `cb_state_a` | 23 | `tile_size(f32)` | `Kt·Vb` | the running state `S` (stage S forward) / `S_i` (stage G); spans `(K, Vb)` | f32 | compute | compute | stages S, G |
| `cb_state_b` | 24 | `tile_size(f32)` | `Kt·Vb` | `dS` (stage S reverse) / `c_i` / `dS_{i+1}` (stage G); spans `(K, Vb)` | f32 | compute | compute | stages S, G |
| `cb_grad_vec` | 25 | `tile_size(f32)` | `4·Ct` | `d_decay`, `dγ`, `dw`, `dβ` column-tiles; spans `C` | f32 | compute | compute | stage G |
| `cb_egress` | 26 | `tile_size(f32)` | `D_e · max(Ct·Kt, Ct·Vb, Ct·Ct, Kt·Vb)` | one block in flight to the writer plus one being filled; sized to the **largest** block that crosses the boundary (`Kt·Vb`, the state) | f32 | compute | writer | stages P, S |
| `cb_grad_egress` | 27 | `tile_size(in_dtype)` | `D_e · max(Ct·Kt, Ct·Vb)` | as above, for the six gradient outputs; in_dtype because outputs match the input dtype | in_dtype | compute | writer | stage G |

Every CB has exactly one producer kernel and one consumer kernel. `cb_gather_stage` is reader→reader
(a reader-local scratch allocation; the generic-op model has no other L1 allocator) and is never seen
by compute or the writer. `cb_decay_block` and the working CBs `cb_kmat_*`/`cb_vmat_*`/`cb_cc_*` are
compute→compute in-place working buffers — Rule 3 pattern 2 — and no dataflow kernel reads them; the
writer never touches them, which is exactly why `cb_egress` exists.

---

## Block Operation Realization

`Ct`/`Kt`/`Vb` as above. "state" columns note only non-obvious CB lifetimes.

| # | Block operation | Block shape | Helper? | Input CB (name, pages, state) | Output CB (name, pages) | CB state after |
|---|---|---|---|---|---|---|
| 1 | `build_constant_tiles` | `4 × [Ct,Ct]` | no (raw L1 writes) | — | `cb_const_lt_ones/strict_tril/ut_ones/eye` (`Ct²` each) | pushed once, **never popped** — compute reads by index for the whole program |
| 2 | `gather_chunk_inputs` | `[C,K]×2, [C,Vb]×2, [C,1]×2` | no | DRAM via `TensorAccessor` → `cb_gather_stage` (`D_g`) | `cb_q_block`, `cb_k_block`, `cb_v_block`, `cb_do_block`, `cb_gate_block` | also `noc_async_write`s the compact copies to `sc_q/sc_k/sc_v/sc_do` before push |
| 3 | `chunk_decay_block` | `[C,C]@[C,32] → [C,1]` | no | `cb_const_lt_ones` (`Ct²`), `cb_gate_block` (`2Ct`) | `cb_decay_block` (`4Ct`) | `decay, γ=exp(decay), Γ=γ[C-1] broadcast, w=Γ/γ` |
| 4 | `decay_mask_block` | `[C,C]` | no | `cb_decay_block`, `cb_const_lt_ones` | `cb_cc_b` (`Ct²`) | `L` resident for all of the stage |
| 5 | `ut_matrix_block` | `[C,K]@[K,C] → [C,C]` | no | `cb_kmat_a` (`kβ`), `cb_k_block`, `cb_cc_b`, `cb_const_strict_tril` | `cb_cc_c` (`Ct²`) | `A` |
| 6 | `ut_inverse_block` | `neumann_steps × 2 × [C,C]@[C,C]` | no | `cb_cc_c` (`A`), `cb_const_eye` | `cb_cc_a` (`Ct²`), `cb_cc_d` scratch | `Tinv = Π_{j}(I + A^{2^j})`; `cb_cc_c` free for reuse |
| 7 | `chunk_prep_products_block` | `[C,C]@[C,K]`, `[C,C]@[C,Vb]` | no | `cb_cc_a`, `cb_kmat_a`, `cb_v_block`, `cb_decay_block`, `cb_gate_block` | `cb_kmat_b` (`kcd`), `cb_vmat_a` (`v_corr`), `cb_kmat_c` (`Q`), `cb_kmat_a` ← `P` (in place) | `U = kβ⊙γ` is re-derivable from `cb_k_block`,`cb_gate_block`,`cb_decay_block` in stage G |
| 8 | `intra_attn_block` | `[C,K]@[K,C] → [C,C]` | no | `cb_q_block`, `cb_k_block`, `cb_cc_b`, `cb_const_lt_ones` | `cb_cc_c` (`Ct²`) | `intra` |
| 9 | `scan_seed_block` | `[C,C]ᵀ@[C,Vb]`, `[C,K]ᵀ@[C,Vb]` | no | `cb_cc_c` (`intra`), `cb_kmat_c` (`Q`), `cb_do_block` | `cb_vmat_b` (`u`, `Ct·Vb`), `cb_state_b` (`c`, `Kt·Vb`) | |
| 10 | `store_prep_block` | 6 blocks | no | `cb_egress` (`D_e·max`) | DRAM `sc_attn, sc_kcd, sc_vcorr, sc_u, sc_c, sc_vec` | writer pops in a compile-time-fixed order |
| 11 | `signal_scan_owner` | — | no | — | semaphore | one `noc_semaphore_inc` per prep item |
| 12 | `load_initial_state` | `[K,Vb]` | no | DRAM `initial_state` or zero-fill | `cb_state_a` (`Kt·Vb`) | gated on `has_h0` |
| 13 | `state_forward_step` | `[C,K]@[K,Vb]`, `[K,C]@[C,Vb]` | no | `cb_kmat_b` (`kcd`), `cb_vmat_a` (`v_corr`), `cb_kmat_a` (`P`), `cb_state_a` | `cb_vmat_a` ← `v_new` (in place), `cb_state_a` ← `S_{i+1}` (in place), `cb_egress` ← `S_i`, `v_new_i` | `cb_state_a` **stays resident across all `tensor_chunks` steps** — never round-trips DRAM between steps |
| 14 | `load_final_state_grad` | `[K,Vb]` | no | DRAM `dht` or zero-fill | `cb_state_b` (`Kt·Vb`) | gated on `has_dht` |
| 15 | `state_reverse_step` | `[C,K]@[K,Vb]`, `[K,C]@[C,Vb]` | no | `cb_kmat_a` (`P`), `cb_vmat_b` (`u`), `cb_kmat_b` (`kcd`), `cb_state_b` (`c` then `dS`) | `cb_vmat_b` ← `dv_new` (in place), `cb_state_b` ← `dS_i` (in place), `cb_egress` ← `dS_{i+1}`, `dv_new_i` | same residency rule as 13 |
| 16 | `store_dh0_block` | `[K,Vb]` | no | `cb_egress` | DRAM `dh0` | gated on `has_h0` |
| 17 | `release_group` | — | no | — | semaphore | one `noc_semaphore_inc` per stage-G core in the group |
| 18 | `load_grad_block_invariants` | `[C,K]×2, [C,C], [C,1]×4` | no | DRAM `sc_q, sc_k, sc_attn, sc_vec` | `cb_q_block`, `cb_k_block`, `cb_cc_a`, `cb_decay_block`, `cb_gate_block` | full-page compact reads |
| 19 | `zero_grad_accumulators` | `[C,K]×2, [C,C]×2, [C,1]×4` | no | — | `cb_kmat_c`, `cb_kmat_d`, `cb_cc_c`, `cb_cc_d`, `cb_grad_vec` | live across the V-block loop |
| 20 | `load_grad_block_vslice` | `[C,Vb]×4, [K,Vb]×2` | no | DRAM `sc_v, sc_do, sc_vnew, sc_dvnew, sc_S, sc_dS` | `cb_v_block`, `cb_do_block`, `cb_vmat_a`, `cb_vmat_b`, `cb_state_a`, `cb_state_b` | |
| 21 | `state_grad_products_block` | `[C,Vb]@[Vb,K]` ×3, `[K,Vb]` reduce-scalar | **yes** (`reduce` for `dΓ`) | `cb_do_block`, `cb_vmat_a`, `cb_vmat_b`, `cb_state_a`, `cb_state_b`, `cb_reduce_scaler` | `cb_kmat_c` (`dQ`,`dP`,`d_kcd` accumulated), `cb_grad_vec` (`dΓ`) | `S_iᵀ`/`dSᵀ` via the matmul in1-transpose flag + tile-grid index swap |
| 22 | `output_attn_grad_block` | `[C,Vb]@[Vb,C] → [C,C]` | no | `cb_do_block`, `cb_vmat_a`, `cb_const_lt_ones`, `cb_cc_b` | `cb_cc_c` (`M` accumulated over V) | |
| 23 | `ut_backward_block` | `[C,C]ᵀ@[C,*]`, `[C,C]@[C,C]@[C,C]` | no | `cb_cc_a` (`Tinv`), `cb_vmat_b`, `cb_kmat_c`, `cb_v_block`, `cb_const_strict_tril` | `cb_vmat_c` (`dvβ`), `cb_kmat_d` (`dU`), `cb_cc_d` (`d_attn` → `dA`) | `dA = (Tinvᵀ·d_attn·Tinvᵀ)⊙strict_tril` — two `[C,C]` matmuls, no re-inversion |
| 24 | `value_grad_block` | `[C,Vb]` | no | `cb_vmat_c`, `cb_gate_block` | `cb_grad_egress` ← `dv` | the only V-local output; written inside the V loop |
| 25 | `key_grad_block` | `[C,C]@[C,K]` ×3 + elementwise | no | `cb_cc_c` (`M`), `cb_cc_d` (`dA`), `cb_cc_b` (`L`), `cb_q_block`, `cb_k_block`, `cb_kmat_c`, `cb_kmat_d`, `cb_decay_block`, `cb_gate_block` | `cb_grad_egress` ← `dk` | |
| 26 | `query_grad_block` | `[C,C]@[C,K]` + elementwise | no | `cb_cc_c` (`M`), `cb_k_block`, `cb_kmat_c` (`dQ`), `cb_decay_block` | `cb_grad_egress` ← `dq` | scaled by `scale` |
| 27 | `beta_grad_block` | `[C,K]`/`[C,Vb]` row-reduce | **yes** (`reduce`, `REDUCE_ROW`) | `cb_vmat_c`, `cb_v_block`, `cb_kmat_d`, `cb_k_block`, `cb_reduce_scaler` | `cb_grad_vec` (`dβ`) → `cb_grad_egress` | `rowsum_V` accumulates across the V loop |
| 28 | `decay_grad_block` | `[C,C]` row-reduce ×2, `[C,C]` transpose | **yes** (`reduce`, `REDUCE_ROW`) | `cb_cc_c`, `cb_cc_d`, `cb_cc_b`, `cb_q_block`, `cb_k_block`, `cb_grad_vec`, `cb_reduce_scaler` | `cb_grad_vec` (`d_decay`) | includes the `d_decay[C-1] += Σ dw·w + dΓ·Γ` correction |
| 29 | `gate_grad_scan_block` | `[C,C]@[C,32] → [C,1]` | no | `cb_const_ut_ones`, `cb_grad_vec` | `cb_grad_egress` ← `dg` | reverse cumsum as a single matmul |
| 30 | `scatter_grad_block` | `[C,K]×2, [C,Vb], [C,1]×2` | no | `cb_grad_egress` | DRAM `dq, dk, dv, dg, dbeta` | face-row scatter; rows `t ≥ T` never written |

---

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| every compute op (boot) | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h` | `SrcOrder::Reverse` because matmul maps in0→SrcB (`matmul.h:170-173`) | — | — | — |
| all block matmuls (3,5,6,7,8,9,13,15,21,22,23,25,26,29) | raw_api | `matmul_block_init(in0, in1, transpose, ct_dim, rt_dim, kt_dim)` / `matmul_block(in0, in1, i0, i1, idst, transpose, ct_dim, rt_dim, kt_dim)` | `tt_metal/hw/inc/api/compute/matmul.h:195-205` and `:243-256` | `transpose` transposes **B (in1) only** (`matmul.h:189`); `ct_dim`/`rt_dim` bounded by `DEST_AUTO_LIMIT` (`matmul.h:190-192`) | per row above | per row above | `ct_dim`, `rt_dim`, `kt_dim` are the **subblock** walk; the **block** extents are `Ct`, `Kt`, `Vb` |
| transposes of `[C,C]` (`Tinvᵀ`, `Rᵀ`, `Mᵀ`) | raw_api | `transpose_init(icb)` / `transpose_block(icb, start_itile, start_idst, ntiles)` | `tt_metal/hw/inc/api/compute/transpose.h:39` and `:164` | `ntiles` ≤ `DEST_AUTO_LIMIT` | `cb_cc_*` | `cb_cc_*` | `ntiles` = `Ct²` clamped to DEST |
| transposes of `[K,V] → [V,K]` and `[C,V] → [V,C]` inside a matmul (21, 22) | raw_api | `matmul_block(..., transpose = 1, ...)` with the tile-grid indices swapped by the caller | `tt_metal/hw/inc/api/compute/matmul.h:243-256` | the LLK flag transposes *within* the tile; the kernel swaps the `(row_tile, col_tile)` walk order | — | — | — |
| `dΓ = Σ_{K,V}(dS⊙S)` | **helper** | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, in_dfb, scaler_dfb, out_dfb>(ReduceInputBlockShape::of(Kt, Vb))` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:381-396`; `ReduceInputBlockShape::of` at `:140` | `ReduceInputPolicy::WaitAndPopPerTile` (default) chunks the input at `DEST_AUTO_LIMIT` | `cb_state_b`, `cb_reduce_scaler` | `cb_grad_vec` | `ReduceInputBlockShape::of(Kt, Vb)` — **both are block extents** |
| `rowsum_K`, `rowsum_V` (`dγ, dw, dβ`) | **helper** | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, …>(ReduceInputBlockShape::of(Ct, Kt or Vb), contiguous(), Accumulate::at(cb, iter))` | same, `:381-396`; `Accumulate` at `:200`-ish (`AccumulationConfig` `:152`) | the `Accumulate` path folds the running V-block partial without a separate add phase | `cb_kmat_*`/`cb_vmat_*`, `cb_reduce_scaler` | `cb_grad_vec` | `ReduceInputBlockShape::of(Ct, Kt)` / `of(Ct, Vb)`; the `Accumulate` iteration index is the V-block index |
| `rowsum(R)` and `rowsum(Rᵀ)` for `d_decay` | **helper** | `reduce<SUM, REDUCE_ROW, …>(ReduceInputBlockShape::of(Ct, Ct))` | `reduce_helpers_compute.hpp:381-396` | | `cb_cc_*`, `cb_reduce_scaler` | `cb_grad_vec` | `of(Ct, Ct)` |
| reduce scaler setup | **helper** | `compute_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` (and the `REDUCE_SCALAR` instantiation) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` (`prepare_reduce_scaler` / `calculate_and_prepare_reduce_scaler`) | **pool-type-aware overload**, as required — SUM uses scaler `1.0`; the packed tile is `bfloat16` | — | `cb_reduce_scaler` | — |
| `exp`, `recip` on `[C,1]` and `[C,C]` | raw_api | `exp_tile_init()/exp_tile(idst)`, `recip_tile_init()/recip_tile(idst)` | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`, `.../recip.h` | | `cb_decay_block`, `cb_cc_b` | same (in place) | — |
| broadcast multiplies (`⊙γ`, `⊙w`, `⊙β`, `L` construction) | raw_api | `mul_tiles_bcast<BroadcastType::COL>` / `sub_tiles_bcast<BroadcastType::ROW>` + their `_init` | `tt_metal/hw/inc/api/compute/bcast.h` | `COL` = operand valid in column 0, broadcast across columns; `ROW` = valid in row 0, broadcast down rows | see Broadcast Verification | | — |
| elementwise mul/add/sub on blocks | raw_api | `mul_tiles`, `add_tiles`, `sub_tiles` (+ `binary_op_init_common`) | `tt_metal/hw/inc/api/compute/eltwise_binary.h` | | | | tile count = block extent product |
| triangular masking | raw_api | `mul_tiles` against `cb_const_lt_ones` / `cb_const_strict_tril` | `tt_metal/hw/inc/api/compute/eltwise_binary.h` | a 0/1 mask multiply, applied block-wide | `cb_cc_*` | `cb_cc_*` | — |
| DEST budget query | **helper** | `compute_kernel_lib::DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103` | compile-time; reflects `fp32_dest_acc_en` | — | — | caps every subblock walk |
| reader/writer address generation | raw_api | `TensorAccessor` + `get_noc_addr(page_id, offset)` | `tech_reports/tensor_accessor/tensor_accessor.md` | `TensorAccessorArgs` as CT args; optional tensors get the no-arg placeholder | — | — | — |
| barriers | raw_api | `noc_semaphore_inc`, `noc_semaphore_wait_min` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | 2 semaphores per core | — | — | — |

### Helpers considered and rejected (one entry per `raw_api` block operation)

| Block operation | Helper considered | Concrete reason, with citation |
|---|---|---|
| every block matmul (6, 7, 9, 13, 15, 21, 22, 23, 25, 26, 29, 3, 5, 8) | `matmul_block_helpers.hpp::matmul_block()` | **The header does not exist in this working tree.** `ttnn/cpp/ttnn/kernel_lib/` contains exactly 16 files (`dest_helpers.hpp`, `dfb_helpers_*`, `l1_helpers.hpp`, `reduce_helpers_*`, `tilize_helpers.*`, `untilize_helpers.*`, `CMakeLists.txt`) and no matmul helper. The block-matmul operations are therefore built directly on `tt_metal/hw/inc/api/compute/matmul.h:243-256`, which is the layer the helper itself wraps. This is the "build the missing block operation" case from `blocking-model.md` §2, not a unit-at-a-time fallback: each call is a **block** matmul with `ct_dim`/`rt_dim`/`kt_dim` shaped from the block extents. |
| all elementwise / broadcast / mask / exp operations | `eltwise_convenience.hpp`, `eltwise_chain.hpp` | **Neither header exists in this working tree** (same 16-file listing). Falls back to `tt_metal/hw/inc/api/compute/eltwise_binary.h`, `bcast.h`, `eltwise_unary/*.h`. |
| barriers (11, 17) | `mcast_pipe.hpp`, `host/mcast_host.hpp` | **Neither exists in this working tree.** Also a poor fit even if present: the handshake here is a *fan-in counter* (`NC` unicast increments to one core) and a *fan-out unicast list*, not a payload multicast to a rectangle; `SenderPipe::send()` moves data and `McastRect` requires a rectangular receiver set, which the `split_work_to_cores` group layout does not guarantee. |
| `ut_inverse_block` (6) | any triangular-solve / inverse helper | A repo-wide search of `kernel_lib/` finds no `inverse`, `invert`, `triangular`, `substitution` or `solve` symbol. The Neumann-doubling formulation exists precisely so the operation becomes `2·neumann_steps` **block matmuls** rather than a `C`-step scalar forward substitution. |
| triangular masking (4, 5, 8, 22, 23) | `mask.h::mask_tile` | `mask_tile(idst_data, idst2_mask, ...)` (`tt_metal/hw/inc/api/compute/mask.h:43`) requires the mask tile to sit at `idst_data + 1` **inside DEST**, consuming a DEST slot per masked tile. With `fp32_dest_acc_en=true` the budget is 4 tiles, so pairing halves the matmul subblock for every phase that masks. A `mul_tiles` against a 0/1 constant CB costs no DEST slot and composes with the surrounding block matmul walk. |
| `chunk_decay_block` (3), `gate_grad_scan_block` (29) | `cumsum.h::cumsum_tile` | `cumsum_tile(idst, first)` is **columnwise within a tile** and, for `Ct > 1`, requires the tiles to arrive "in NWH order ... and *first* must be `false` for all tiles where H != 0" (`tt_metal/hw/inc/api/compute/cumsum.h:18-21`) — a per-tile-row ordering constraint on the DEST walk, and it has **no reverse** form for `dg`. The triangular-ones matmul gives forward cumsum (`LT_ones @ x`) and reverse cumsum (`UT_ones @ x`) with the *same* primitive, the same block shape, and no ordering constraint. |
| tilize / untilize | `tilize_helpers.hpp`, `untilize_helpers.hpp` | Not applicable: every tensor in this op is TILE in and TILE out (`feature_spec.py` TARGET is TILE-only). No row-major conversion occurs anywhere. |

---

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `decay_mask_block` (row term) | `sub_tiles_bcast` | `cb_decay_block[decay]` broadcast source: Col0 | transposed `decay` : Row0 | `ROW` |
| `decay_mask_block` (col term) | `sub_tiles_bcast` | `cb_decay_block[decay]` : Col0 | — | `COL` |
| `chunk_prep_products_block` (`U = kβ⊙γ`) | `mul_tiles_bcast` | `cb_kmat_a` : All `[C,K]` | `cb_decay_block[γ]` : Col0 | `COL` |
| `chunk_prep_products_block` (`P = k⊙w`) | `mul_tiles_bcast` | `cb_k_block` : All `[C,K]` | `cb_decay_block[w]` : Col0 | `COL` |
| `chunk_prep_products_block` (`kβ = k⊙β`, `vβ = v⊙β`) | `mul_tiles_bcast` | `cb_k_block` / `cb_v_block` : All | `cb_gate_block[β]` : Col0 | `COL` |
| `state_forward_step` (`Γ·S`) | `mul_tiles_bcast` | `cb_state_a` : All `[K,Vb]` | `cb_decay_block[Γ]` : `[0,0]` | `SCALAR` |
| `state_reverse_step` (`Γ·dS`) | `mul_tiles_bcast` | `cb_state_b` : All `[K,Vb]` | `cb_decay_block[Γ]` : `[0,0]` | `SCALAR` |
| `query_grad_block` (`dQ⊙γ`) | `mul_tiles_bcast` | `cb_kmat_c` : All `[C,K]` | `cb_decay_block[γ]` : Col0 | `COL` |
| `key_grad_block` (`dP⊙w`, `dkβ⊙β`, `dU⊙γ`) | `mul_tiles_bcast` | `cb_kmat_*` : All `[C,K]` | `cb_decay_block`/`cb_gate_block` : Col0 | `COL` |
| `value_grad_block` (`dvβ⊙β`) | `mul_tiles_bcast` | `cb_vmat_c` : All `[C,Vb]` | `cb_gate_block[β]` : Col0 | `COL` |
| `decay_grad_block` (`dγ⊙γ`, `dw⊙w`) | `mul_tiles` | `cb_grad_vec` : Col0 | `cb_decay_block` : Col0 | none (both column-shaped) |
| `beta_grad_block` / `decay_grad_block` reduce outputs | `reduce<REDUCE_ROW>` | input `[Ct, Kt]` or `[Ct, Vb]` : All | `cb_reduce_scaler` : `[0,0]` | output valid region: **Col0** |
| `state_grad_products_block` (`dΓ`) | `reduce<REDUCE_SCALAR>` | `[Kt, Vb]` : All | `cb_reduce_scaler` : `[0,0]` | output valid region: **`[0,0]`** |

---

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| **The `(H,K)` tiling of a `[B,T,H,K]` tensor** | A page is `[32 heads × 32 dims]` for one token, `H ≤ 8` on every INPUTS entry, so a naive per-head page read uses 3% of every byte it moves and a naive whole-tensor read moves `32/H ×` the logical size. An implementer who reads `q` "one tile at a time" will produce a correct but ~30× slow op. | The face-row span gather (Dataflow Strategy row 1) and the compact head-major copies `sc_q/sc_k/sc_v/sc_do`, which make it a **one-time** cost. Regime R6 is the rejected dead end; regime R3 is the deferred `H ×` improvement. |
| **Two barriers in one program** | Stages P/G and S have different core assignments, and a single dispatch is mandatory, so the stage boundary must be a semaphore rendezvous. A wait placed before a producer's increment deadlocks the device. | **All** prep items are issued before **any** wait; the barrier is per-`(bh)` group (fan-in `NC`, fan-out the group's G cores), never global; cores with no work in a group wait for 0. The deadlock argument is written out in the Dataflow Strategy rows for Barrier 1 and Barrier 2 and must be preserved verbatim by the implementer. |
| **`neumann_steps` is a correctness constant, not a tuning knob** | `(I−A)^{-1} = Π_{j=0}^{m-1}(I + A^{2^j})` is exact only for `2^m ≥ chunk_size`. Too few steps silently truncates the highest-order intra-chunk dependency and produces gradients that pass at `chunk_size = 32` and fail at 64. | `neumann_steps = ceil(log2(chunk_size))`, derived on host from `chunk_size` and passed as a compile-time arg. Listed in Mechanism caps. |
| **`dg` is the weakest gradient** (measured: bf16 min PCC 0.9984 host-side, and it is the only gradient through `exp(cumsum)`) | It is the *last* value in a long chain: `dL → R → rowsum/colsum → d_decay → reverse cumsum`. Every earlier rounding lands in it. | All decay/`L`/`Tinv`/state CBs are `Float32` regardless of input dtype; `fp32_dest_acc_en = True` by default; the reverse cumsum is a single matmul (one rounding) rather than an `O(C)` running sum. |
| **`d_decay[C-1]` at a padded tail** | The reflex is to mask every padded position out of every gradient. `d_decay[C-1]` is the exception: it carries `dΓ·Γ + Σ dw·w`, and the reverse cumsum propagates it into **real** `dg` rows. Zeroing it produces a `dg` that is subtly wrong only on `chunk_ragged` shapes. | Written out in the padded-tail table. Masking happens at the **output scatter** (rows `t ≥ T` are never written), not at `d_decay`. |
| **`dht` dropped** | The `do` path dominates every cartesian cell, so an implementation that never reads `dht` passes almost everything and fails only the `zero_do` regression. | `load_final_state_grad` is a named block operation with its own `has_dht` gate; the acceptance test includes a `dht`-only case at `g_scale = 0.01`. |
| **`dh0` must be `None`, not zeros** | A zero tensor type-checks downstream and hides a missing gradient path. | The host allocates five outputs when `initial_state is None`; `has_h0` is a compile-time arg so the two shapes are distinct programs. |
| **`|dh0| ≈ exp(Σg)·|dht|`** | At the default gate distribution `Σg ≈ −53` over 128 tokens, so `dh0 ≈ 1e-23` — numerically dead. A probe that seeds `dht` at the default decay and sees zeros is looking at correct behaviour. | Every `dht`-exercising test in the acceptance suite uses `g_scale ≤ 0.05`. Stated here so a debugger does not chase it. |
| **Matmul transpose is B-only** | `matmul.h:189` transposes **in1** only. Six products in the derivation need a transposed **in0** (`intraᵀ@do`, `Qᵀ@do`, `Tinvᵀ@·`, `Mᵀ@q̃`, `Wᵀ@kβ`, `kcdᵀ@dv_new`). | Each is re-associated as `(Bᵀ A)ᵀ` where possible, or materialized once with `transpose_block` (`transpose.h:164`) into a `cb_cc_*`/`cb_kmat_*` slot. The `[K,V]→[V,K]` cases (21) use the in1-transpose flag with a swapped tile-grid walk, never a materialized `[V,K]` copy — a materialized transpose of the state would cost another `Kt·Vb` CB. |
| **L1 at `K=128, V=256, C=64`** | The closed-form footprint at `block_val_tiles = Vt = 8` is ≈1.4 MB, over budget. Silently taking a smaller block "because it fits" would hide that this is the *only* shape class that needs it. | `block_val_tiles` is solved once on host from the `l1_ledger.md` closed form; `num_v_blocks > 1` is a real code path (the V-loop in stage G accumulates), and the acceptance suite pins a shape that exercises it. |
| **One dispatch, six outputs** | `generic_op` returns only `io_tensors.back()`. | All outputs are host-pre-allocated and passed in `io_tensors`; the return handle is discarded and the entry point returns its own tuple. |
| **Constants must not be a host op dispatch** | `ttnn.zeros`/`ttnn.eye` would each be a second dispatch. | The four constant tiles are generated **in the reader kernel** at boot with raw L1 stores (`build_constant_tiles`), pushed once and never popped. No host constant tensors at all — a tighter contract than the forward op, which uploads `eye/tril/ones/masks`. |
| **q/k must be L2-normalized** | With un-normalized `q/k` the UT transform is not contractive and the forward diverges (measured `|o|max = 2.9e18`). This is a **caller contract**, not something the op fixes. | Documented in the entry point's docstring; every test and probe builds inputs through `make_reference_inputs`. |

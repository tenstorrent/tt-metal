# Operation Design: chunk_gated_delta_rule_fwd

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + data movement, three stages, **one** `ttnn.generic_op` dispatch) |
| Goal | Training-side forward of the chunked gated delta rule (Gated DeltaNet linear attention). Returns the output `o`, the final recurrent state, and every intermediate a backward consumes: per-chunk entering states `h`, corrected values `v_new`, chunk-local decay cumsum `g_cumsum`, and the UT inverse `A`. |
| Math | See **Reference math** below; oracle `eval/golden_tests/chunk_gated_delta_rule_fwd/helpers.py::pytorch_chunk_gated_delta_rule_fwd` |
| Mode | Derivative (same math, I/O layout and precision surface as `gated_delta_net_backward`'s stages P and S) |
| References | `eval/golden_tests/chunk_gated_delta_rule_fwd/{helpers.py,feature_spec.py,test_regression.py}`; `ttnn/ttnn/operations/gated_delta_net_backward/{op_design.md,op_requirements.md,gated_delta_net_backward_program_descriptor.py,kernels/}` (same I/O layout; its reader/writer solve the `[B,T,H,D]` face-row gather/scatter and its op file documents the measured `decay` precision floor); `models/experimental/gated_attention_gated_deltanet/torch_functional/delta_rule_ops.py` (in-tree torch reference); `.claude/references/{blocking-model.md,l1-footprint-discipline.md,precision_convention.md}`; `l1_ledger.md` (this directory) |

### Reference math (per batch `b`, head `h`, chunk `i`; `C = chunk_size`)

With `q̃ = q·scale`, `kβ = k⊙β`, `vβ = v⊙β`, all per chunk:

```
decay[t]  = Σ_{u≤t} g[u]                    (chunk-local inclusive cumsum)   -> g_cumsum
γ[t]      = exp(decay[t]);   Γ = exp(Σ_u g[u]) = γ[C-1];   w[t] = exp(Σ_{u>t} g[u])
D[t,s]    = Σ_{s<u≤t} g[u]  = decay[t]-decay[s] for s≤t, 0 for s>t    (built from g, see Precision contract)
L         = exp(D) ⊙ LT                     (LT = inclusive-lower ones)
N         = (kβ @ kᵀ) ⊙ L ⊙ SL              (SL = strict-lower ones;  N = −A_strict)
Tinv      = (I + N)^{-1} = (I − N)·Π_{j=1}^{m-1}(I + N^{2^j}),  m = neumann_steps = ceil(log2 C)   -> A
nkcd      = −Tinv @ (kβ ⊙ γ)                [C,K]   (negated k_cumdecay)
v_corr    = Tinv @ vβ                       [C,V]
Q         = q̃ ⊙ γ                           [C,K]
intra     = (q̃ @ kᵀ) ⊙ L                    [C,C]
Pᵀ        = (k ⊙ w)ᵀ                        [K,C]
--- sequential over i (the only cross-chunk dependency) ---
h_i       = S                               (state ENTERING chunk i)                  -> h
v_new_i   = v_corr + nkcd @ h_i             [C,V]                                    -> v_new
S         = Γ·h_i + Pᵀ @ v_new_i            [K,V]                                    -> final_state (after last chunk)
--- state-independent given h_i, v_new_i (chunk-parallel again) ---
o_i       = Q @ h_i + intra @ v_new_i       [C,V]                                    -> o
```

`(I − N)` for the `j = 0` factor is exact: `(I + N)^{-1} = Σ_n (−N)^n` and every factor with `j ≥ 1`
is an even power, so only the first carries the sign. `N` is strictly lower, so `N^C = 0` and the
product is exact once `2^m ≥ C`.

**Three algebraic moves this design makes on purpose** (each is an equality, not an approximation):

| Move | Why |
|---|---|
| `o_i` is taken **off the scan**: it needs `h_i` and `v_new_i`, both of which the scan already emits as outputs, and nothing in the scan needs `o_i`. | The scan is the only sequential stage; `o`'s two matmuls and — far more important — its face-row output scatter (the op's most transaction-heavy write) move to a chunk-parallel stage E that fills the cores that would otherwise idle while the scan runs. This is the stall-shadow fill. |
| `nkcd = −kcd` and `Pᵀ` are materialized **in stage P** (negated and transposed there). | Stage S then needs no transpose and no negate: `v_new = v_corr + nkcd@h` is one DEST-accumulated matmul on top of `v_corr`, and `Pᵀ@v_new` feeds `in0` directly (matmul transposes `in1` only, `matmul.h:189`). |
| `D` is built as `LT @ diag(g) @ SL` rather than `decay ⊗ 1 − 1 ⊗ decay`. | Measured finding (`gated_delta_net_backward.py` module docstring; `op_requirements.md` Refinement 1): `decay` reaches `|≈250|` at `g_scale = 8`, and any FPU source register holding it (~tf32) loses the difference. `g` is small. |

### Padded tail (`T % C ≠ 0`)

The reader zero-fills rows `t ≥ T` of every gathered block (`q = k = v = 0`, `g = β = 0`).

| Quantity at a padded row | Value | Consequence |
|---|---|---|
| `kβ, vβ, Q, P, N` rows | 0 | padded tokens write nothing into the state |
| `Tinv` row | identity row | `v_corr`, `nkcd` rows are 0 ⇒ `v_new` row is 0 |
| `decay` | `decay[last real]` (g_pad = 0) | `Γ = exp(Σ real g)` — the correct chunk decay |
| `final_state` | state after the `T` real tokens | identity update, by construction |
| `o, v_new, A, g_cumsum` padded rows | **never written** | writers emit exactly `T` rows |
| `h[:, NC-1]` | the state entering the partial chunk | real output row; `h` has `NC` chunks |

---

### Dispatch contract

**Exactly one native device-program dispatch per public call.** Stages P, S, E live in one
`ProgramDescriptor` (three kernel binaries: reader, compute, writer) and are sequenced by
**segmented** per-group semaphore handoffs (Dataflow Strategy). Host work — validation, `scale`
default, output + DRAM-scratch pre-allocation (`ttnn.allocate_tensor_on_device`), the extent solves —
is not a dispatch. The entry point calls no TTNN op (`permute`, `to_layout`, `pad`, `slice`, `reshape`,
`zeros`, `matmul`, `cumsum`, …); the `[B,T,H,*] ↔ per-head` gather/scatter, the scale multiply, the
padded-tail handling and every constant tile are produced inside the kernels. `generic_op` returns only
the last `io_tensors` entry, so the entry point discards it and returns its own 6-tuple.

A single program provably fits L1 at every INPUTS shape (worst INPUTS case ≈ 1.18 MB at `Vi = Vt` against the ≈ 1.38 MB budget of the measured part,
`l1_ledger.md`), so the multi-dispatch escape hatch is never used.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `q` | `ttnn.Tensor [B,T,H,K]` | yes | TILE; L2-normalized along K (caller contract) | — | tensor |
| `k` | `ttnn.Tensor [B,T,H,K]` | yes | TILE; L2-normalized along K (caller contract) | — | tensor |
| `v` | `ttnn.Tensor [B,T,H,V]` | yes | TILE | — | tensor |
| `g` | `ttnn.Tensor [B,T,H]` | yes | TILE; log-space gate ≤ 0 | — | tensor |
| `beta` | `ttnn.Tensor [B,T,H]` | yes | TILE; in (0,1) | — | tensor |
| `initial_state` | `ttnn.Tensor [B,H,K,V]` or `None` | no (kw) | TILE | `None` (== zeros, **never read**) | tensor + `has_h0` CT flag |
| `chunk_size` | `int` | no (kw) | positive multiple of 32 (`{32,64}` in TARGET) | 64 | CT `Ct = chunk_size/32`, `neumann_steps` |
| `scale` | `float` | no (kw) | finite | `K ** -0.5` | RT (fp32 bit pattern), applied on device |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (kw) | any fidelity / approx; see Precision | `default_compute_kernel_config()` | program config (+ CT `dest_limit`) |
| `memory_config` | `ttnn.MemoryConfig` | no (kw) | interleaved DRAM or L1 | `q.memory_config()` | host (output placement only; scratch is always DRAM) |

Five tensors positional, the rest keyword-only; `g` before `beta`.

`default_compute_kernel_config()` — the ONE definition of `None`, exported from the package
`__init__` (the golden regression suite imports it): a fresh
`ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` per
call. The entry point resolves `None` through it and otherwise honors the caller's fidelity /
approx / `fp32_dest_acc_en` (DEST limit derived from it: 4 tiles at fp32 half-sync, 8 at 16-bit).
`float32` input with `fp32_dest_acc_en=False` is **natively rejected** with `ValueError`
(`precision_convention.md` — lossy and pointless; there is no `fp32_dest_acc_en` axis in this op's
TARGET, so it is a hard refusal, not an `EXCLUSIONS` entry). Internal CBs are `Float32` regardless.

### Validation (`validate()` is the entry point's first line)

| Check | Raises |
|---|---|
| `q,k,v` not rank 4; `g,beta` not rank 3; `initial_state` (if given) not rank 4 | `ValueError` |
| any `B/T/H/K/V` mismatch across `q,k,v,g,beta,initial_state` | `ValueError` |
| `chunk_size` not a positive multiple of 32; `K % 32` or `V % 32` ≠ 0 | `ValueError` |
| dtype or layout differs across the input tensors | `UnsupportedAxisValue` |
| `dtype ∉ SUPPORTED["dtype"]`, `layout ∉ SUPPORTED["layout"]`, any tagged axis ∉ SUPPORTED, then EXCLUSIONS | `UnsupportedAxisValue` / `ExcludedCell` |
| **mechanism cap** `H > 32` (page index assumes `ceil(H/32) == 1`) | `ValueError` (documented cap, not a support axis) |
| **mechanism cap** no `item_block_val_tiles` fits L1 (host solve, `l1_ledger.md`) | `ValueError` naming the footprint |
| `float32` + `fp32_dest_acc_en=False` | `ValueError` |

`state_mode` is kwarg-derived in `validate()` (`"no_h0"` iff `initial_state is None`); `dtype`/`layout`
are read off `q`. INPUT_TAGGERS, in this order, over `inputs = ((B,T,H,K,V), chunk_size)`:
`tag_chunk_size → int(inputs[1])`; `tag_seq_alignment → "chunk_aligned" iff T % axes["chunk_size"] == 0
else "chunk_ragged"`; `tag_head_dims → "square" iff K == V else "wide_v"`. The op file does not declare
INVALID (it is `[]` in `feature_spec.py`).

---

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `q,k [B,T,H,K]`; `v [B,T,H,V]`; `g,beta [B,T,H]`; `initial_state [B,H,K,V]` |
| Dtype | `float32` or `bfloat16`, uniform across all inputs |
| Layout | TILE |
| Memory | interleaved (DRAM or L1), read through `TensorAccessor` |

**The tiling fact that shapes the dataflow.** `q` is `[B,T,H,K]` tiled over `(H,K)`: a page is
`[32 heads × 32 dims]` for **one token**, and one head is **one row** of every page (16 elements in face
`2·(h/16)`, 16 in the next face, 256 elements apart). `g`/`beta` are `[B,T,H]` tiled over `(T,H)`: a page
is `[32 tokens × 32 heads]`, one head is one **column**. `initial_state`, `final_state` and `h` are tiled
over `(K,V)` — one head's state is a contiguous run of full pages.

### Output

| # | Name | Shape | Layout / tiling | Written by | Granularity |
|---|---|---|---|---|---|
| 0 | `o` | `[B,T,H,V]` | TILE over `(H,V)` | stage E | face-row scatter, rows `t < T` only |
| 1 | `final_state` | `[B,H,K,V]` | TILE over `(K,V)` | stage S | full pages |
| 2 | `h` | `[B,NC,H,K,V]` | TILE over `(K,V)`; page `((b·NC+i)·H+h)·Kt·Vt + kt·Vt + vt` | stage S | full pages; `h[:,0] = initial_state` or exact zeros |
| 3 | `v_new` | `[B,T,H,V]` | TILE over `(H,V)` | stage E | face-row scatter, rows `t < T` |
| 4 | `g_cumsum` | `[B,T,H]` | TILE over `(T,H)` | stage P | per-token scalar scatter into column `h` |
| 5 | `A` | `[B,T,H,C]` | TILE over `(H,C)`; row `r` of `Tinv_i` at token `i·C+r` | stage P | face-row scatter, rows `t < T` |

Every output: input dtype, TILE, placed per `memory_config` (default `q.memory_config()`). All six are
always allocated and returned; none is `None`.

### Phase 0 support intent (the implementer owns `SUPPORTED`)

| Axis (TARGET name) | TARGET | Design covers | Note |
|---|---|---|---|
| `dtype` | `float32, bfloat16` | both | internal CBs `Float32`; bf16 is a boundary-format difference only (in-dtype input / output CBs) |
| `layout` | `TILE` | `TILE` | whole universe |
| `state_mode` | `no_h0, with_h0` | both | `has_h0` CT flag; `no_h0` zero-fills the state in compute and binds no tensor |
| `chunk_size` | `32, 64` | both | `Ct ∈ {1,2}`; `neumann_steps` derived |
| `seq_alignment` | `chunk_aligned, chunk_ragged` | both | reader zero-fills tail; writers emit `T` rows |
| `head_dims` | `square, wide_v` | both | K and V are independent extents everywhere |

One code path serves the whole rectangle. EXCLUSIONS: none. If Phase 0 must narrow, drop `bfloat16`
(boundary-format refinement) — nothing else.

**Structural impossibilities.** None beyond `feature_spec.py`'s `INVALID = []`.

---

## Blocking Model

Symbols: `Ct = C/32`, `Kt = K/32`, `Vt = V/32`, `NC = ceil(T/C)`, `Tt = ceil(T/32)`, `BH = B·H`,
`NI = BH·NC` (items), `G = grid.x·grid.y` from `device.compute_with_storage_grid_size()`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` B | **independent** — no term couples batch elements | `block_batch` | 1 (flattened into `bh`) | derived from item index on host | flattened with `head`; spread over the grid in all three stages | — (already split) |
| `head` H | **independent** — each head owns its own state `S[K,V]` | `block_heads` | 1 — one head per item because a head is one row of a page; a multi-head block is the page-harvest regime R3, not an extent turn | host constant `BLOCK_HEADS = 1` → CT | as `batch` | scheme-change (R3) |
| `chunk` NC | **dependent** — `S` entering chunk `i+1` depends on chunk `i` | `block_chunks` | 1 | host constant `BLOCK_CHUNKS = 1` → CT | **split across cores in P and E** (state-independent); **walked sequentially in S** on the unit's core | scheme-change (R5 parallel scan) |
| `token-in-chunk` C | **dependent** — the UT transform couples every token of the chunk | `block_chunk_tiles` | `Ct` — the whole chunk, never a sub-chunk | `chunk_size` → CT `Ct` | never split | none (would need a cross-core block-triangular inverse) |
| `key_dim` K | **dependent** — contracted in `kβ@kᵀ`, `q̃@kᵀ`, `nkcd@h`; the state's row axis | `block_key_tiles` | `Kt` whole | `q.shape[-1]` → CT `Kt` | not split | scheme-change (a K-split needs a cross-core sum of every `[C,V]` on the critical path — see Traffic ranking #5) |
| `value_dim` V (scan) | **independent** — the state and every scan output factorize exactly by V column; nothing in the forward reduces over V | `scan_block_val_tiles` (`Vs`) | occupancy-first: `NV = Vt/Vs` = largest divisor of `Vt` with `BH·NV ≤ G` (1 if `BH ≥ G`) | host solve `_solve_scan_split` → CT `Vs`, derived `NV` | **split across cores in S**: one scan unit per `(bh, v_block)` | knob-turn (it is already split; mcast of the shared operands is R4) |
| `value_dim` V (items) | **independent** — `v_corr`, `o`, `v_new` are V-local | `item_block_val_tiles` (`Vi`) | largest divisor of `Vt` whose closed-form footprint fits L1 (= `Vt` at every INPUTS shape) | host solve `_solve_item_block_val_tiles` (closed form in `l1_ledger.md`) → CT `Vi`, derived `num_item_v_blocks = Vt/Vi` | looped inside the core (P, E) | knob-turn |
| `handoff segment` (implementation axis of the P→S and S→E rendezvous) | chunks grouped into readiness segments | `ready_segments` (`NS`) | `min(NC, 4)`; `seg_chunks = ceil(NC/NS)` | host constant `READY_SEGMENTS_MAX = 4` → CT `NS`, `seg_chunks` | n/a | knob-turn (≤ 8: two semaphores per segment, 16 per core) |
| `gather token window` | implementation axis of the face-row gather | `gather_stage_tokens` | 16 | host constant `GATHER_STAGE_TOKENS` → CT | reader-local | knob-turn |

Every axis has a row. The "not split" answers are decisions: `C` and `K` are the coupling / contraction
axes of the chunk algorithm, and splitting either buys nothing that the V split does not already buy
without a combine.

**Intermediate-stage axes (re-running the table on what each stage produces).**

| Stage output | Axes | Assignment |
|---|---|---|
| P → `nkcd, Pᵀ, Q, intra, v_corr, Γ` per item | `(bh, chunk, C, K or C or V)` | item-parallel across the grid (the same core that made them does not consume them, except E's `Q, intra`) |
| S → `h_i, v_new_i` per chunk, `final_state` | `(bh, chunk, K/C, V)` — **V stays split** | one scan unit per `(bh, v_block)`; the chunk walk is the one sequential axis |
| E → `o_i`, `v_new` output rows | `(bh, chunk, C, V)` — **chunk is independent again** | item-parallel on the P cores; this is where the would-be-idle cores go during the scan |

No stage computes on a designated core while its group waits: the "group owner" roles below are
communication roles (who signals whom), not work assignments. The only structurally sequential
work is the chunk walk of S, and it is spread over `BH·NV` cores.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_gather_stage` | `GATHER_DEPTH` | 2 | Overlaps the next token window's face-row DRAM reads with the RISC-V re-pack of the current one — the op's dominant input-side cost |
| `cb_kmat_in`, `cb_scan_pt`, `cb_scan_vcorr`, `cb_scan_gamma` | `SCAN_STREAM_DEPTH` | 2 | The scan's per-chunk operands do not depend on the state, so the reader prefetches chunk `i+1` while compute runs chunk `i`. This is the critical path's only data-movement term; depth 2 hides it completely |
| `cb_scratch_egress`, `cb_out_egress` | `EGRESS_DEPTH` | 2 | Writer drains block `n` (scratch, scatter, `h`) while compute packs block `n+1` |
| in-place-updated compute CBs (`cb_kb`, `cb_T`, `cb_pow`, `cb_state`) | `ACCUM_DEPTH` | 2 | **Not a tuning knob**: the only way a compute→compute CB can express `X ← f(X)` — reserve the new block behind the still-fronted old one — without hanging the pack/unpack threads (`gated_delta_net_backward/op_requirements.md` Refinement 4 note). Turning it to 1 is a hang |
| every other block CB (`cb_q_in`, `cb_k_in`, `cb_vblock_in`, `cb_gate_in`, `cb_L`, `cb_cc_*`, `cb_qs`, `cb_kw`, `cb_vmat`, `cb_intra_in`, `cb_vnew_in`, `cb_vec`, `cb_scan_vnew`) | `BLOCK_DEPTH` | 1 | These CBs **are** the per-item working set, resident across ~15 phases. Depth 2 doubles them for overlap that only pays when a core holds > 1 item → **overlap** perf lamp |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST: 4 tiles at `fp32_dest_acc_en` half-sync, 8 at 16-bit (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103` `DEST_AUTO_LIMIT`) | matmul output **subblock** (`rt_dim·ct_dim`), eltwise chain window — not the block | implementer walks every block matmul in subblocks ≤ `dest_limit` (CT arg derived from the resolved config); `matmul_block` requires `ct_dim, rt_dim ≤ 8` half-sync (`tt_metal/hw/inc/api/compute/matmul.h:189-192`) | silent DEST overrun, corrupted tiles |
| Two DEST accumulators live at once (`v_corr` preload + `nkcd@h`; `Γ·h` + `Pᵀ@v_new`; `Q@h` + `intra@v_new`) | the subblock of the three accumulate-onto-DEST block ops | subblock ≤ `dest_limit` tiles **including** the preloaded term (same slots — accumulate in place, no second slot) | wrong sums |
| Neumann doubling depth | number of squaring steps | `neumann_steps = ceil(log2(chunk_size))` (5 / 6), CT, derived from `chunk_size` | **wrong results**: truncates the series; passes C=32, fails C=64 |
| Uniform push quantum per CB (ring-wrap invariant, `blocking-model.md` §2) | pages per push of every CB reused across stages | each CB has ONE push quantum = max over its stage roles (`l1_ledger.md`); a stage needing fewer tiles leaves the tail unused; the host passes each quantum as a CT arg | FIFO pointer never hits `fifo_limit` exactly → hang or overlap |
| Semaphores per core: 16 (`tt_metal/impl/buffers/semaphore.hpp:16`) | `ready_segments` | `NS ≤ 8` (two semaphores per segment); Phase 0 `NS = min(NC, 4)` | program build failure |
| `H ≤ 32` | head-tile count | page index `((b·T + t)·Dt + dt)` assumes `ceil(H/32) == 1`; `validate()` raises for `H > 32` | silently wrong page indices |
| NoC read alignment: L1 destination ≡ DRAM source mod 64 (`gated_delta_net_backward_reader.cpp:19-24`) | face-row span read staging | each staging line starts at `stage64 + (src_off & 63)`; `row_span_stride = round_up(272·esz + 64, 64)` | misaligned reads, corrupted rows |
| NoC write alignment: L1 source ≡ destination mod 16 (`gated_delta_net_backward_writer.cpp:96-100`) | scalar scatter (`g_cumsum`) | stage each scalar at `wstage + r·16 + (dst_off & 15)` before the write | corrupted neighbouring heads |
| Face-row span | `gather_stage_tokens` | ≤ 32 (a destination tile holds 32 tokens) | staging over/under-fills a tile row |
| L1 residency (`l1_ledger.md` closed form) | `item_block_val_tiles` | largest divisor of `Vt` with `footprint ≤ ttnn.get_max_worker_l1_unreserved_size() − L1_KERNEL_CONFIG_RESERVE` (80 KB, the kernel-config ring buffer — measured on the backward) | L1 allocation failure at build (loud) |
| Scan-unit placement | `NV` | `BH·NV ≤ G` (one unit per core) when `BH ≤ G`; `NV = 1` and round-robin units otherwise | two units of one group on one core would share a readiness semaphore without a combined expected count (handled by the sum rule, Dataflow row 6) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 — phased P → S → E with segmented handoffs.** P: item-parallel prep `(bh, chunk)`; S: one scan unit per `(bh, v_block)` walking chunks; E: item-parallel output assembly `(bh, chunk)` on the P cores. | **built** | always (only Phase 0 path) | item: `(1 chunk) × Ct × Kt × Vi`; scan step: `Kt × Vs` state, `Ct × Kt` / `Kt × Ct` / `Ct × Vs` operands | **Minimum** (DRAM boundary): each of `q,k,v,g,beta,initial_state` read once, each output written once. R1: every input crosses **once** (face-row granularity for `q,k,v`); every output **once**. Above the minimum: the P→S and S→E scratch (`nkcd, Pᵀ, Γ` written once, read `NV×`; `v_corr, Q, intra, v_new` written once, read once) and one re-read of `h` by E. **Structurally unreachable minimum**: the three stages have different core assignments, so every value crossing a stage crosses DRAM. Full counts in `l1_ledger.md`. | Per item: ~15 phase inits + format reconfigs (intended **once per item**), ~20 CB handshakes, one reader pipeline fill. Per scan step: 2 block matmuls + 1 SFPU scale, inits **once per scan unit** (not per chunk). Rendezvous: `2·NS` semaphore waits per core per program — **not** per chunk |
| **R2 — fused per-(bh) single stage** (one core per `bh` does prep + scan + output for all chunks, no scratch, no handoffs) | **deferred** — R1 serves every shape R2 would, with ≥ R2's parallelism in P and E (`min(G, NI)` vs `BH` cores) and strictly more in S (`BH·NV` vs `BH`). R2 only saves the scratch round trip, which is full-page traffic — ≈ 90 page transactions per item at the largest shape against ≈ 3 300 face-row transactions R2 would then serialize onto `BH` cores. Reachable: R2 is R1's three per-item block sequences run back-to-back on one core with the handoffs removed. | `BH·NC ≤ small` (e.g. `NI == 1`) | same | saves scratch `2·(3·Ct·Kt + Ct² + 2·Ct·Vt + 1)` fp32 tiles per item and the `h` re-read | removes `2·NS` handoffs |
| **R3 — page-harvest / de-interleave** (a stage 0 reads each `[32 heads × 32]` source page **once**, harvests all `H` heads into head-major compact tiles; a mirrored stage writes `o`, `v_new`, `A` as full pages assembled from all heads) | **deferred** — layered **on top of** R1, not instead of it: R1's per-item gather is exactly what R3 replaces with a full-page compact read, and stages P/S/E survive unchanged. R1 is correct on every shape. Its payoff is `÷H` on gather/scatter transaction count (1× at `H=1`, 32× at `H=32`), but it adds a per-`(b, chunk)` fan-out rendezvous (one producer, `H` consumers) in front of P and a fan-in (`H` producers, one writer) behind E. | worth it at `H ≥ 4`, dominant at `H ∈ {16, 32}` (LOOSE Qwen3.5 geometry, `(1,256,32,128,128)`) | page window `32 tokens × Dt` for all `H` heads | gather transactions `B·Tpad·(2Kt+Vt)·H` → `B·Tpad·(2Kt+Vt)`; scatter writes `2·B·T·(2Vt+Ct)·H` → `B·T·(2Vt+Ct)` full pages; adds one compact write+read of `q,k,v` and of `o,v_new,A` | nothing new; moves the per-item fixed cost to `B·NC` page-window units |
| **R4 — multicast the scan's shared operands** (`nkcd_i, Pᵀ_i, Γ_i` do not vary with `v_block`; read once per `(bh, chunk)` by one scan unit and mcast to the other `NV−1`) | **deferred** — the operand-reuse check on the chosen S split: these operands are **reuse-shared by construction of the V split**. R1 re-reads them from DRAM `NV×` as full-page transfers that the depth-2 scan stream already hides behind compute. Reachable: the scan units of a group are placed on consecutive cores (`G−1−u`), so the receiver set is a contiguous run; the scan reader is the only thing that changes. `mcast_pipe.hpp` does not exist in this tree (Explore survey, `kernel_lib/` listing), so the handshake would be raw `noc_async_write_multicast` + semaphores. | `NV > 1` | same | DRAM reads of `(2·Ct·Kt + 1)` tiles per `(bh, chunk)` drop from `NV×` to `1×` — at `(1,4096,16,128,128)` fp32 that is **272 MB → 68 MB**, the largest scratch term of the op by bytes; adds one mcast (or `NV−1` unicasts) of the same payload per chunk over the NoC | fewer mcast rounds per chunk |
| **R5 — parallel prefix scan over chunks** (the recurrence is linear: `S_{i+1} = (Γ_i I + Pᵀ_i nkcd_i) S_i + Pᵀ_i v_corr_i`; a Blelloch scan over `[K,K]` transitions) | **deferred** — the chunk axis is the dependent one; R1 already spreads the scan over `BH·NV` cores. R5 composes `[K,K]@[K,K]` per merge (`K³`) against `K²V` per sequential step — no better at `V ≤ 2K`, and the Amdahl ceiling is the scan's share, which R1 has already shrunk by moving `o` to E. Reachable: S is a separate stage with its own units and scratch contract. | `NC ≫ BH·NV / G` | `[K,K]` transition block | adds `O(NC·log NC)` `[K,K]` blocks of cross-core traffic | fewer merge levels |
| **R6 — resident E** (when a core holds exactly one item, `Q` and `intra` stay in L1 from P to E instead of the scratch round trip) | **deferred** — a predicate-guarded fast path on R1 (`items_on_core == 1`, i.e. `NI ≤ G`); saves `Ct·Kt + Ct²` tiles written + read per item, which is < 3% of the item's transactions. Reachable: `cb_qs`/`cb_cc_a` already hold exactly those blocks at the end of P. | `NI ≤ G` | same | −`(Ct·Kt + Ct²)` write+read per item | — |
| **R7 — scan emits `o`** (compute `o_i` inside the scan step, no stage E) | **rejected** — superseded by R1's stage E. It puts two extra matmuls **and** the `o` face-row scatter (`2·C·Vs` writes per chunk per unit) on the only sequential path, and leaves every non-scan core idle for the whole scan. It is not a stepping stone: nothing of it survives in R1. | — | — | saves `Q, intra, v_new` scratch + `h` re-read | — |
| **R8 — per-head whole-page re-stream** (each item reads whole `[32×32]` pages and uses one row) | **rejected** — the dead end, superseded by R1's face-row span gather (and by R3). It moves `32·Dt·4 KB` per token per tensor to use `1/32` of it; at `(1,4096,16,128,128)` that is ≈ 1.6 GB of reads where R1's span gather moves ≈ 0.43 GB. | — | — | `32×` logical per item | — |

**Selection.** Exactly one regime is built, so the descriptor has no regime branch. Three **extents**
vary with shape and must be pinned by tests (the acceptance test does, see Work Distribution):
`NV` (1 vs > 1), `NS` (1 vs > 1), and scan units per core (1 vs > 1 when `BH > G`).

### Traffic ranking

`L = ` logical bytes of `q+k+v`. Tiers: DRAM (face-row transactions are the governing term — the
backward measured its reader RISC-issue bound at ~280 ns per gathered row, not bandwidth bound), then
cross-core (semaphores, no payload in R1), then core-local.

| # | Candidate split | DRAM crossings of `q,k,v` | Input / output transactions | Cross-core traffic | Occupancy (P / S / E) | Verdict |
|---|---|---|---|---|---|---|
| 1 | **(bh, chunk) for P and E; (bh, v_block) for S** — R1 | 1 (span reads, 272 elem per 32 used) | gather `C·(2Kt+Vt)` spans per item, scatter `2·C·(2Vt+Ct)` runs per item, both **once**, spread over `min(G,NI)` cores | `NV + 2·NS`-ish semaphore incs per item / unit, no payload; scratch is DRAM | `min(G,NI)` / `BH·NV` / `min(G,NI)` | **chosen** |
| 2 | (bh) for everything — R2 | 1 | same count, serialized on `BH` cores | none | `BH` / `BH` / `BH` | same bytes minus scratch, far worse critical path |
| 3 | (b, chunk) covering all heads — the R3 shape | 1 (**whole pages**, 100% useful) | `÷H` both ways | per-(b,chunk) fan-out/fan-in | needs its own stage to keep `(bh,chunk)` occupancy | cheapest in transactions → **R3 deferred** (layered on #1) |
| 4 | (bh, chunk) for S too, with a cross-core state combine — R5 | 1 | as #1 | `O(NC log NC)` `[K,K]` blocks | full grid in S | largest payload, smallest remaining Amdahl share |
| 5 | (bh, k_block) for S — split the contracted K axis | 1 | as #1 | a cross-core sum of `v_new` partials `[C,Vs]` **every chunk, on the critical path**, then a re-broadcast | `BH·NK` | dominated by #1: V gives the same parallelism with zero combine |
| 6 | (bh, v_block) for P and E too | `NV×` for `q,k` (V-independent prep replicated per V block) | `NV×` gather of `q,k` | none | `BH·NC·NV` | re-reads the dominant term — loses on bytes before occupancy |
| 7 | per-head whole-page reads — R8 | `32×/H`… per pass | `C·Dt` 4 KB pages per item | none | as #1 | dead end |

#1 is the unique candidate that pays the gather and the scatter **once**, keeps the scan free of any
combine, and hands the scan the maximum independent parallelism (V). #3 is cheaper in transactions and
is the recorded next step (R3); it is built on #1, not instead of it.

**Stall-shadow check.** Three stages wait:

| Stage that waits | On what | What runs in the window | Legal because |
|---|---|---|---|
| S unit, chunk `i` | P of `(bh, i)` on another core | P of later chunks (other cores) — the **segmented** handoff lets S start after segment 0, not after all of P; the strided item order makes P finish in chunk order grid-wide | `S_i` depends only on items `≤ i` |
| E item `(bh,i)` | S of `(bh, seg(i))` | nothing on that core is independent of it **except** E items of earlier segments, which the segmented handoff releases first; the scan itself runs on other cores | `o_i` depends only on `h_i, v_new_i` — the algebraic move that took `o` off the scan (associativity of `o = Q h + intra v_new`, exact) |
| S unit, next chunk's operands | the reader | the current chunk's matmuls (depth-2 prefetch) | the operands are state-independent |

No reorder changes floating-point order except "o off the scan", which evaluates the identical
expression `Q@h_i + intra@v_new_i` with `h_i` taken from the stored `h` (bit-identical for fp32; one
bf16 rounding of `h_i` for bf16, inside the bf16 band).

### Block schedule

Logical schedule; reader, compute and writer realize their parts asynchronously. Every core runs the
three stages in this order — **all of its P items, then its scan units, then its E items** — which is
what makes the handoffs deadlock-free.

```cpp
build_constant_tiles();                                    // reader, once per core
// ---- Stage P : items wi = core_idx + r·num_item_cores, r = 0.. (strided, chunk-major wi = i·BH + bh)
for (uint32_t r = 0; r < core_num_items; ++r) {
    gather_item_inputs(r);          // q,k [Ct,Kt]; g,beta full-width column tiles; tail zero-fill
    gate_columns_block(r);          // decay, gamma, w, Gamma_full; emit g_cumsum
    decay_mask_block(r);            // L = exp(LT@diag(g)@SL) ⊙ LT
    key_prep_block(r);              // q~ = scale·q ; k_beta = k ⊙ beta
    ut_matrix_block(r);             // N = (k_beta @ kᵀ) ⊙ L ⊙ SL
    ut_inverse_block(r);            // Tinv (Neumann doubling); emit A
    key_products_block(r);          // nkcd, Q, intra, Pᵀ, Gamma -> scratch
    for (uint32_t vb = 0; vb < num_item_v_blocks; ++vb) {
        gather_item_values(r, vb);  // v [Ct,Vi]
        value_products_block(r, vb);// v_corr = Tinv @ (v ⊙ beta) -> scratch
    }
    signal_scan_units(r);           // writer: after write barrier, +1 on sem_ready[seg(i)] of the NV units of bh
}
// ---- Stage S : scan units u owned by this core (unit u on core G-1-(u mod G))
for (uint32_t u = 0; u < core_num_scan_units; ++u) {
    load_initial_state(u);          // h0[:, vb] or zero-fill (no read)
    for (uint32_t j = 0; j < NS; ++j) {
        wait_segment_ready(j);      // reader: sem_ready[j] == core_expected_ready[j]
        for (uint32_t i = j*seg_chunks; i < min(NC,(j+1)*seg_chunks); ++i)
            state_step(u, i);       // emit h_i ; v_new = v_corr + nkcd@S -> scratch ; S = Γ·S + Pᵀ@v_new
        signal_segment_done(u, j);  // writer: after write barrier, +1 on sem_done[j] of every E core of (bh, seg j)
    }
    emit_final_state(u);
}
// ---- Stage E : the same items as P, same order (ascending segment)
for (uint32_t r = 0; r < core_num_items; ++r) {
    wait_segment_done(seg(r));      // reader, before the first item of each segment
    load_output_invariants(r);      // Q, intra
    for (uint32_t vb = 0; vb < num_item_v_blocks; ++vb) {
        load_output_vslice(r, vb);  // h_i[:, vb] (from the h output), v_new_i[:, vb] (scratch)
        output_block(r, vb);        // o = Q@h + intra@v_new ; v_new -> in-dtype
        scatter_outputs(r, vb);     // writer: o and v_new face-row scatter, rows t < T
    }
}
```

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `build_constant_tiles` | `EYE, LT, SL, SU` (`Ct²` each) + `ONES_ROW` (`Ct`) | all constants, whole program | once per core |
| `gather_item_inputs` / `gather_item_values` | `[Ct,Kt]×2`, `[Ct]×2` full-width columns / `[Ct,Vi]` | the gathered blocks, all of the item | one reader pipeline fill per item |
| `gate_columns_block` … `key_products_block` | see Block Operation Realization | `cb_vec` (decay, γ, w, Γ), `cb_L`, `Tinv`, `q̃`, `kβ` across all of P for the item | **one** init + one reconfig per operation per item, never per tile |
| `ut_inverse_block` | `Ct×Ct`, `2·neumann_steps − 1` block matmuls + `neumann_steps − 1` block adds | `cb_T` accumulator, `cb_pow` running power | one matmul init for the whole doubling loop |
| `value_products_block` | `[Ct,Ct]@[Ct,Vi]` | `Tinv` | once per V block |
| `state_step` | `[Ct,Kt]@[Kt,Vs]` + `[Kt,Ct]@[Ct,Vs]` + `Kt·Vs` SFPU scale | `cb_state` **for all `NC` steps** — never leaves L1 between steps | matmul init **once per scan unit**; 2 block matmuls per chunk |
| `output_block` | `[Ct,Kt]@[Kt,Vi]` + `[Ct,Ct]@[Ct,Vi]` accumulated in DEST | `Q`, `intra` across the V loop | once per V block |
| handoffs | — | — | `NS` waits per stage per core per program; one semaphore inc per (item, scan unit) and per (unit, segment, E core) |

### Provenance of the perf figures

`ttnn/ttnn/operations/examples/master.md` (the measured perf-pattern catalog) **does not exist on this
branch** (`ttnn/ttnn/operations/examples/` is absent). Every knob here is argued structurally —
transaction counts, bytes per tier, residency — plus the backward's on-device measurements recorded
in `gated_delta_net_backward/op_requirements.md` (reader issue-bound at ~280 ns per gathered row;
`gather_depth` 2→4 worth only 1.01–1.02×; the 80 KB kernel-config reserve; ACCUM_DEPTH = 2 required).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap (items)** | Block CBs are depth 1; at `NI > G` (LOOSE: ~9–10 items/core) the reader cannot run ahead into item `r+1` of P, and the gather is the dominant term | depth 2 on `cb_q_in`, `cb_k_in`, `cb_vblock_in`, `cb_gate_in` only (+48–96 KB at the largest shapes; re-run the `Vi` solve) |
| **Overlap (block_chunks)** | `block_chunks = 1`; at `K=V≤64` the item working set is ≪ L1 and ~15 phase inits are paid per chunk | `BLOCK_CHUNKS = 2` at small `K·V` — and note it also halves the scan's handoff granularity |
| **Grid synchronization** | At `NI ≤ 2` (`(1,32,1,32,32)`, `(1,64,1,64,64)`) the handoffs and scratch are pure overhead between two cores | measure the R2 predicate (`NI == 1`: one core, no handoffs) |
| **Scan placement** | Scan units sit on the last cores, which also hold P items when `NI ≥ G`; the scan then starts only after that core's own P items (~`NI/G` item-times) | exclude the scan cores from P (`num_item_cores = G − NU`) when `NC·s > (NI/G)·p` — measured, not argued |
| **Scan V-split bandwidth** | Occupancy-first `NV` (max split) multiplies the reuse-shared `nkcd/Pᵀ/Γ` DRAM reads by `NV`. At `K=128, Vs=1` a scan step reads 68 KB for ~16 tile-matmuls; with 64 scan units that is an aggregate demand that can exceed DRAM bandwidth, so the step becomes bandwidth-bound rather than compute-bound | `NV/2` (coarser `Vs`) against max `NV`; the structural fix is R4 |
| **Segments** | `NS = 4` trades 8 semaphores and `NS` waits for overlap; at `NC ≤ 4` segments are single chunks | `NS ∈ {1, 2, 8}` |
| **Gather granularity** | 272-element span read per `(token, d_tile)` (~12% useful bytes fp32) vs two 64 B reads vs whole page | sweep the three (the backward's lamp, same analysis); R3 is the structural fix |
| **fp32 DEST** | `fp32_dest_acc_en=True` halves DEST to 4 tiles, capping every subblock | `False` at bfloat16 only, against the `A`/`o` PCC gate; never override a caller config |
| **Reconfig elision** | ~15 phase boundaries per item, most fp32→fp32 | elide `reconfig_data_format` only on boundaries that cannot touch an in-dtype CB (`cb_q_in`, `cb_k_in`, `cb_vblock_in`, `cb_gate_in`, `cb_out_egress`) |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `q,k,v` DRAM → L1 (P) | TILE, in dtype | **face-row span gather** (the backward's, reused): for head `h`, token `t`, d-tile `dt`, one `noc_async_read` of 272 elements starting at `(h/16)·512 + (h%16)·16` of page `(b·T + t)·Dt + dt` into `cb_gather_stage`, then two 16-element RISC-V re-packs into the destination tile's face rows. `gather_stage_tokens` tokens per window, `GATHER_DEPTH` windows in flight. | the dominant cost of the op. `TensorAccessor::get_noc_addr(page, offset)` supplies addresses |
| `g,beta` DRAM → L1 (P) | TILE, in dtype | **whole-page read + local column extract**: `Ct` page reads per tensor per item (page `b·Tt + t/32`), then the reader copies column `h` into **every column** of the destination tile (full-width gate tiles). | `C/32` transactions instead of `C` scalar reads. Full-width tiles make every derived gate tile (`decay, γ, w, Γ`) valid in all columns, so a plain elementwise multiply and a COL broadcast are both correct, and `Γ_full` needs no broadcast in the scan |
| padded tail | — | reader zero-fills rows `t ≥ T` of every gathered block, including `g`, `beta` | makes the padded-tail table hold by construction |
| `initial_state` DRAM → L1 (S) | TILE, in dtype | full-page reads of `[Kt, Vs]` at page `(b·H + h)·Kt·Vt + kt·Vt + vt`, typecast into `cb_state` (fp32). **Not bound, not read** when `has_h0 = 0`: compute zero-fills `cb_state` | `h[:,0]` is then an exact zero |
| scratch (P → S, P → E, S → E) | TILE, **Float32 always** | one flat DRAM tensor `sc` (`[tiles·32, 32]`, fp32, `ttnn.allocate_tensor_on_device`), per-item strides: `nkcd Ct·Kt`, `pt Kt·Ct`, `gam 1`, `vcorr Ct·Vt`, `qd Ct·Kt`, `intra Ct²`, `vnew Ct·Vt`; bases on host → CT args. Full-page reads/writes. | scratch is always DRAM regardless of `memory_config` |
| **P → S handoff** (segmented) | — | Writer of the P core, after `noc_async_write_barrier()` for item `(bh, i)`, sends `noc_semaphore_inc(+1)` to `sem_ready[seg(i)]` on each of the `NV` scan-unit cores of `bh`. A scan core's reader waits `sem_ready[j] == core_expected_ready[j]`, where `core_expected_ready[j] = Σ_{units on this core} (#chunks in segment j)` (host RT arg), before reading any segment-`j` scratch. | deadlock-free: every core issues **all** P items before **any** wait, and P never waits |
| **S → E handoff** (segmented) | — | Scan-unit writer, after the write barrier of the last chunk of segment `j`, sends `+1` to `sem_done[j]` on every core holding an E item of `(bh, segment j)` (host-computed unicast list). An E core's reader waits `sem_done[j] == core_expected_done[j] = NV · #{distinct bh with an item in segment j on this core}` before its first segment-`j` item. | E items are processed in ascending `wi = i·BH + bh`, hence ascending segment — waits are monotone. S waits only on P, E only on S: no cycle |
| `h_i`, `final_state` L1 → DRAM (S) | TILE, in dtype | full-page writes from `cb_out_egress` | `h` is also E's source for `h_i` — read back as full pages |
| `A`, `g_cumsum` L1 → out (P) | TILE, in dtype | **face-row scatter** of `Tinv` rows (two 16-element writes per `(token, c_tile)`), **scalar scatter** of `decay` column into column `h` (staged at matching mod-16 offset) | rows `t < T` only |
| `o`, `v_new` L1 → out (E) | TILE, in dtype | face-row scatter, two 16-element writes per `(token, v_tile)` | rows `t < T` only; this is the op's largest write term and it runs on the E cores, off the scan |
| Sharded placement | — | Not in TARGET (no `memory_layout` axis). If added: a `(bh)` cut of `q,k,v` is an **independent** cut → knob-turn (bind the input CBs to the shard via `ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, no NoC re-read); a `V` cut of `v`/`initial_state` matches the scan's V split → knob-turn for S, and P/E keep reading; a `T` cut crosses the **dependent** chunk axis → scheme-change (the S handoff becomes the only cross-shard traffic, R1 already expresses it) | stated so it need not be rediscovered |

---

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | P/E: one item `(bh, chunk)`; S: one scan unit `(bh, v_block)` walking all chunks |
| Grid | `grid = device.compute_with_storage_grid_size()`, `G = grid.x·grid.y`, row-wise logical core order `c = 0..G−1` |
| Item order | chunk-major `wi = i·BH + bh` (so P completes in chunk order across the grid and the segmented handoff can release early segments) |
| Per-core items | `num_item_cores = min(G, NI)`; core `c < num_item_cores` owns `wi ∈ {c, c + num_item_cores, …}` (strided) — `core_num_items = ceil((NI − c)/num_item_cores)`: the first `NI mod num_item_cores` cores get one extra, the same balance `split_work_to_cores` gives. E items = P items, same core, same order |
| Scan split | `NV` = largest divisor of `Vt` with `BH·NV ≤ G` (else 1); `Vs = Vt/NV`; `NU = BH·NV`; unit `u = bh·NV + vb` on core `G − 1 − (u mod G)` (reverse order: lands on the cores with the fewest items, or none when `NI + NU ≤ G`) |
| Active cores | union of item cores and scan cores; every active core gets all three kernels and all CBs; inactive roles have zero counts in their RT args |
| Tile geometry | `Ct = C/32` (exact, validated), `Kt = K/32`, `Vt = V/32` (validated), `NC = ceil(T/C)`, `Tt = ceil(T/32)`. `T` is not chunk- or tile-aligned on 7 of 18 INPUTS; every token-row count uses `ceil` and a `t < T` guard |
| RT args (per core) | addresses; `core_item_start`, `core_num_items`, `num_item_cores`; `core_num_scan_units` + unit list; `core_expected_ready[NS]`, `core_expected_done[NS]`; per-item scan-unit NoC coords (P writer); per-(unit, segment) E-core NoC coord lists (S writer); `scale` bits |

Only one regime is built. The **extent-pinned tests** the acceptance test carries:

| Extent case | Shape that pins it (any grid ≥ 8×8) |
|---|---|
| `NV > 1` (V split across scan cores), `Vs = 1` | `(1,256,4,128,256)` c64 (`BH·Vt = 32`) |
| `NV = 1`, `Vs = Vt > 1` | `(4,128,16,64,64)` c32 (`BH = 64`, `NV = 2` only on grids ≥ 128 cores) |
| `NV > 1` **and** `Vs > 1` | `(1,256,32,128,128)` c64 (`BH = 32`, `NV = 2`, `Vs = 2`) |
| scan units per core > 1 (`BH > G`) | `(4,64,32,32,32)` c32 (`BH = 128`) |
| `NS = 1` / `NS = 4` | `NC = 1` shapes / `NC ≥ 4` shapes |
| items per core > 1 | `(4,128,16,64,64)` c32 (`NI = 256`) |
| `NC = 1`, `T < C` | `(1,48,3,64,64)` c64 |

---

## Circular Buffers

`in` = input dtype; `f32` = Float32. `Da = ACCUM_DEPTH = 2`, `Ds = SCAN_STREAM_DEPTH = 2`,
`De = EGRESS_DEPTH = 2`, `Dg = GATHER_DEPTH = 2`. Push quanta (one per CB, uniform across stages):
`Qf = Ct·max(Kt, Ct, Vi)` for `cb_scratch_egress`, `Qo = max(Ct², Ct, Kt·Vs, Ct·Vi)` for `cb_out_egress`.
Full sizing and sharing justification: `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_const` | 0 | f32 tile | `4·Ct² + Ct` | `EYE, LT, SL, SU` `[C,C]` + `ONES_ROW` `[32,C]`; spans `C`, constant in every other axis | f32 | reader | compute | whole program (pushed once, never popped) |
| `cb_gather_stage` | 1 | `gather_stage_tokens·row_span_stride + 64` | `Dg` | one token window of span reads; reader-local | raw | reader | reader | P (reader-local; the writer's scalar staging is the separate `cb_scalar_stage`) |
| `cb_scalar_stage` | 2 | 16 B·32 | 1 | per-token mod-16 staging for the `g_cumsum` scalar scatter | raw | writer | writer | P |
| `cb_q_in` | 3 | in tile | `Ct·Kt` | spans `(C,K)` | in | reader | compute | P |
| `cb_k_in` | 4 | in tile | `Ct·Kt` | spans `(C,K)` | in | reader | compute | P |
| `cb_vblock_in` | 5 | in tile | `max(Ct,Kt)·Vi` | one uniform quantum `max(Ct,Kt)·Vi` serving three disjoint reader→compute roles: P `v[:, vb]` (`Ct·Vi` valid), S `initial_state[:, vb]` (`Kt·Vs` valid, once per unit, only when `has_h0`), E `h_i[:, vb]` from the `h` output (`Kt·Vi` valid) | in | reader | compute | P, S, E |
| `cb_gate_in` | 6 | in tile | `2·Ct` | full-width `g` and `beta` column tiles | in | reader | compute | P |
| `cb_vec` | 7 | f32 tile | `4·Ct` | `decay, γ, w, Γ_full` column tiles | f32 | compute | compute | P |
| `cb_qs` | 8 | f32 tile | `Ct·Kt` | P: `q̃`, resident for the item; spans `(C,K)` | f32 | compute | compute | P |
| `cb_kb` | 9 | f32 tile | `Da·Ct·Kt` | P: `kβ → U` in place | f32 | compute | compute | P |
| `cb_kw` | 10 | f32 tile | `Ct·Kt` | P: `k ⊙ w` (transposed out to egress) | f32 | compute | compute | P |
| `cb_L` | 11 | f32 tile | `Ct²` | `L`, resident for the item | f32 | compute | compute | P |
| `cb_cc_a` | 12 | f32 tile | `Ct²` | P: `diag(g)` → `N` | f32 | compute | compute | P |
| `cb_cc_b` | 13 | f32 tile | `Ct²` | P: `X = LT@diag(g)` | f32 | compute | compute | P |
| `cb_T` | 14 | f32 tile | `Da·Ct²` | `Tinv` accumulator, `T ← T + T@Pw` in place | f32 | compute | compute | P |
| `cb_pow` | 15 | f32 tile | `Da·Ct²` | Neumann power `Pw ← Pw@Pw` in place | f32 | compute | compute | P |
| `cb_vmat` | 16 | f32 tile | `Ct·Vi` | P: `vβ` | f32 | compute | compute | P |
| `cb_intra_in` | 17 | f32 tile | `Ct²` | E: `intra` from scratch | f32 | reader | compute | E |
| `cb_vnew_in` | 18 | f32 tile | `Ct·Vi` | E: `v_new_i[:, vb]` from scratch | f32 | reader | compute | E |
| `cb_kmat_in` | 20 | f32 tile | `Ds·Ct·Kt` | S: `nkcd_i`, prefetched one chunk ahead; E: `Q` from scratch (same quantum `Ct·Kt`, disjoint lifetime) | f32 | reader | compute | S, E |
| `cb_scan_pt` | 21 | f32 tile | `Ds·Kt·Ct` | S: `Pᵀ_i` | f32 | reader | compute | S |
| `cb_scan_vcorr` | 22 | f32 tile | `Ds·Ct·Vs` | S: `v_corr_i[:, vb]` | f32 | reader | compute | S |
| `cb_scan_gamma` | 23 | f32 tile | `Ds` | S: `Γ_full_i` | f32 | reader | compute | S |
| `cb_state` | 25 | f32 tile | `Da·Kt·Vs` | the running state, in place, resident across all `NC` steps | f32 | compute | compute | S |
| `cb_scan_vnew` | 26 | f32 tile | `Ct·Vs` | S: `v_new_i` (feeds `Pᵀ@v_new`) | f32 | compute | compute | S |
| `cb_scratch_egress` | 27 | f32 tile | `De·Qf` | P: `nkcd, Pᵀ, Γ_full, Q, intra, v_corr`; S: `v_new_i` | f32 | compute | writer | P, S |
| `cb_out_egress` | 28 | in tile | `De·Qo` | P: `Tinv` (→ `A`), `decay` (→ `g_cumsum`); S: `h_i`, `final_state`; E: `o`, `v_new` | in | compute | writer | P, S, E |

`cb_qs` (compute → compute, P) and `cb_kmat_in` (reader → compute, S/E) hold the same-shaped block but
are kept separate because one CB may not have two producers; keeping `Q` resident in `cb_qs` across the
handoff instead is regime R6 (deferred). Index slots 19, 24, 29 are unused (merged into
`cb_vblock_in` / `cb_kmat_in`, see `l1_ledger.md`).

Every CB has exactly one producer kernel and one consumer kernel over the program. Compute→compute
CBs (`cb_vec`, `cb_qs`, `cb_kb`, `cb_kw`, `cb_L`, `cb_cc_*`, `cb_T`, `cb_pow`, `cb_vmat`, `cb_state`,
`cb_scan_vnew`) are never touched by a dataflow kernel — which is exactly why the two egress CBs exist.
Reader-local (`cb_gather_stage`) and writer-local (`cb_scalar_stage`) scratch are single-thread L1
allocations (the generic-op model has no other allocator).

`cb_state` and the scan-stream CBs need `UnpackToDestFp32` on `cb_state` and `cb_scan_gamma` in the
compute config's `unpack_to_dest_mode` for the exact carry path (Precision contract); every other CB
keeps the default.

---

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|---|---|---|---|---|---|
| 1 | `build_constant_tiles` | `4×[Ct,Ct]` + `[1,Ct]` | no (raw L1 stores, reader) | — | `cb_const` | pushed once, read by index all program |
| 2 | `gather_item_inputs` | `[Ct,Kt]×2`, `[Ct]×2` | no (dataflow) | DRAM `q,k,g,beta` → `cb_gather_stage` | `cb_q_in`, `cb_k_in`, `cb_gate_in` | tail rows zero |
| 3 | `gate_columns_block` | `[Ct,Ct]@[Ct,1]` ×3 (+ `[1,Ct]@[Ct,1]`), `exp` fused in DEST | matmul raw; `Exp` in DEST | `cb_const` (`LT`, `SU`, `ONES_ROW`), `cb_gate_in` (g) | `cb_vec` (`decay, γ, w, Γ_full`), `cb_out_egress` (← `decay`, for `g_cumsum`) | `decay` packed before `exp`; `γ = exp(LT@g)`, `w = exp(SU@g)`, `Γ_full = exp(ONES@g)` all from the **same** DEST accumulations |
| 4 | `decay_mask_block` | `[Ct,Ct]` ×3 | eltwise helper for `diag(g) = EYE ⊙ g`; matmul raw | `cb_const` (`EYE, LT, SL`), `cb_gate_in` | `cb_cc_a` (diag) → `cb_cc_b` (`X`) → `cb_L` | `L = exp(X@SL)` in DEST, then `⊙ LT` mask, packed once |
| 5 | `key_prep_block` | `[Ct,Kt]` ×2 | **helper** (`unary<MulUnary>` for scale, `mul` COL-bcast for `⊙β`) | `cb_q_in`, `cb_k_in`, `cb_gate_in` (β) | `cb_qs` (`q̃`), `cb_kb` (`kβ`) | |
| 6 | `ut_matrix_block` | `[Ct,Kt]@[Kt,Ct]` (`in1` transposed) then `⊙L⊙SL` | matmul raw; masks via eltwise helper | `cb_kb`, `cb_k_in`, `cb_L`, `cb_const` (`SL`) | `cb_cc_a` (`N`) | |
| 7 | `ut_inverse_block` | `[Ct,Ct]`: `T = EYE − N`; `Pw = N@N`; `j=1..m−1: T ← T + T@Pw; Pw ← Pw@Pw (j<m−1)` | matmul raw; `sub` helper for `EYE − N` | `cb_cc_a`, `cb_const` (`EYE`) | `cb_T` (`Tinv`), `cb_pow` scratch; `cb_out_egress` ← `Tinv` (→ `A`) | `cb_T` resident to the end of the item |
| 8 | `key_products_block` | `U = kβ⊙γ` `[Ct,Kt]` in place; `nkcd = −Tinv@U`; `Q = q̃⊙γ`; `intra = (q̃@kᵀ)⊙L`; `Pᵀ = (k⊙w)ᵀ`; `Γ_full` | eltwise helper (`mul` COL, `Negative`), matmul raw, `transpose_block` raw | `cb_kb`, `cb_T`, `cb_qs`, `cb_k_in`, `cb_L`, `cb_vec` | `cb_scratch_egress` ← `nkcd`, `Q`, `intra`, `Pᵀ`, `Γ_full` (in that fixed order); `cb_kw` scratch | `Q` and `intra` are packed **straight into egress** (no CB); `nkcd` negated in DEST before pack |
| 9 | `gather_item_values` | `[Ct,Vi]` | no (dataflow) | DRAM `v` | `cb_vblock_in` | |
| 10 | `value_products_block` | `vβ = v⊙β` `[Ct,Vi]`; `v_corr = Tinv@vβ` | `mul` helper COL-bcast; matmul raw | `cb_vblock_in`, `cb_gate_in`, `cb_T` | `cb_vmat` (`vβ`) → `cb_scratch_egress` ← `v_corr` | |
| 11 | `signal_scan_units` | — | no | — | `sem_ready[seg(i)]` on `NV` cores | after write barrier |
| 12 | `load_initial_state` | `[Kt,Vs]` | copy helper (typecast in → f32) when `has_h0`; zero-fill otherwise | `cb_vblock_in` | `cb_state` | |
| 13 | `state_step` | emit `h_i` (`[Kt,Vs]` → in dtype); `v_new = v_corr + nkcd@S` (preload `v_corr` into DEST, matmul accumulates); `S ← Γ_full·S + Pᵀ@v_new` (`Γ·S` on the SFPU in fp32 DEST, matmul accumulates on top) | matmul raw; `mul_binary_tile` SFPU; `copy` helper for the `h_i` typecast pack | `cb_state`, `cb_kmat_in`, `cb_scan_vcorr`, `cb_scan_pt`, `cb_scan_gamma` | `cb_out_egress` ← `h_i`; `cb_scan_vnew` + `cb_scratch_egress` ← `v_new` (two packs of the same DEST tile); `cb_state` ← `S_{i+1}` (in place) | `cb_state` never leaves L1 between steps |
| 14 | `emit_final_state` | `[Kt,Vs]` | `copy` helper | `cb_state` | `cb_out_egress` | |
| 15 | `load_output_invariants` / `load_output_vslice` | `[Ct,Kt]`, `[Ct,Ct]` / `[Kt,Vi]`, `[Ct,Vi]` | no (dataflow) | DRAM scratch, `h` output | `cb_kmat_in`, `cb_intra_in` / `cb_vblock_in`, `cb_vnew_in` | |
| 16 | `output_block` | `o = Q@h + intra@v_new` `[Ct,Vi]` (both products accumulate in the same DEST subblock); `v_new` typecast | matmul raw; `copy` helper for `v_new` | `cb_kmat_in`, `cb_vblock_in`, `cb_intra_in`, `cb_vnew_in` | `cb_out_egress` ← `o`, then ← `v_new` | |
| 17 | `scatter_outputs` | `[Ct,Vi]` ×2 | no (dataflow) | `cb_out_egress` | DRAM `o`, `v_new` | rows `t < T` |

Writer routing is a compile-time-fixed order per stage (P: `decay→g_cumsum`, `Tinv→A`, `nkcd, Q, intra,
Pᵀ, Γ_full → sc`, then `v_corr` per V block; S: per chunk `h_i` then `v_new`; E: per V block `o` then
`v_new`), so the writer never inspects data to route it.

### Precision contract

| Quantity | Rule | Why |
|---|---|---|
| `D`, hence `L` | built as `(LT@diag(g))@SL`, `exp` applied **in DEST** on the fp32 accumulation, masked `⊙LT` after — `decay` never enters an FPU source register on its way to `L` | measured: 5.8× better `L` rel-RMS at `g_scale = 8` (backward `op_requirements.md` Refinement 1). The golden suite's `g_scale = 8` LOOSE case and `test_gate_strength[saturated_decay]` are the instrument |
| `w` | `exp(SU@g)` — a direct sum of `g`, not `Γ/γ` or `decay[C−1] − decay` | same cancellation hazard as `L` |
| carried state | `Γ·S` on the SFPU (`mul_binary_tile`, fp32 DEST, `S` and `Γ_full` unpacked to DEST as fp32); `Pᵀ@v_new` matmul-accumulated onto it | the carry path never passes through a tf32 source register; only the added term does |
| internal CBs | `Float32` always (every accumulator, state, decay, `Tinv`, scratch) | prompt mandate + `A`/`h` are backward inputs |
| boundary | in-dtype only at `cb_q_in, cb_k_in, cb_vblock_in, cb_gate_in, cb_out_egress` | bf16 is a boundary-format difference |

---

## API Mapping

Paths: `KL = ttnn/cpp/ttnn/kernel_lib`, `CAPI = tt_metal/hw/inc/api/compute`.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup(in0, in1, out)` | `CAPI/compute_kernel_hw_startup.h` | called once before any helper | — | — | — |
| `key_prep_block` (`q̃ = scale·q`) | **helper** | `compute_kernel_lib::unary<MulUnary<>, In, Out>(IterationShape::tiles(Ct·Kt))` | `KL/eltwise/api/convenience.hpp:68`; `MulUnary` `KL/eltwise/unary/scalar.hpp:32` | scale bits RT arg | `cb_q_in` | `cb_qs` | `IterationShape::tiles(Ct·Kt)` |
| `⊙β`, `⊙γ`, `EYE⊙g` (5, 8, 10, 4) | **helper** | `compute_kernel_lib::mul<AInput, BroadcastInputSpec, Output>(IterationShape::grid(Ct, Dt))` with `BroadcastDim::Col` on the gate tile | `KL/eltwise/api/convenience.hpp:51`; `BroadcastDim` `KL/eltwise/api/chain.hpp:305`; `IterationShape::grid` `chain.hpp:135` | B = gate column tile (valid in every column — full-width) | `cb_k_in`/`cb_q_in`/`cb_vblock_in`/`cb_kb`, `cb_gate_in`/`cb_vec` | `cb_kb`/`cb_vmat`/`cb_cc_a` | `grid(Ct, Kt)` / `grid(Ct, Vi)` / `grid(Ct, Ct)` |
| mask multiplies (`⊙L`, `⊙SL`, `⊙LT`) not fused into a matmul | **helper** | `compute_kernel_lib::mul<…>(IterationShape::tiles(Ct²))` | `convenience.hpp:51` | 0/1 masks from `cb_const` | `cb_cc_*`, `cb_L`, `cb_const` | `cb_cc_*`, `cb_L` | `tiles(Ct²)` |
| `T = EYE − N` | **helper** | `compute_kernel_lib::sub<…>(IterationShape::tiles(Ct²))` | `convenience.hpp:48` | | `cb_const`, `cb_cc_a` | `cb_T` | `tiles(Ct²)` |
| `nkcd` negate, `exp` on DEST results | **helper** chain element | `Negative<>`, `Exp<>` inside `eltwise_chain` / applied on the matmul's DEST window | `KL/eltwise/unary/misc.hpp:19`, `KL/eltwise/unary/math.hpp:22`; `eltwise_chain` `chain.hpp:532` | `Exp<Approx::Exact>` (math_approx_mode=False) | DEST | DEST | — |
| typecast packs (`h_i`, `final_state`, `v_new`, `decay`, `Tinv` → in dtype) | **helper** | `compute_kernel_lib::copy<In, Out>(IterationShape::tiles(n))` | `convenience.hpp:87` | pack format = `cb_out_egress` format | f32 CBs | `cb_out_egress` | `n` = block tile count |
| every block matmul (3, 4, 6, 7, 8, 10, 13, 16) | raw_api | `matmul_block_init(in0, in1, transpose, ct_dim, rt_dim, kt_dim)` / `matmul_block(in0, in1, i0, i1, idst, transpose, ct_dim, rt_dim, kt_dim)` | `CAPI/matmul.h:195`, `:246`; constraints `:189-192` | `transpose` transposes **in1 tiles only**; the caller swaps the tile-grid walk for `kᵀ` | per row | per row | block extents `Ct, Kt, Vi, Vs`; `ct_dim/rt_dim` are the subblock walk ≤ `dest_limit` |
| `Pᵀ` materialization | raw_api | `transpose_init(icb)` / `transpose_block(icb, start_itile, start_idst, ntiles)` | `CAPI/transpose.h:39`, `:164` | `ntiles ≤ dest_limit`; caller writes tiles in `[Kt, Ct]` order | `cb_kw` | `cb_scratch_egress` | — |
| `Γ·S` carry | raw_api | `copy_tile(cb, i, dst)` (UnpackToDestFp32) + `mul_binary_tile(d0, d1, odst)` | `CAPI/tile_move_copy.h:126`; `CAPI/eltwise_binary_sfpu.h:67` | `Γ_full` in the second DEST slot | `cb_state`, `cb_scan_gamma` | DEST (matmul accumulates on top) | subblock ≤ `dest_limit` |
| reader / writer addressing | raw_api | `TensorAccessor`, `get_noc_addr(page, offset)` | `tech_reports/tensor_accessor/tensor_accessor.md` | `TensorAccessorArgs` CT; absent `initial_state` borrows `q`'s args, guarded by `has_h0` | — | — | — |
| handoffs | raw_api | `noc_semaphore_inc`, `noc_semaphore_wait` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2261`, `:1940` | `2·NS` semaphores (`SemaphoreDescriptor`, initial 0) | — | — | `NS` |

### Helpers considered and rejected (one per `raw_api` block operation)

| Block operation | Helper considered | Concrete reason, with citation |
|---|---|---|
| every block matmul | `matmul_block_helpers.hpp::matmul_block()` | **Does not exist in this tree**: `ttnn/cpp/ttnn/kernel_lib/` holds `dest_helpers.hpp`, `dfb_helpers_*`, `l1_helpers.hpp`, `reduce_helpers_*`, `tilize_helpers.*`, `untilize_helpers.*`, `eltwise/` and nothing else (Explore survey; the only "matmul" strings are comments, e.g. `reduce_helpers_common.hpp:23`). The block matmuls are built on `CAPI/matmul.h:246`, the layer the helper would wrap — building the missing block op, not a tile loop |
| `Pᵀ` transpose | `eltwise_chain` / `unary_bcast` | the chain's elements (`chain.hpp:482-504`: `CopyTile`, `BinaryFpu`, `DestReuseBinary`, `PackTile`, SFPU unaries) have no transpose element; `transpose_block` (`transpose.h:164`) is the block op |
| `Γ·S` carry | `mul<…, BroadcastDim::Scalar>` (`convenience.hpp:51`) | an FPU binary unpacks `S` into a source register (~tf32) on every chunk — exactly the carry-path rounding the Precision contract forbids; the SFPU binary (`eltwise_binary_sfpu.h:67`) operates on fp32 DEST. The chain's SFPU binary (`binary_sfpu`, `convenience.hpp:80`) is the helper form and **should** be used if its input lifecycle can express "S from `cb_state` in place + Γ from `cb_scan_gamma`, no pop" — the implementer chooses |
| cumsum (`gate_columns_block`) | `CAPI/cumsum.h:33 cumsum_tile` | columnwise within a tile, needs NWH ordering with `first=false` across tile rows (`cumsum.h:18-21`); the triangular-ones matmul produces `decay`, `w` (reverse-exclusive), `Γ` and `D`'s left factor with one primitive and one block shape |
| reductions | `reduce_helpers_compute.hpp::reduce` | not needed: the forward has no reduction — `Σg` is `ONES_ROW @ g` inside `gate_columns_block` |
| handoffs | `mcast_pipe.hpp` | does not exist in this tree; and the handshake is a fan-in counter, not a payload multicast |
| tilize / untilize | `tilize_helpers.hpp`, `untilize_helpers.hpp` | TILE in, TILE out everywhere |

---

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `key_prep_block` `kβ = k⊙β` | `mul` | `cb_k_in`: All `[C,K]` | `cb_gate_in[β]`: All (full-width column tile; Col0 suffices) | Col |
| `decay_mask_block` `diag(g) = EYE ⊙ g` | `mul` | `cb_const[EYE]`: All | `cb_gate_in[g]`: All (full-width) | Col |
| `key_products_block` `U = kβ⊙γ`, `Q = q̃⊙γ` | `mul` | `cb_kb` / `cb_qs`: All | `cb_vec[γ]`: All | Col |
| `key_products_block` `k⊙w` | `mul` | `cb_k_in`: All | `cb_vec[w]`: All | Col |
| `value_products_block` `vβ = v⊙β` | `mul` | `cb_vblock_in`: All `[C,Vi]` | `cb_gate_in[β]`: All | Col |
| masks `⊙L`, `⊙SL`, `⊙LT` | `mul` | `[C,C]`: All | `[C,C]`: All | None |
| `state_step` `Γ·S` | SFPU `mul_binary_tile` | `cb_state`: All | `cb_scan_gamma[Γ_full]`: All | None (full tile) |

---

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| **`decay` through an FPU source register** | at `g_scale = 8` `|decay| ≈ 250`; a tf32 operand keeps ~0.2 absolute — every `L[t,s]` near the diagonal is wrong by up to ~20% | `D = LT@diag(g)@SL` with `exp` in DEST; `w = exp(SU@g)`; no `decay ⊗ 1` outer product anywhere. The saturated-gate golden cells are the check |
| **`h` is the ENTERING state** | the natural loop writes `S` after the update | `state_step` emits `h_i` **before** the update; `h[:,0]` is `initial_state` (or packed zeros); the state after the last chunk goes only to `final_state` |
| **`no_h0` must read nothing and give exact zeros** | a placeholder accessor is bound for CT-arg layout stability | compute zero-fills `cb_state` (no reader involvement) under `has_h0 = 0`; `test_no_h0_first_state_is_zero` checks `== 0.0` exactly |
| **Handoff deadlock** | three stages, `2·NS` semaphores, cores with every role | per-core order is P → S → E in all three kernels; P never waits; S waits only on P; E only on S; expected counts are host-computed sums (Dataflow rows 6–7). A wait placed before the core's own P signalling deadlocks — do not reorder |
| **Signal before data lands** | semaphore incs are not ordered after posted writes | `noc_async_write_barrier()` before every handoff `noc_semaphore_inc` |
| **Uniform push quantum** | `cb_out_egress` carries `Tinv`, `decay`, `h_i`, `final_state`, `o`, `v_new` of different sizes; `cb_scratch_egress` likewise | every push/pop is the CB's single quantum (`Qo`, `Qf`); the writer writes only the valid tiles. A variable push is a hang (backward descriptor docstring on `_cb_blocks`) |
| **Scan-unit semaphore sharing when `BH > G`** | two units on one core share `sem_ready[j]` | expected count is the **sum** over the core's units; pinned by `(4,64,32,32,32)` |
| **Neumann depth** | too few squarings silently truncate | `neumann_steps = ceil(log2 C)`, CT, from `chunk_size` |
| **matmul transposes in1 only** | `q̃@kᵀ`, `kβ@kᵀ` need `kᵀ` as in1 (fine, flag + grid swap); `Pᵀ@v_new` needs a transposed **in0** | `Pᵀ` materialized in stage P (`transpose_block`), so the scan has no transpose |
| **Mixed-format matmul in E** | `Q` (f32) @ `h_i` (bf16 when dtype is bf16) | reconfig both unpackers per matmul pair; both formats have 8-bit exponents. bf16 `h_i` costs one rounding of the state in `o` — inside the bf16 band |
| **`UnpackToDestFp32` on `cb_state`** | the unpack-to-dest mode may also route `cb_state`'s matmul unpack | verify on bring-up; if it conflicts, fall back to the FPU `mul_tiles_bcast_scalar` carry (the backward's measured-acceptable tf32 carry) and file the precision delta |
| **L1 outputs shrink the CB budget** | `memory_config=L1` allocates outputs in the same L1 the CBs live in | the solve budgets against `get_max_worker_l1_unreserved_size() − 80 KB`; with L1 outputs the build fails loudly rather than corrupting. `lowest_occupied_compute_l1_address()` (`tt_metal/api/tt-metalium/device.hpp:130`) is the exact bound if it is ever bound in Python |
| **Kernel binary size** | the backward's three binaries hit the kernel-config ring buffer and needed `#pragma GCC optimize("Os")` | same pragma; `L1_KERNEL_CONFIG_RESERVE = 80 KB` stays in the solve |
| **q/k must be L2-normalized** | un-normalized input makes the UT transform non-contractive (backward measured `|o|max = 2.9e18`) | caller contract, documented; tests build inputs with `make_reference_inputs` |
| **One dispatch, six outputs** | `generic_op` returns only the last `io_tensors` entry | pre-allocate all six + `sc`, pass all in `io_tensors`, return the tuple |

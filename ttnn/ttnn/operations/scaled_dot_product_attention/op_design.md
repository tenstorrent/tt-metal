# Operation Design: scaled_dot_product_attention (Flash Attention)

## Blocking Model

Flash Attention computes `O = softmax(Q·Kᵀ·scale [+ mask]) · V` with the **online-softmax
recurrence**: for each Q-block, stream every KV-block once, maintaining a running row-max
`m`, running row-sum `l`, and running output accumulator `O`. The S_q×S_kv score matrix is
**never materialized** — only one B_q×B_kv score block lives in L1 at a time. This is the
load-bearing constraint of the op.

### Axes

| Axis | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| B (batch) | **independent** — each batch computes attention in isolation | (flattened into work-list; no sub-blocking) | whole | folded into the flat `B·H_q·q_num_chunks` work-list split across the grid | knob-turn |
| H_q (query heads) | **independent** — each head computes attention in isolation | (flattened into work-list) | whole | folded into the flat work-list split across the grid | knob-turn |
| S_q (query sequence) | **independent** — each Q-block produces its own output rows; no cross-Q-block dependency | `q_chunk_tiles` (Sq_chunk_t, tiles of 32 rows) | **4 tiles (128 tokens)** | Q-blocks are the unit of the flat work-list; disjoint Q-blocks → disjoint output, spread across the grid | knob-turn |
| S_kv (key/value sequence) | **dependent** — softmax + weighted sum span all of S_kv; a result row is a reduction over this axis | `k_chunk_tiles` (Sk_chunk_t, tiles of 32 cols) | **4 tiles (128 tokens)** | **single-core** in phase-1: streamed as the inner online-softmax loop within the core that owns the Q-block | **scheme-change** (cross-core split-KV / flash-decoding) |
| D (head_dim) | contraction of QKᵀ / free dim of O — not a cross-core split axis | matmul K-blocking `num_k_blocks = DHt` for QKᵀ; free width for PV | whole (DHt) | not split — lives inside `matmul_block` | knob-turn (matmul block size) |

`DHt = ceil(D/32)`, `Sq_chunk_t = q_chunk_tiles`, `Sk_chunk_t = k_chunk_tiles`.

### Bandwidth ranking of candidate splits (qualitative, bytes/fan-out)

1. **Split (B, H_q, S_q-blocks) across cores — CHOSEN.** Independent, no cross-core combine.
   Each core reads its own Q-block once and streams KV. Work-item count
   `= B·H_q·ceil(S_q/B_q)` — for the target profile (B=1, H=10, S=9472, B_q=128 →
   `10·74 = 740` items) this over-fills a 110-core grid with no communication. This is the
   preferred split: zero combine traffic.
2. **Split S_kv across cores (split-KV / flash-decoding) — LAMP.** Dependent axis: each core
   reduces a KV slice into a partial `(m, l, O)`, then partials are combined across cores by
   rescaling with the global max (an all-gather + rescale epilogue). Moves the partial-state
   bytes over the NoC. Only wins when candidate #1 under-fills the grid (decode: S_q small;
   or very few heads with long context). Not needed at phase-1 shapes; left as a lamp.

### Operand-reuse check (for the chosen split over (B, H_q, S_q))

- **Q** varies along S_q, H_q, B → each core reads its own Q-block. No reuse.
- **K, V** do **not** vary along S_q. Every core owning a different Q-block of the *same*
  (batch, head) re-reads the identical K/V from DRAM → **K/V are reuse-shared by construction
  of the split.** For GQA/MQA they additionally do not vary across a query-head group. **LAMP:
  broadcast/forward K/V** — read each KV-block once on an injector and mcast (`mcast_pipe`) to
  the cores sharing that (batch, head), or a store-and-forward chain, instead of N cores each
  pulling from DRAM (perf pattern `shared_input_reuse`, ~1.71× at 22 cores).

### Buffer-depth knobs (per streaming CB)

| CB | Depth knob | Phase-1 value |
|----|-----------|---------------|
| `cb_k_in`, `cb_v_in`, `cb_mask_in` | `kv_buffer_factor` | 2 (double-buffer — overlap KV DRAM read with compute) |
| `cb_q_in` | `q_buffer_factor` | 1 (held resident across the whole KV loop; ×2 only if a core owns >1 Q-block) |
| intermediate score / stats / accumulator CBs | — | 1 (sequential compute helpers, full-block sized) |

### Lamps (scheme-changes phase-1 leaves room for)

1. **Causal masking (`mask_mode=causal`)** — the KV-loop upper bound is already a parameter, so
   causal only *truncates* it (block-skip all strictly-future KV-blocks ≈ half the KV work for
   a decoder) and stamps an on-device triangular −inf mask on the single diagonal-straddling
   block. No new loop nest. Arms the `{causal, cross}` EXCLUSION (causal requires S_q==S_kv) and
   the `is_causal + attn_mask` mutual-exclusion ValueError.
2. **Split-KV / flash-decoding (dependent-axis cross-core combine)** — reachable because the
   online recurrence already produces a per-core partial `(m, l, O)`; the combine is an
   all-gather + max-rescale epilogue. Physical realization: `WIDTH_SHARDED` / `BLOCK_SHARDED` KV.
3. **KV broadcast/forward (reuse-shared mcast)** — reachable because the reader already fetches
   each KV-block per (batch, head); swap the per-core DRAM pull for one injector read + mcast.
4. **Sharded inputs** — `HEIGHT_SHARDED` Q/O is the logical S_q-height-shard made physical: a
   knob-turn (CB backed on the sharded buffer via `ttnn.cb_descriptor_from_sharded_tensor`,
   zero-copy from the core's own L1 — **no NoC re-read**; work stays per-core). `WIDTH_/BLOCK_SHARDED`
   KV is the split-KV scheme-change (#2).
5. **Precision** — `float32`, `bfloat8_b`, and `fp32_dest_acc_en=False` (the 16-bit-DEST
   streaming path). Phase-1 ships the maxed corner (bf16 @ `fp32_dest_acc_en=True`).

All knobs (`q_chunk_tiles`, `k_chunk_tiles`, `kv_buffer_factor`, `q_buffer_factor`, the grid) are
**parameters, never inlined constants**. Each has a single host-side source of truth; all CB
page counts, loop bounds, and derived kernel args are computed *from* those sources.

---

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul → softmax → matmul, streamed) |
| Goal | Memory-efficient exact attention via the Flash Attention online-softmax recurrence; the S_q×S_kv score matrix is never materialized. |
| Math | `O = softmax(Q·Kᵀ·scale [+ mask], dim=-1) · V`, per (batch, head), online over KV-blocks |
| Mode | Hybrid (generic-op / `ProgramDescriptor`) |
| References | Tri Dao "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"; prior `ttnn/cpp/ttnn/operations/transformer/sdpa/` (git `c1d11e9f0c^`); perf catalog `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `matmul_output_subblock`, `compute_block_size`, `shared_input_reuse`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| query | ttnn.Tensor | yes | 4D (B,H_q,S_q,D), bf16, TILE, tile-aligned | — | tensor |
| key | ttnn.Tensor | yes | 4D (B,H_kv,S_kv,D); H_q%H_kv==0 | — | tensor |
| value | ttnn.Tensor | yes | 4D (B,H_kv,S_kv,D); shape == key | — | tensor |
| attn_mask | ttnn.Tensor | no | (B,1,S_q,S_kv) or (B,H_q,S_q,S_kv), additive, bf16, TILE | None | tensor |
| is_causal | bool | no | {False} in Phase-1 (True arms with causal refinement) | False | scalar |
| scale | float | no | any float; None → 1/sqrt(D) | None | RT (folded into exp) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | no | resolved via `default_compute_kernel_config()`; Phase-1 requires `fp32_dest_acc_en=True` | None | drives compute config |
| q_chunk_tiles | int (host param) | internal | ≥1, ≤ceil(S_q/32) | 4 | CT (per-core) |
| k_chunk_tiles | int (host param) | internal | ≥1, ≤ceil(S_kv/32) | 4 | CT (per-core) |

`validate()` derives `mask_mode` (`causal` if is_causal; `custom` if attn_mask given; else `none`;
is_causal+attn_mask → ValueError) and `scale_mode` (`explicit` if scale given else `auto`). It reads
`compute_kernel_config.fp32_dest_acc_en` and rejects `fp32`+`False` (EXCLUSION, arms with dtype
refinement); it must never silently force `True`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q (B,H_q,S_q,D); K,V (B,H_kv,S_kv,D), H_q%H_kv==0; optional mask (B,{1,H_q},S_q,S_kv) |
| Dtype | bfloat16 (Phase-1); float32 / bfloat8_b are refinements |
| Layout | TILE_LAYOUT only |
| Memory | DRAM or L1 interleaved (Phase-1); sharded is a lamp |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H_q, S_q, D) — same as Q |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | interleaved DRAM/L1 |

## Dataflow Strategy

Per core, for each Q-block it owns:

1. Reader reads the Q-block once (`cb_q_in`, held resident), plus the reduce scaler (`cb_scaler`,
   value 1.0, bf16) and column-of-ones (`cb_col_identity`) for the final in-tile row-sum.
2. Reader streams each KV-block: K-block → `cb_k_in`, V-block → `cb_v_in`, and (mask_mode=custom)
   the corresponding mask block → `cb_mask_in`, all double-buffered.
3. Compute runs the online-softmax recurrence per KV-block (below), keeping running `(m, l, O)`
   in ping-pong intermediate CBs.
4. After the KV loop, compute normalizes `O /= l` and writes the Q-block's output to `cb_out`;
   writer drains `cb_out` to DRAM.

**GQA/MQA head broadcasting** is handled entirely in the reader by integer-dividing the query-head
index: `k_head = nq / (H_q / H_kv)`. No K/V data duplication — each Q-head work-item addresses its
group's shared KV head. (Supports kv_heads_mode ∈ {mha, gqa, mqa} with no extra compute work.)

**Cross-attention** (S_q ≠ S_kv) falls out naturally: the KV loop count is `ceil(S_kv/B_kv)`,
independent of `ceil(S_q/B_q)`.

**Tensix-to-Tensix contract for the unlocked lamps** (phase-1 does not use these):
- *Split-KV*: cores sharing a (batch, head, Q-block) each own a disjoint KV slice, produce a partial
  `(m, l, O)`, then one designated core gathers the partials (semaphore-signalled), rescales each by
  `exp(m_core − m_global)`, and sums — a max-consistent all-reduce over the running state.
- *KV broadcast*: an injector core reads each KV-block once and `mcast_pipe::SenderPipe::send()`s it
  (double-buffered prefetch) to the receiver cores sharing that (batch, head); receivers
  `ReceiverPipe::receive()` into `cb_k_in`/`cb_v_in` instead of reading DRAM.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **Q-block** = (batch, query-head, Q-chunk) triple |
| Grid | `device.compute_with_storage_grid_size()` (runtime parameter — core count never inlined) |
| Per-core work | flatten `total_q_blocks = B · H_q · q_num_chunks` (with `q_num_chunks = ceil(S_q / (32·q_chunk_tiles))`), then `ttnn.split_work_to_cores(grid, total_q_blocks)`; each core gets a contiguous `[q_start, q_start+q_count)` slice as runtime args |
| Remainder | `core_group_1` cores do `units_per_core_g1`, `core_group_2` cores do one fewer (`units_per_core_g2`); handled by `split_work_to_cores` |

Tile geometry is alignment-aware and per-image: `q_num_chunks = ceil(S_q / (32·q_chunk_tiles))`,
`k_num_chunks = ceil(S_kv / (32·k_chunk_tiles))`, `DHt = ceil(D/32)`. (Phase-1 only exercises
tile-aligned shapes, but `ceil` is used everywhere so a later alignment refinement — masking the
padded tail rows/cols — needs no formula rewrite.)

### Regime selection

Phase-1 selects the compute kernel path on **mask_mode** (a compile-time regime):
- `none` — no mask CB, QKᵀ score block feeds softmax directly.
- `custom` — `cb_mask_in` present; compute adds the streamed mask block to the score block before
  the running row-max.

Predicate: `attn_mask is not None → custom`, else `none`. (`causal` is a lamp; when it arms it adds a
third regime that generates the triangular mask on-device and truncates the KV loop.) Because the
mask regime is fixed at compile time from the presence of the mask tensor — not grid-dependent — no
grid-varying regime hazard exists; both `none` and `custom` are exercised by the golden matrix and
the acceptance test.

## Circular Buffers

Sizes are **per-block**, functions of `q_chunk_tiles` (Q), `k_chunk_tiles` (K), `DHt`, and the
buffer-depth knobs — never of the full S_q/S_kv. `T = tile size for the CB's dtype`.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| cb_q_in | 0 | tile | `q_chunk_tiles·DHt · q_buffer_factor` | bf16 | reader | compute | held across KV loop |
| cb_k_in | 1 | tile | `k_chunk_tiles·DHt · kv_buffer_factor` | bf16 | reader | compute | per KV-block |
| cb_v_in | 2 | tile | `k_chunk_tiles·DHt · kv_buffer_factor` | bf16 | reader | compute | per KV-block |
| cb_mask_in | 3 | tile | `q_chunk_tiles·k_chunk_tiles · kv_buffer_factor` | bf16 | reader | compute | per KV-block (custom only) |
| cb_scaler | 4 | tile | 1 | bf16 (packed row-0) | reader | compute | whole op |
| cb_col_identity | 5 | tile | 1 | bf16 | reader | compute | whole op |
| cb_qk_scores | 24 | tile | `q_chunk_tiles·k_chunk_tiles` | fp16_b | compute | compute | per KV-block (full block) |
| cb_max_A / cb_max_B | 25/26 | tile | `q_chunk_tiles` | fp16_b | compute | compute | running max (ping-pong) |
| cb_sum_A / cb_sum_B | 27/28 | tile | `q_chunk_tiles` | fp16_b | compute | compute | running sum (ping-pong) |
| cb_out_accum_A / _B | 29/30 | tile | `q_chunk_tiles·DHt` | fp16_b | compute | compute | output accumulator (ping-pong) |
| cb_exp_max_diff | 31 | tile | `q_chunk_tiles` | fp16_b | compute | compute | rescale factor |
| cb_out | 16 | tile | `q_chunk_tiles·DHt` | bf16 | compute | writer | per Q-block |

Notes: intermediate stats/score/accumulator CBs are produced and consumed entirely within the
compute kernel (one producer thread, one consumer thread) and are sized to the **full block** the
next sequential compute helper consumes (helpers can't pipeline — each owns all 3 TRISCs). Score and
stats CBs use `fp16_b` even under `fp32_dest_acc_en=True` (matches prior SDPA; DEST accumulation, not
the CB, carries the fp32 precision). `cb_mask_in` exists only in the `custom` regime.

## API Mapping

Base: `ttnn/cpp/ttnn/kernel_lib/`. Every helper's block/chunk template param is a tunable knob.

| Phase | Type | Function | File:Line | Template Params / Args (knobs **bold**) | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|-----------------------------------------|----------|-----------|--------------|
| boot | helper | `mm_block_init()` + `compute_kernel_hw_startup()` | matmul_block_helpers.hpp:99-103,321-326 | init CBs | — | — | once at kernel top, before any helper |
| QKᵀ | helper | `matmul_block<transpose=true>` | matmul_block_helpers.hpp:335-367 | `MatmulBlockShape::of(...)` **block size = q_chunk_tiles×k_chunk_tiles**, `num_k_blocks=DHt`; `in0_policy=WaitAndRetainOnLastBlock` (retain Q across KV) | cb_q_in, cb_k_in | cb_qk_scores | out CB distinct from inputs; subblocking derived by tuner from block size + fp32_dest flag |
| mask add (custom) | helper | `add<...>` (in-place: out CB == A CB) | eltwise_convenience.hpp:42-43 | `EltwiseShape::grid(q_chunk_tiles, k_chunk_tiles)` | cb_qk_scores, cb_mask_in | cb_qk_scores | add before row-max |
| row-max | helper | `reduce<MAX, REDUCE_ROW, ..., WaitUpfrontNoPop>` | reduce_helpers_compute.hpp:522-538 | `ReduceInputBlockShape::of(q_chunk_tiles, k_chunk_tiles)`; `WaitUpfrontNoPop` keeps scores resident for the sub/exp | cb_qk_scores, cb_scaler | cb_max_* | scaler=1.0, pool-type-aware; MAX+ROW cross-block CB-accumulate is restricted → do running-max update in eltwise |
| running-max update | helper | `binary_sfpu<BinaryMax<>,...>` | eltwise_convenience.hpp:73-74; eltwise_binary_sfpu_minmax.hpp:15-26 | DEST-to-DEST max(m_prev, m_new) | cb_max_A, cb_max_B | cb_max_* | ping-pong |
| exp(scaled, shifted) | helper | `sub<...,BroadcastDim::Col>` then `unary<Exp<>,...>` | eltwise_convenience.hpp:45-46,65-66; eltwise_math.hpp:20-22 | scale 1/sqrt(D) folded into exp via `MulUnary`/scaler; row-max (N,1) broadcast across cols = `BroadcastDim::Col` | cb_qk_scores, cb_max_* | cb_qk_scores (P, in-place) | scale folded here, not into QKᵀ |
| row-sum | helper | `reduce<SUM, REDUCE_ROW, ...>` | reduce_helpers_compute.hpp:522-538 | `ReduceInputBlockShape::of(q_chunk_tiles, k_chunk_tiles)` | cb_qk_scores, cb_scaler | cb_sum_* | scaler=1.0 |
| PV matmul | helper | `matmul_block<packer_l1_acc=true>` | matmul_block_helpers.hpp:335-367 | **block size = q_chunk_tiles×DHt**, `num_k_blocks=k_chunk_tiles`; accumulate into O | cb_qk_scores, cb_v_in | cb_out_accum_* | L1 accumulate across KV-blocks; subblock ≤4 under fp32 DEST |
| rescale prior state | helper | `sub` + `unary<Exp<>>` + `mul<...,BroadcastDim::Col>` | eltwise_convenience.hpp:45-49,65-66 | correction `exp(m_prev−m_new)` applied to l and O (Col-broadcast) | cb_max_*, cb_sum_*/cb_out_accum_* | cb_sum_*/cb_out_accum_* | only when processed_kv>0 |
| final normalize | helper | `reduce` `post_reduce_op=Recip` then `mul<...,BroadcastDim::Col>` | reduce_helpers_compute.hpp:499-508; eltwise_math.hpp:32-34 | 1/l then O·(1/l) | cb_sum_*, cb_out_accum_* | cb_out | after KV loop |
| scaler fill | helper | `calculate_and_prepare_reduce_scaler<cb_scaler, MAX/SUM, REDUCE_ROW>` | reduce_helpers_dataflow.hpp:97-99 | pool-type-aware overload (scaler=1.0) | — | cb_scaler | reader-side, bf16 packed |
| Q/K/V/mask/out I/O | raw_api | `TensorAccessor` async reads/writes | tech_reports/tensor_accessor/tensor_accessor.md | interleaved DRAM addressing; GQA head via `k_head=nq/(H_q/H_kv)` | DRAM | cb_*_in / DRAM | issue block of ~4-8 reads then one barrier (double_buffer pattern) |

**Helpers considered and rejected — none.** Every compute phase maps to a kernel_lib helper. The
only raw-API use is interleaved tensor I/O (reader/writer), which is TensorAccessor's domain, not a
compute helper's.

## Compute Phases

Per KV-block within a Q-block's KV loop (running `(m, l, O)` in ping-pong CBs):

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 1 | QKᵀ (transpose K) | matmul_block | cb_q_in (q·DHt, held), cb_k_in (k·DHt) | cb_qk_scores (q·k) | Q retained; K popped |
| 2 | + mask (custom only) | add (in-place) | cb_qk_scores, cb_mask_in | cb_qk_scores | mask popped |
| 3 | row-max → update running m | reduce MAX ROW + BinaryMax | cb_qk_scores (kept), cb_max_prev | cb_max_cur | scores resident |
| 4 | exp((S−m)·scale) → P; row-sum | sub(Col)+Exp; reduce SUM ROW | cb_qk_scores, cb_max_cur | cb_qk_scores (P), cb_sum_cur | P in place |
| 5 | P·V → accumulate O | matmul_block (L1 acc) | cb_qk_scores (P), cb_v_in | cb_out_accum_cur | P, V popped |
| 6 | rescale prior l, O by exp(m_prev−m_cur) | sub+Exp+mul(Col) | cb_max_*, cb_sum_prev, cb_out_accum_prev | cb_sum_cur, cb_out_accum_cur | ping-pong swapped |
| 7 (post-loop) | normalize O /= l | reduce Recip + mul(Col) | cb_sum_final, cb_out_accum_final | cb_out | ready for writer |

CB sync: for every CB, producer push count == consumer wait count. Reader pushes `k_num_chunks`
K/V/mask blocks per Q-block; compute waits exactly that many. `cb_q_in` pushed once per Q-block, held
(retain policy) across the KV loop, popped once. `cb_out` pushed once per Q-block, drained by writer.

## Key Risks and Gotchas

- **Never materialize S_q×S_kv.** `cb_qk_scores` is sized `q_chunk_tiles·k_chunk_tiles` (one block),
  never the full matrix. This is the op's defining constraint.
- **Full-block intermediates.** `cb_qk_scores` and `cb_out_accum_*` must hold the whole block the
  next sequential helper consumes, or the producing matmul blocks on `cb_reserve_back`.
- **Scaler CB is bf16, row-0 packed, via the pool-type-aware overload** (`MAX,REDUCE_ROW` and
  `SUM,REDUCE_ROW` use different fills).
- **DEST budget.** `fp32_dest_acc_en=True` → 8 fp16 / 4 fp32 DEST tiles. Cap matmul subblock at 4
  under fp32 DEST; expose *block size* + the fp32 flag and let the matmul tuner derive subblocking.
- **fp32-DEST precision.** PV matmul has `num_k_blocks = k_chunk_tiles > 1` → K-accumulation; keep
  `fp32_dest_acc_en=True` (HiFi2), never HiFi4+fp32-dest with bf16 inputs (silent corruption, #38306).
- **Ping-pong stats must not alias.** `max_A/B`, `sum_A/B`, `out_accum_A/B` swap each KV-block;
  matmul in0/in1/out CBs must all be distinct (aliasing corrupts FIFO state).
- **scale folded into exp**, not into QKᵀ — matches the online-softmax numerically-exact form.
- **GQA/MQA is reader addressing only** — no data duplication; `k_head = nq/(H_q/H_kv)`.
- **Persist across phases**: `cb_q_in` (whole KV loop), running `(m, l, O)` CBs (whole KV loop),
  `cb_scaler`/`cb_col_identity` (whole op).

## Structural impossibilities

None beyond `feature_spec.py`'s `INVALID = []`. SDPA is TILE-only by design, so the canonical
`bf8b + ROW_MAJOR` rule is vacuous. No op-specific INVALID candidate identified.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| exp shift (S − m) | sub | cb_qk_scores [q·32, k·32] All | cb_max_cur (q,1) Col0 | Col (row-max broadcast across columns) |
| rescale l | mul | cb_sum_prev (q,1) Col0 | cb_exp_max_diff (q,1) Col0 | Col |
| rescale O | mul | cb_out_accum_prev [q·32, DHt·32] All | cb_exp_max_diff (q,1) Col0 | Col |
| normalize | mul | cb_out_accum [q·32, DHt·32] All | 1/l (q,1) Col0 | Col |
| mask add (custom) | add | cb_qk_scores All | cb_mask_in All | None (full block) |

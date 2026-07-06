# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused |
| Goal | Compute softmax(Q @ K^T * scale) @ V using the Flash Attention algorithm — tiled, online softmax, O(S) memory. The full S_q × S_kv attention matrix is never materialized. |
| Math | `output[b,h,i,d] = (Σ_j exp((Q[b,h,i,:]·K[b,h,j,:]*scale + mask[i,j]) - m_i) / l_i) * V[b,h,j,:]` where `m_i` and `l_i` are the running row-max and row-sum maintained by online softmax across KV blocks. |
| Mode | Hybrid |
| References | Tri Dao, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"; `torch.nn.functional.scaled_dot_product_attention` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| query | ttnn.Tensor | yes | (B,H,S_q,D), bf16, TILE | — | RT |
| key | ttnn.Tensor | yes | (B,H,S_kv,D), bf16, TILE | — | RT |
| value | ttnn.Tensor | yes | (B,H,S_kv,D), bf16, TILE | — | RT |
| attn_mask | ttnn.Tensor | no | (B,1,S_q,S_kv) or (B,H,S_q,S_kv), bf16, TILE | None | RT |
| is_causal | bool | no | {True, False} | False | CT |
| scale | float | no | any float | None (→ 1/sqrt(D)) | RT |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape (Q) | (B, H, S_q, D), rank 4, S_q % 32 == 0, D % 32 == 0 (Phase 0) |
| Shape (K) | (B, H, S_kv, D), rank 4, S_kv % 32 == 0, D == Q's D |
| Shape (V) | (B, H, S_kv, D), rank 4, S_kv == K's S_kv, D == K's D |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE_LAYOUT |
| Memory | DRAM (interleaved) or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H, S_q, D) — same as Q |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | DRAM (interleaved) |

## Dataflow Strategy

Data enters as tiled bf16 tensors in DRAM. The reader kernel (NCRISC) streams Q, K, V tile blocks from DRAM into L1 circular buffers. No tilize/untilize is needed — inputs are already TILE_LAYOUT.

The compute kernel (TRISC) implements the Flash Attention algorithm:
- **Outer loop** over Q blocks (B_q tiles along S_q).
- **Inner loop** over KV blocks (B_kv tiles along S_kv).
- For each (Q_block, KV_block) pair, the compute produces a score block S = Q_block @ K_block^T * scale (B_q × B_kv tiles in L1 CB — never the full S_q × S_kv matrix), applies online softmax (running max m_i, running sum l_i, running output O_i), and accumulates P @ V_block into O_i.
- Running state (m_i, l_i, O_i) persists in L1 CBs across KV block iterations within a Q block.
- After all KV blocks for a Q block, O_i is normalized by l_i and written to the output CB.

The writer kernel (BRISC) reads output tiles from the output CB and writes them to DRAM.

No inter-Tensix communication is required — each (B, H) pair is fully independent and processed on a single core.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One (B, H) pair — the full attention computation for one batch-head |
| Grid | `ttnn.split_work_to_cores(grid_size, B * H)` — distributes (B,H) pairs across available cores |
| Per-core work | One or more (B,H) pairs; each pair processes S_q_t / B_q Q-blocks × S_kv_t / B_kv KV-blocks |
| Remainder | If B*H doesn't divide evenly, some cores get one extra (B,H) pair — handled by `split_work_to_cores` core_group_1 / core_group_2 split |

Block-size parameters (compile-time, configurable per kernel build):
- **B_q** = Q block size in tiles along S_q (default 4 = 128 rows). Outer-loop tile count.
- **B_kv** = KV block size in tiles along S_kv (default 4 = 128 rows). Inner-loop tile count.
- **D_t** = head_dim in tiles = D / 32. The K/N dimension of the matmuls.

These defaults are chosen so the score block (B_q × B_kv = 16 tiles = 32 KB) and running output (B_q × D_t tiles) fit comfortably in L1 alongside Q/K/V stream buffers.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| cb_q | 0 | tile_size(bf16) | 2 × (B_q × D_t) | bfloat16 | Reader | Compute (QK^T matmul in0) | Streamed per KV block; re-pushed each iteration |
| cb_k | 1 | tile_size(bf16) | 2 × (B_kv × D_t) | bfloat16 | Reader | Compute (QK^T matmul in1, transpose) | Streamed per KV block; consumed and freed |
| cb_v | 2 | tile_size(bf16) | 2 × (B_kv × D_t) | bfloat16 | Reader | Compute (PV matmul in1) | Streamed per KV block; consumed and freed |
| cb_attn_mask | 3 | tile_size(bf16) | 2 × (B_q × B_kv) | bfloat16 | Reader | Compute (eltwise add to scores) | Streamed per KV block; only when mask_mode=custom |
| cb_scale | 4 | tile_size(bf16) | 1 | bfloat16 | Reader | Compute (scale tile for score scaling) | Pushed once per Q-block; consumed per KV block |
| cb_output | 16 | tile_size(bf16) | 2 × (B_q × D_t) | bfloat16 | Compute (normalized output) | Writer | Streamed per Q block |
| cb_scores | 24 | tile_size(bf16) | B_q × B_kv | bfloat16 | Compute (QK^T matmul) | Compute (eltwise + reduce + PV matmul) | Full block — sequential helper intermediate. Holds scores then P (in-place exp). Also reused as PV matmul output after P is consumed. |
| cb_m | 25 | tile_size(bf16) | max(2, B_q) | bfloat16 | Compute (running max) | Compute (BinaryMax, factor_old, exp) | Persists across all KV blocks within a Q block |
| cb_l | 26 | tile_size(bf16) | max(2, B_q) | bfloat16 | Compute (running sum) | Compute (rescale, rowsum accumulation) | Persists across all KV blocks within a Q block |
| cb_o | 27 | tile_size(bf16) | max(2, B_q × D_t) | bfloat16 | Compute (running output) | Compute (rescale, PV accumulation, normalize) | Persists across all KV blocks within a Q block |
| cb_m_new | 28 | tile_size(bf16) | max(2, B_q) | bfloat16 | Compute (block max / m_new) | Compute (BinaryMax, exp, copy) | Temporary per KV block |
| cb_psum | 29 | tile_size(bf16) | max(2, B_q) | bfloat16 | Compute (rowsum reduce) | Compute (l_i add) | Temporary per KV block |
| cb_pv | 30 | tile_size(bf16) | max(2, B_q × D_t) | bfloat16 | Compute (PV matmul) | Compute (O_i add) | Temporary per KV block |
| cb_scaler | 31 | tile_size(bf16) | 2 | bfloat16 | Reader (prepare_reduce_scaler) | Compute (reduce MAX, reduce SUM) | 2 pages: tile 0 = MAX scaler (row-0 fill), tile 1 = SUM scaler (col-0 fill). Pushed per KV block. |

CB sizing rationale:
- **cb_scores (24)**: Full B_q × B_kv block — matmul_block and eltwise_chain/reduce are sequential helpers occupying all TRISCs, so the intermediate must hold the entire block. After in-place exp, it holds P. After rowsum reduce (WaitUpfrontNoPop preserves P for PV matmul), P is consumed by PV matmul. Then cb_scores is reused as PV matmul output (matmul_block writes to it, distinct from in0=cb_pv).
- **cb_m, cb_l (25-26)**: B_q tiles each, persist across KV blocks. ≥ 2 pages for in-place eltwise (pop-before-reserve).
- **cb_o (27)**: B_q × D_t tiles, persists across KV blocks. ≥ 2 pages for in-place eltwise.
- **cb_m_new, cb_psum (28-29)**: B_q tiles each, temporary per KV block.
- **cb_pv (30)**: B_q × D_t tiles, temporary per KV block. Holds P copy for PV matmul input.
- **cb_scaler (31)**: 2 tiles — different fill patterns for MAX (row-0) vs SUM (col-0) per `reduce_helpers_dataflow.hpp:L65-67`.

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------------|----------|-----------|--------------|
| QK^T matmul | helper | `matmul_block` | `matmul_block_helpers.hpp:L333-365` | `<transpose=true, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, in0_policy=WaitAndPopPerKBlock, in1_policy=WaitAndPopPerKBlock>` shape=`MatmulBlockShape::of(B_q, B_kv, B_q, B_kv, D_t, 1)` | cb_q, cb_k | cb_scores | Boot: `mm_block_init(cb_q, cb_k, cb_scores, transpose=1, ct_dim=B_kv, rt_dim=B_q, kt_dim=D_t)`. interm_buf=cb_scores (num_k_blocks=1 placeholder). TileRowMajor so tile order matches downstream eltwise/reduce row-major iteration. |
| Scale scores | helper | `transform_in_place` | `eltwise_convenience.hpp:L227-242` | `<cb_scores, MulUnary<>{scale_bits}>` shape=`EltwiseShape::grid(B_q, B_kv)` | cb_scores | cb_scores (in-place) | `MulUnary` from `eltwise_math.hpp` — multiplies each tile by a compile-time scalar. Requires `compute_kernel_hw_startup` before first use. |
| Apply mask | helper | `eltwise_chain` | `eltwise_chain.hpp:L578-579` | `BinaryFpu<cb_scores, cb_attn_mask, BinaryFpuOp::Add, BroadcastDim::None, Streaming, Streaming>` + `PackTile<cb_scores, Streaming>` shape=`EltwiseShape::grid(B_q, B_kv)` | cb_scores, cb_attn_mask | cb_scores (in-place) | Only when mask_mode=custom. In-place: chain pops input before reserving output. |
| Row-max | helper | `reduce` | `reduce_helpers_compute.hpp:L411-426` | `<MAX, REDUCE_ROW, cb_scores, cb_scaler, cb_m_new, ReduceInputPolicy::WaitUpfrontNoPop>` shape=`ReduceInputBlockShape::of(B_q, B_kv)` | cb_scores, cb_scaler | cb_m_new | WaitUpfrontNoPop: waits all score tiles, does NOT pop — tiles persist for exp phase. Scaler tile 0 (MAX, row-0 fill) consumed. |
| Online max | helper | `eltwise_chain` | `eltwise_chain.hpp:L578-579` | `CopyTile<cb_m, D0, HeldBulk, Block>` + `CopyTile<cb_m_new, D1, Streaming, Block>` + `BinaryMax<D0, D1, D0>` + `PackTile<cb_m_new, D0, Streaming>` shape=`EltwiseShape::tiles(B_q)` | cb_m, cb_m_new | cb_m_new (in-place) | BinaryMax from `eltwise_binary_sfpu.hpp:L71-77`. cb_m read as HeldBulk (all B_q tiles waited upfront, not popped — m_i needed for factor_old). cb_m_new overwritten with m_new = max(m_i, m_block). |
| Exp scores | helper | `eltwise_chain` | `eltwise_chain.hpp:L578-579` | `BinaryFpu<cb_scores, cb_m_new, BinaryFpuOp::Sub, BroadcastDim::Col, Streaming, HeldBulk, ..., D0, Block, Col>` + `Exp<Approx::Fast, Approx::Fast, D0>` + `PackTile<cb_scores, D0, Streaming>` shape=`EltwiseShape::grid(B_q, B_kv)` | cb_scores, cb_m_new | cb_scores (in-place) | BroadcastDim::Col: m_new (B_q×1) broadcast across B_kv columns. cb_m_new HeldBulk+Col (waited upfront, indexed by row, not popped). cb_scores Streaming (popped/pushed per-tile). After this, cb_scores holds P = exp(S - m_new). |
| Copy P | helper | `copy` | `eltwise_convenience.hpp:L172-182` | `<cb_scores, cb_pv>` shape=`EltwiseShape::tiles(B_q × B_kv)` | cb_scores | cb_pv | Pops P from cb_scores (streaming), pushes to cb_pv. Needed because rowsum reduce and PV matmul both need P — reduce uses WaitUpfrontNoPop on cb_scores (the copy), while matmul pops cb_pv. |
| Rescale l_i | helper | `eltwise_chain` | `eltwise_chain.hpp:L578-579` | `CopyTile<cb_l, D0, Streaming>` + `CopyTile<cb_m, D1, Streaming>` + `CopyTile<cb_m_new, D2, Streaming>` + `SubBinary<D1, D2, D1>` + `Exp<Approx::Fast, Approx::Fast, D1>` + `MulBinary<D0, D1, D0>` + `PackTile<cb_l, D0, Streaming>` shape=`EltwiseShape::tiles(B_q)` | cb_l, cb_m, cb_m_new | cb_l (in-place) | SubBinary from `eltwise_binary_sfpu.hpp:L46-52`, MulBinary from `eltwise_binary_sfpu.hpp:L54-60`. Fuses factor_old computation (exp(m_i - m_new)) with l_i scaling. Uses 3 DEST slots (D0, D1, D2) — ≤ DEST_AUTO_LIMIT=4 with fp32_dest_acc_en. Pops cb_m (m_i) and cb_m_new (m_new) per-tile. |
| Rescale O_i | helper | `eltwise_chain` | `eltwise_chain.hpp:L578-579` | `CopyTile<cb_o, D0, Streaming, Block>` + `CopyTile<cb_m, D1, HeldBulk, Col>` + `CopyTile<cb_m_new, D2, HeldBulk, Col>` + `SubBinary<D1, D2, D1>` + `Exp<Approx::Fast, Approx::Fast, D1>` + `MulBinary<D0, D1, D0>` + `PackTile<cb_o, D0, Streaming>` shape=`EltwiseShape::grid(B_q, D_t)` | cb_o, cb_m, cb_m_new | cb_o (in-place) | Same fused factor_old computation. cb_m and cb_m_new as Col+HeldBulk (broadcast across D_t columns). After this, pop cb_m_new (manually: `cb_pop_front(cb_m_new, B_q)`). |
| Update m_i | helper | `copy` | `eltwise_convenience.hpp:L172-182` | `<cb_m_new, cb_m>` shape=`EltwiseShape::tiles(B_q)` | cb_m_new | cb_m | Before copy: pop old m_i from cb_m (`cb_pop_front(cb_m, B_q)`). After copy: pop m_new from cb_m_new (`cb_pop_front(cb_m_new, B_q)` — already popped in rescale O_i if streaming was used, otherwise manual pop). |
| Row-sum | helper | `reduce` | `reduce_helpers_compute.hpp:L411-426` | `<SUM, REDUCE_ROW, cb_scores, cb_scaler, cb_psum, ReduceInputPolicy::WaitUpfrontNoPop>` shape=`ReduceInputBlockShape::of(B_q, B_kv)` | cb_scores (P copy), cb_scaler | cb_psum | WaitUpfrontNoPop on the P copy in cb_scores. Scaler tile 1 (SUM, col-0 fill) consumed. cb_scores P tiles NOT popped (WaitUpfrontNoPop) — but this is the copy; original P in cb_pv is still available. Actually: cb_scores holds the P copy (from Copy P step). The reduce doesn't pop them. Then the copy's source (cb_pv) has the original P. Wait — after Copy P, cb_scores is empty (popped by copy) and cb_pv has P. Let me re-examine: the reduce should read from cb_pv (which has P), not cb_scores. |
| Row-sum (corrected) | helper | `reduce` | `reduce_helpers_compute.hpp:L411-426` | `<SUM, REDUCE_ROW, cb_pv, cb_scaler, cb_psum, ReduceInputPolicy::WaitUpfrontNoPop>` shape=`ReduceInputBlockShape::of(B_q, B_kv)` | cb_pv (P), cb_scaler | cb_psum | WaitUpfrontNoPop: waits all B_q×B_kv P tiles in cb_pv, doesn't pop — P persists for PV matmul. Scaler tile 1 (SUM) consumed. |
| l_i accumulation | helper | `add` | `eltwise_convenience.hpp:L44-61` | `<cb_l, cb_psum, cb_l>` shape=`EltwiseShape::tiles(B_q)` | cb_l, cb_psum | cb_l (in-place) | l_i = l_i + psum. In-place on cb_l. Pops cb_psum. |
| PV matmul | helper | `matmul_block` | `matmul_block_helpers.hpp:L333-365` | `<transpose=false, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, in0_policy=WaitAndPopPerKBlock, in1_policy=WaitAndPopPerKBlock>` shape=`MatmulBlockShape::of(B_q, D_t, B_q, D_t, B_kv, 1)` | cb_pv (P), cb_v | cb_scores (reused as PV output) | in0=cb_pv (P tiles, distinct from out). out=cb_scores (empty after P was copied to cb_pv and rowsum used WaitUpfrontNoPop on cb_pv — wait, rowsum used cb_pv too). Let me re-examine the CB flow. |
| O_i accumulation | helper | `add` | `eltwise_convenience.hpp:L44-61` | `<cb_o, cb_scores, cb_o>` shape=`EltwiseShape::grid(B_q, D_t)` | cb_o, cb_scores (PV output) | cb_o (in-place) | O_i += PV. In-place on cb_o. Pops cb_scores (PV tiles). |
| Normalize | helper | `eltwise_chain` | `eltwise_chain.hpp:L578-579` | `CopyTile<cb_o, D0, Streaming, Block>` + `CopyTile<cb_l, D1, HeldBulk, Col>` + `DivBinary<D0, D1, D0>` + `PackTile<cb_output, D0, Streaming>` shape=`EltwiseShape::grid(B_q, D_t)` | cb_o, cb_l | cb_output | DivBinary from `eltwise_binary_sfpu.hpp:L62-68`. cb_l as Col+HeldBulk (broadcast across D_t). Pops cb_o and cb_l. After all KV blocks for a Q block. |
| Scaler prep | helper | `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:L94-101` | `<cb_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>` and `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` | — | cb_scaler | Reader kernel. Pushes 2 scaler tiles per KV block iteration. MAX scaler (row-0 fill, value 1.0) then SUM scaler (col-0 fill, value 1.0). |
| Scale tile prep | raw_api | `cb_reserve_back` + `fill_tile` + `cb_push_back` | — | — | — | cb_scale | Reader pushes a tile filled with the scale factor (or 1/sqrt(D)) for the `MulUnary` in scale-scores phase. Justification: `MulUnary` takes a compile-time scalar, but scale is runtime. See below. |
| m_i init | raw_api | `cb_reserve_back` + `fill_tile<NEG_INF>` + `cb_push_back` | — | — | — | cb_m | Compute kernel. Initializes running max to -inf. |
| l_i init | raw_api | `cb_reserve_back` + `fill_tile<ZERO>` + `cb_push_back` | — | — | — | cb_l | Compute kernel. Initializes running sum to 0. |
| O_i init | raw_api | `cb_reserve_back` + `fill_tile<ZERO>` + `cb_push_back` | — | — | — | cb_o | Compute kernel. Initializes running output to 0. |

### Helpers considered and rejected (raw-API fallbacks)

#### Scale tile prep (cb_scale)

| Helper considered | File:Line | Mismatch reason |
|---|---|---|
| `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:L94-101` | This helper computes standard reduce scalers (1/N for AVG, 1.0 for SUM/MAX). The flash attention scale factor is `1/sqrt(D)` or a user-provided float — it is not a standard reduce scaler. The helper's signature `calculate_and_prepare_reduce_scaler<dfb_id, pool_type, reduce_dim>()` takes no float argument (it auto-computes from N), and `prepare_reduce_scaler<dfb_id, pool, dim>(scaler_f)` fills in a specific pattern (row-0/col-0) tied to reduce semantics, not a uniform tile fill needed for `MulUnary`. The scale tile needs every element set to the scale value (uniform fill), while reduce scalers fill only row-0 or col-0. Verified at `reduce_helpers_dataflow.hpp:L65-67` (prepare_reduce_scaler) and `L94-101` (calculate_and_prepare_reduce_scaler): neither produces a uniform tile fill. |

#### m_i / l_i / O_i initialization

| Helper considered | File:Line | Mismatch reason |
|---|---|---|
| `FillScalar` (eltwise_chain element) | `eltwise_chain.hpp:L546-547` | `FillScalar<DstSlot>` writes a scalar to a DEST register, not to a CB. It is a chain element used inside `eltwise_chain`, not a standalone function that pushes tiles to a CB. To initialize persistent CBs (cb_m, cb_l, cb_o) with constant tiles before the main loop, raw `cb_reserve_back` + `fill_tile` + `cb_push_back` is the correct approach — it directly writes constant tiles to the CB without needing a chain context. Verified at `eltwise_chain.hpp:L546-547`: `FillScalar` is a chain element (has `init/wait/exec/pop/reserve/push` static methods), not a standalone CB-writing function. |

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | Init: push scaler tiles, scale tile; init m_i=-inf, l_i=0, O_i=0 | raw_api (fill_tile) | — | cb_m (B_q tiles, -inf), cb_l (B_q tiles, 0), cb_o (B_q×D_t tiles, 0), cb_scaler (2 tiles), cb_scale (1 tile) | cb_m/l/o populated; cb_scaler has 2 tiles; cb_scale has 1 tile |
| 1 | QK^T: S = Q_block @ K_block^T | matmul_block<transpose=true> | cb_q (B_q×D_t, streaming), cb_k (B_kv×D_t, streaming) | cb_scores (B_q×B_kv tiles) | cb_scores has score block. cb_q/cb_k consumed. |
| 2 | Scale: S *= scale | transform_in_place (MulUnary) | cb_scores (B_q×B_kv, in-place) | cb_scores (scaled) | cb_scores holds scaled scores |
| 2b | Mask (optional): S += attn_mask | eltwise_chain (BinaryFpu Add) | cb_scores (B_q×B_kv, in-place), cb_attn_mask (B_q×B_kv, streaming) | cb_scores (masked) | cb_scores holds masked scores; cb_attn_mask consumed |
| 3 | RowMax: m_block = rowmax(S) | reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop> | cb_scores (B_q×B_kv, not popped), cb_scaler (tile 0) | cb_m_new (B_q tiles) | cb_scores tiles persist (not popped). cb_m_new has block max. Scaler tile 0 consumed. |
| 4 | OnlineMax: m_new = max(m_i, m_block) | eltwise_chain (BinaryMax) | cb_m (B_q, HeldBulk not popped), cb_m_new (B_q, streaming) | cb_m_new (m_new, in-place) | cb_m still has m_i (not popped). cb_m_new has m_new. |
| 5 | ExpScores: P = exp(S - m_new) | eltwise_chain (Sub+Exp, in-place) | cb_scores (B_q×B_kv, streaming popped/pushed), cb_m_new (B_q, Col HeldBulk not popped) | cb_scores (P, in-place) | cb_scores now holds P. cb_m_new still has m_new (not popped). |
| 6 | CopyP: copy P to cb_pv | copy | cb_scores (B_q×B_kv, streaming popped) | cb_pv (B_q×B_kv tiles) | cb_scores empty (P popped). cb_pv has P. |
| 7 | Rescale l_i: l_i = exp(m_i-m_new) * l_i | eltwise_chain (Sub+Exp+Mul, fused) | cb_l (B_q, in-place), cb_m (B_q, streaming), cb_m_new (B_q, streaming) | cb_l (scaled, in-place) | cb_l has scaled l_i. cb_m and cb_m_new popped (streaming). |
| 8 | Rescale O_i: O_i = exp(m_i-m_new) * O_i | eltwise_chain (Sub+Exp+Mul, fused) | cb_o (B_q×D_t, in-place), cb_m (empty! — already popped in step 7) | — | **PROBLEM**: cb_m was popped in step 7. Need to restructure. |

**Restructured phases 7-8** (fuse l_i and O_i rescaling to avoid double-popping cb_m):

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 7 | Rescale l_i AND O_i (sequential, both read m_i/m_new before popping) | eltwise_chain × 2 (see below) | cb_l (B_q, in-place), cb_o (B_q×D_t, in-place), cb_m (B_q, HeldBulk), cb_m_new (B_q, HeldBulk) | cb_l (scaled), cb_o (scaled) | cb_l and cb_o scaled by factor_old. cb_m and cb_m_new held (not popped). |

**Corrected approach for steps 7-8**: Use HeldBulk for cb_m and cb_m_new in both rescale chains, then manually pop after both are done.

Step 7 (l_i rescale): `eltwise_chain(tiles(B_q), CopyTile<cb_l, D0, Streaming, Block>, CopyTile<cb_m, D1, HeldBulk, Block>, CopyTile<cb_m_new, D2, HeldBulk, Block>, SubBinary<D1,D2,D1>, Exp<D1>, MulBinary<D0,D1,D0>, PackTile<cb_l, D0, Streaming>)` — cb_m and cb_m_new held, not popped.

Step 8 (O_i rescale): `eltwise_chain(grid(B_q,D_t), CopyTile<cb_o, D0, Streaming, Block>, CopyTile<cb_m, D1, HeldBulk, Col>, CopyTile<cb_m_new, D2, HeldBulk, Col>, SubBinary<D1,D2,D1>, Exp<D1>, MulBinary<D0,D1,D0>, PackTile<cb_o, D0, Streaming>)` — cb_m and cb_m_new held, not popped.

After step 8: manually pop cb_m (old m_i) and cb_m_new (m_new). Then copy m_new → m_i for next iteration.

**Full corrected phase table:**

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|------------------------|-------------------|----------------|
| 0 | Init: scaler/scale tiles; m_i=-inf, l_i=0, O_i=0 | raw_api | — | cb_m(B_q,-inf), cb_l(B_q,0), cb_o(B_q×D_t,0), cb_scaler(2), cb_scale(1) | Running state initialized |
| 1 | QK^T: S = Q@K^T | matmul_block<transpose=true,TileRowMajor> | cb_q(B_q×D_t), cb_k(B_kv×D_t) | cb_scores(B_q×B_kv) | cb_scores has scores |
| 2 | Scale: S *= scale | transform_in_place<MulUnary> | cb_scores(in-place) | cb_scores(scaled) | — |
| 2b | Mask (opt): S += mask | eltwise_chain(BinaryFpu Add) | cb_scores(in-place), cb_attn_mask | cb_scores(masked) | cb_attn_mask consumed |
| 3 | RowMax: m_block=rowmax(S) | reduce<MAX,REDUCE_ROW,WaitUpfrontNoPop> | cb_scores(not popped), cb_scaler(tile 0) | cb_m_new(B_q) | cb_scores persists; scaler tile 0 consumed |
| 4 | OnlineMax: m_new=max(m_i,m_block) | eltwise_chain(BinaryMax) | cb_m(HeldBulk), cb_m_new(streaming) | cb_m_new(m_new, in-place) | cb_m still has m_i; cb_m_new has m_new |
| 5 | ExpScores: P=exp(S-m_new) | eltwise_chain(Sub+Exp,in-place) | cb_scores(streaming), cb_m_new(Col,HeldBulk) | cb_scores(P, in-place) | cb_scores has P; cb_m_new still has m_new |
| 6 | CopyP: P→cb_pv | copy | cb_scores(streaming) | cb_pv(B_q×B_kv) | cb_scores empty; cb_pv has P |
| 7 | Rescale l_i: l_i*=exp(m_i-m_new) | eltwise_chain(Sub+Exp+Mul,fused) | cb_l(in-place), cb_m(HeldBulk), cb_m_new(HeldBulk) | cb_l(scaled) | cb_m,cb_m_new still held |
| 8 | Rescale O_i: O_i*=exp(m_i-m_new) | eltwise_chain(Sub+Exp+Mul,fused) | cb_o(in-place), cb_m(Col,HeldBulk), cb_m_new(Col,HeldBulk) | cb_o(scaled) | cb_m,cb_m_new still held |
| 9 | Update m_i: pop old, copy m_new→m_i | copy + manual pop | cb_m_new(streaming) | cb_m(m_new) | cb_m has m_new(=new m_i); cb_m_new empty |
| 10 | RowSum: psum=rowsum(P) | reduce<SUM,REDUCE_ROW,WaitUpfrontNoPop> | cb_pv(P,not popped), cb_scaler(tile 1) | cb_psum(B_q) | cb_pv persists; scaler tile 1 consumed |
| 11 | l_i += psum | add(in-place) | cb_l, cb_psum | cb_l(updated) | cb_psum consumed |
| 12 | PV: PV=P@V | matmul_block<false,TileRowMajor> | cb_pv(P,popped), cb_v(B_kv×D_t) | cb_scores(B_q×D_t, reused) | cb_pv consumed; cb_scores has PV |
| 13 | O_i += PV | add(in-place) | cb_o, cb_scores(PV) | cb_o(updated) | cb_scores consumed |
| 14 | Normalize: O=O_i/l_i | eltwise_chain(DivBinary,broadcast) | cb_o(streaming), cb_l(Col,HeldBulk) | cb_output(B_q×D_t) | cb_o,cb_l consumed; cb_output has result. After all KV blocks only. |
| 15 | Reset: pop cb_m,cb_l,cb_o remnants; reinit for next Q block | raw_api | — | — | Clean state for next Q block |

**Stage checkpoints** (for TDD — what to DPRINT/verify at each phase):

| Phase | Checkpoint | Slice |
|-------|-----------|-------|
| 1 | cb_scores tile 0 (first Q-row, first KV-col score block) | first 4×4 elements of tile (0,0) |
| 2 | cb_scores tile 0 after scaling | first 4×4 of tile (0,0) |
| 3 | cb_m_new tile 0 (rowmax of first row) | first 4×4 of tile 0 |
| 4 | cb_m_new tile 0 (m_new after BinaryMax) | first 4×4 of tile 0 |
| 5 | cb_scores tile 0 (P after exp) | first 4×4 of tile (0,0) |
| 7 | cb_l tile 0 (scaled l_i) | first 4×4 of tile 0 |
| 8 | cb_o tile 0 (scaled O_i) | first 4×4 of tile (0,0) |
| 10 | cb_psum tile 0 (rowsum of P) | first 4×4 of tile 0 |
| 12 | cb_scores tile 0 (PV output) | first 4×4 of tile (0,0) |
| 13 | cb_o tile 0 (updated O_i) | first 4×4 of tile (0,0) |
| 14 | cb_output tile 0 (final output) | first 4×4 of tile (0,0) |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 2b | S += mask | cb_scores: All [B_q, B_kv] | cb_attn_mask: All [B_q, B_kv] | None (elementwise) |
| 5 | S -= m_new | cb_scores: All [B_q, B_kv] | cb_m_new: Col0 [B_q, 1] → broadcast across B_kv cols | Col |
| 8 | O_i *= factor_old | cb_o: All [B_q, D_t] | cb_m: Col0 [B_q, 1] → broadcast across D_t cols | Col |
| 8 | O_i *= factor_old | cb_o: All [B_q, D_t] | cb_m_new: Col0 [B_q, 1] → broadcast across D_t cols | Col |
| 14 | O /= l_i | cb_o: All [B_q, D_t] | cb_l: Col0 [B_q, 1] → broadcast across D_t cols | Col |

## Hardware Constraints

- [ ] CB sync: push count = wait count for every CB
  - cb_scores: matmul pushes B_q×B_kv (phase 1); transform_in_place pops/pushes B_q×B_kv (phase 2); eltwise_chain pops/pushes B_q×B_kv (phase 5); copy pops B_q×B_kv (phase 6); matmul pushes B_q×D_t (phase 12); add pops B_q×D_t (phase 13). Different tile counts at different phases — CB num_pages must accommodate the max (B_q×B_kv).
  - cb_pv: copy pushes B_q×B_kv (phase 6); reduce waits B_q×B_kv (phase 10, NoPop); matmul pops B_q×B_kv (phase 12). Sync: copy pushes B_q×B_kv, reduce waits B_q×B_kv (no pop), matmul waits+pops B_q×B_kv.
  - cb_m: init pushes B_q (phase 0); BinaryMax holds B_q (phase 4); rescale l_i holds B_q (phase 7); rescale O_i holds B_q (phase 8); manual pop B_q (phase 9); copy pushes B_q (phase 9). Sync: each consumer waits for B_q tiles.
  - cb_scaler: reader pushes 2 tiles per KV block; reduce MAX consumes tile 0 (phase 3); reduce SUM consumes tile 1 (phase 10).
- [ ] Reduce scaler CB is bfloat16: cb_scaler uses `ttnn.tile_size(ttnn.bfloat16)` page size and `data_format=ttnn.bfloat16`.
- [ ] Reduce scaler uses pool-type-aware API: `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>` and `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` — explicit PoolType and ReduceDim template args.
- [ ] DEST: max 4 tiles (fp32_dest_acc_en=True, SyncHalf). Eltwise chains use ≤ 3 DEST slots (D0, D1, D2). Matmul subblocks use ≤ 4 tiles (2×2).
- [ ] Sequential helper intermediates sized to full block: cb_scores holds B_q×B_kv tiles (full score/P block); cb_pv holds B_q×B_kv tiles (full P copy); cb_o holds B_q×D_t tiles (full output block).
- [ ] Page sizes aligned to tile size: all CBs use `tile_size(bf16)` = 2048 bytes.
- [ ] All CBs are tile-format (TILE_LAYOUT inputs, no RM CBs).
- [ ] All cb_wait_front calls on same CB use same page count within a single helper call (helpers manage their own wait patterns internally).
- [ ] Helpers are not wrapped with extra CB operations: matmul_block, reduce, eltwise_chain, copy, add, transform_in_place all manage their own CB ops. Manual `cb_pop_front` calls only occur between helper calls (phase 9: pop old m_i; phase 8 post: pop cb_m and cb_m_new after HeldBulk).
- [ ] `compute_kernel_hw_startup()` called before any helper usage: called once at the top of `kernel_main()` with the primary CBs (cb_q, cb_k, cb_scores). Subsequent `compute_kernel_hw_startup` calls before each stage that switches between matmul and eltwise/reduce (multi-stage kernel pattern — one boot per stage per eltwise_chain.hpp:L33-39).
- [ ] `mm_block_init()` called once at boot before any `matmul_block` usage (per `matmul_block_helpers.hpp:L99-103`).

## Precision

| Axis | Phase 0 | Target |
|------|---------|--------|
| dtype | bfloat16 | bfloat16, float32, bfloat8_b |
| fp32_dest_acc_en | True (maxed-out corner) | True, False |
| layout | TILE_LAYOUT | TILE_LAYOUT |
| math_fidelity | HiFi4 (default, not gated) | any |
| math_approx_mode | False (default, not gated) | any |

**`default_compute_kernel_config()`** (single source of truth, exported by the op):
```python
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
```

**EXCLUSIONS** (when dtype refinement lands): `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` — maxed input with non-maxed accumulation is lossy/nonsensical. Refused via EXCLUSIONS (xfail-strict), NOT INVALID.

## Rules (conditional — arm when relevant refinement adds the axis value)

### Masking (applies to all mask modes)

Masking is applied by ADDING a mask term onto the QK^T scores (additive: 0 = attend, -inf = mask out), per score-block as the block is produced and BEFORE the running row-max. Masked positions fall out of the max, the exp, and the row-sum together. Never build the full S_q × S_kv mask (or score) matrix.

### Custom attn_mask (mask_mode=custom, Phase 0)

The mask tiles are the caller's data, streamed in alongside K/V. Add the per-(row,col) mask value onto each score position. Don't inspect or special-case the contents.

### Causal masking (mask_mode=causal — refinement, NOT Phase 0)

When adding "causal" to SUPPORTED["mask_mode"]:
- Derive the mask from `is_causal=True` — generate it on-device, never from a caller tensor or a materialized full mask.
- Three regions per (Q-block, KV-block): blocks entirely in the past are unmasked; blocks entirely in the future are whole-tile -inf and SHOULD be skipped outright (don't run QK^T/softmax/PV — this block-skip is the causal perf win); only the block straddling the diagonal needs a per-element triangular -inf mask.
- Declare `{"mask_mode": "causal", "attention_kind": "cross"}` as an EXCLUSION (validate raises NotImplementedError) — causal requires S_q == S_kv.
- Raise ValueError when `is_causal=True` is combined with a non-None `attn_mask` (mutually exclusive — same as Torch).

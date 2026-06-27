# Operation Design: scaled_dot_product_attention

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute) |
| Goal | Flash Attention: compute softmax(Q @ K^T * scale + mask) @ V using tiled, online-softmax with O(S) memory — the full S_q × S_kv score matrix is NEVER materialized. |
| Math | `scores = Q @ K^T * scale`; if `is_causal`: `scores += causal_mask` (0 on/below diagonal, -inf above); elif `attn_mask`: `scores += attn_mask`; `weights = softmax(scores, dim=-1)`; `output = weights @ V`. Flash recurrence per Q-block: for each KV-block, compute `P = exp(Q_blk @ K_blk^T * scale + mask - m_i)`, update running max `m_i`, running sum `l_i`, and running output `O_i = diag(exp(m_old - m_new)) * O_i_old + P @ V_blk`. |
| Mode | Hybrid |
| References | Tri Dao, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"; `torch.nn.functional.scaled_dot_product_attention` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| query | ttnn.Tensor | Yes | (B, H_q, S_q, D), bf16, TILE_LAYOUT | — | RT |
| key | ttnn.Tensor | Yes | (B, H_kv, S_kv, D), bf16, TILE_LAYOUT | — | RT |
| value | ttnn.Tensor | Yes | (B, H_kv, S_kv, D), bf16, TILE_LAYOUT | — | RT |
| attn_mask | ttnn.Tensor | No (keyword) | (B, 1, S_q, S_kv) or (B, H_q, S_q, S_kv), bf16, TILE | None | RT |
| is_causal | bool | No (keyword) | True/False | False | CT |
| scale | float | No (keyword) | any float | None (= 1/sqrt(D)) | RT |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | No (keyword) | ttnn.ComputeConfigDescriptor | None (→ `default_compute_kernel_config()`) | CT |

## Precision Convention

| Field | Value |
|-------|-------|
| Gated axes | `dtype` and `fp32_dest_acc_en` |
| Phase 0 baseline | bfloat16 + `fp32_dest_acc_en=True` (maxed-out corner) |
| Default config factory | `default_compute_kernel_config()` — single exported source of truth; `None` resolves through it |
| Default values | `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, `math_approx_mode=False` |
| Refinement ordering | maxed-out → lower (bf16 both acc values, then fp32, then bf8b) |
| `fp32_dest_acc_en` handling | Always read from config; reject `False` in Phase 0 via SUPPORTED (not silently forced to True) |
| `float32 + fp32_dest_acc_en=False` | Rejected via EXCLUSIONS (when fp32 dtype lands) |
| `math_fidelity` / `math_approx_mode` | Not swept, not rejected — accepted at any value (HiFi4 is the test default) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape (Q) | (B, H_q, S_q, D), rank 4, S_q and D divisible by 32 (Phase 0: tile_aligned) |
| Shape (K) | (B, H_kv, S_kv, D), same B and D as Q; S_kv may differ (cross-attention); H_kv may differ (GQA/MQA) |
| Shape (V) | (B, H_kv, S_kv, D), same B, H_kv, S_kv, D as K |
| Shape (attn_mask) | (B, 1, S_q, S_kv) broadcast across H, or (B, H_q, S_q, S_kv); only when is_causal=False |
| Dtype | bfloat16 for Q, K, V, attn_mask (Phase 0) |
| Layout | TILE_LAYOUT |
| Memory | DRAM or L1 (interleaved) |
| GQA/MQA constraint | H_q % H_kv == 0 (Q heads must be a multiple of KV heads) |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H_q, S_q, D) — same as Q |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | interleaved DRAM (default) or caller-specified |

## Algorithm (Flash Attention v2 recurrence)

The kernel processes one (batch, head) pair per core (or multiple per core in sequence). For GQA/MQA (H_q != H_kv), the reader replicates K/V head tiles to match Q's head count — each Q-head maps to a KV-head group via `kv_head_idx = q_head_idx // (H_q // H_kv)`. For each (B, H) unit:

1. **Initialize** running state: `m_i = -inf` (per row of Q-block), `l_i = 0` (per row of Q-block), `O_i = 0` (per row of Q-block, D-wide). These live in L1 CBs, sized to one Q-block (B_q tiles tall), NOT the full sequence.

2. **Outer loop over Q-blocks** (B_q tiles per block, recommended 128 rows = 4 tiles): For each Q-block:
   - Load Q-block into L1 (retained for all KV iterations).
   - Initialize `m_i, l_i, O_i` for this Q-block.
   - **Inner loop over KV-blocks** (B_kv tiles per block, recommended 128 cols = 4 tiles): For each KV-block:
     - Compute `S_blk = Q_blk @ K_blk^T * scale` (matmul, K_blk transposed). Score block is B_q × B_kv tiles — this is the only score storage, NOT full S_q × S_kv.
     - Apply mask additively to S_blk (custom attn_mask tiles or causal — Phase 0: custom mask only; causal is a refinement).
     - Compute row-max of S_blk, update running max: `m_new = max(m_i, max(S_blk))`.
     - Rescale running sum and output: `alpha = exp(m_i - m_new)`, `l_i = alpha * l_i`, `O_i = alpha * O_i`.
     - Compute `P_blk = exp(S_blk - m_new)` (element-wise exp).
     - Compute row-sum of P_blk: `l_blk = sum(P_blk, dim=-1)`.
     - Update running sum: `l_i = l_i + l_blk`.
     - Accumulate output: `O_i = O_i + P_blk @ V_blk`.
     - Update `m_i = m_new`.
   - Write final `O_i` for this Q-block to output.

### Online softmax numerical precision

All accumulator operations (m_i, l_i, O_i, score computation, exp) use fp32 DEST accumulation (`fp32_dest_acc_en=True` in Phase 0). The online recurrence is mathematically equivalent to the two-pass formulation when implemented in fp32 accumulators.

## Dataflow Strategy

**Data path**: DRAM → reader (NoC reads) → L1 CBs → compute (matmul + eltwise + reduce + matmul) → L1 output CB → writer (NoC writes) → DRAM.

### RISC-V roles

| RISC | Role | Responsibilities |
|------|------|-------------------|
| NCRISC (Reader) | Data movement | Read Q, K, V, attn_mask tiles from DRAM into L1 CBs. Prepare scaler tiles. Stream K/V blocks into CBs as the compute inner loop consumes them. For GQA/MQA: replicate K/V head tiles to match Q head count (the reader reads the KV-head tile for the Q-head's group). |
| BRISC (Writer) | Data movement | Write output O tiles from L1 CB to DRAM. |
| TRISC0/1/2 (Compute) | FPU/SFPU | Flash attention recurrence: QK^T matmul, mask-add, row-max reduce, rescale, exp, row-sum reduce, PV matmul. All via helpers. |

### Within-core data flow (per (B,H) work unit)

```
Reader streams:
  Q-block (once, retained in cb_q) →
  For each KV-block:
    K-block → cb_k
    V-block → cb_v
    (mask-block → cb_mask, if attn_mask)

Compute per KV-block (all in L1, all via helpers):
  1. Q @ K^T → cb_scores (matmul_block, B_q × B_kv tiles)
  2. scores *= scale → cb_scores (eltwise mul, scalar broadcast)
  3. scores += mask → cb_scores_masked (eltwise add, if mask)
  4. row-max(scores_masked) → cb_max_new (reduce MAX REDUCE_ROW, WaitUpfrontNoPop)
  5. alpha = exp(m_old - m_new) → cb_alpha (eltwise_chain: sub + exp)
  6. O *= alpha → cb_o (eltwise mul, Col broadcast)
  7. l *= alpha → cb_sum_old (eltwise mul, Col broadcast)
  8. scores_masked -= m_new → cb_scores_masked (eltwise sub, Col broadcast)
  9. exp(scores_masked) → cb_exp_scores (eltwise unary Exp)
  10. row-sum(exp_scores) → cb_sum_new (reduce SUM REDUCE_ROW)
  11. l_i += sum_new → cb_sum_old (eltwise add)
  12. P @ V → accumulate into cb_o (matmul_block, packer_l1_acc=true)
  13. m_i = m_new → cb_max_old (eltwise copy)

Writer:
  After all KV-blocks for a Q-block: cb_o → DRAM output
```

### No inter-Tensix communication

Flash Attention is embarrassingly parallel across (B, H) pairs. Each core processes its assigned (B, H) units independently. No multicast, no semaphores, no ring topology needed. The only synchronization is intra-core CB synchronization between reader/compute/writer.

### GQA/MQA head replication

For GQA (H_q > H_kv > 1) and MQA (H_kv = 1), the reader maps each Q-head to its KV-head group: `kv_head_idx = q_head_idx // (H_q // H_kv)`. When reading K/V tiles for a given Q-head, the reader uses the kv_head_idx to select the correct KV-head tile from DRAM. This is a reader-side address mapping — no extra DRAM copies, no inter-core communication. The compute kernel is identical for MHA/GQA/MQA because it always sees one Q-head's data with matching K/V-head data in the CBs.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One (batch, head) pair — produces B_q_t × D_t output tiles (S_q/32 × D/32 tiles) |
| Grid | `device.compute_with_storage_grid_size()` — full Tensix grid |
| Per-core work | `num_q_blocks = S_q / (B_q * 32)` Q-blocks per (B,H) unit; `num_kv_blocks = S_kv / (B_kv * 32)` KV-blocks per Q-block. Total work units = B * H_q. Distribute across cores via `split_work_to_cores(grid_size, B*H_q, row_wise=True)`. |
| Remainder | `split_work_to_cores` handles remainder: group_1 gets ceil(B*H_q / num_cores), group_2 gets floor. Each core processes its assigned (B,H) units sequentially. |
| Per-core runtime args | `rt_args[core.x][core.y] = [q_buf_addr, k_buf_addr, v_buf_addr, mask_buf_addr, B_q_per_core, num_q_blocks, num_kv_blocks, ...]` set via `grid_to_cores` iteration over both core groups. |

### Tile block sizing

| Parameter | Value | Rationale |
|-----------|-------|----------|
| B_q_t (Q-block tile rows) | 4 (128 rows) | Fits DEST budget for QK^T: 4×4 subblock. Allows 4 Q-rows subblocks. |
| B_kv_t (KV-block tile cols) | 4 (128 cols) | Score block = 4×4 = 16 tiles. Within DEST limits for reduce. |
| D_t (head dim tiles) | D/32 | Typically 2–8 tiles (D=64→2, D=128→4, D=256→8). |

DEST budget: with `fp32_dest_acc_en=True`, DEST holds 8 tiles. The score block (B_q × B_kv = 4×4 = 16 tiles) exceeds DEST, so the matmul and reduce operate in subblocks. The `matmul_block` helper handles subblocking internally via `MatmulBlockShape::of(in0_sb, in1_sb, sb_h, sb_w, k, num_k)`. For QK^T with B_q=4, B_kv=4, K_dim=D_t: use subblock 2×2 or 4×1 to stay within 8-tile DEST.

**Subblock strategy**: QK^T matmul: M=4 (Q tiles), N=4 (K tiles), K=D_t. Subblock: `out_subblock_h=2, out_subblock_w=2` → 4 tiles per subblock (DEST-safe). PV matmul: M=4 (P tiles), N=D_t (V tiles), K=B_kv=4. Subblock: `out_subblock_h=2, out_subblock_w=min(D_t, 2)`.

## Circular Buffers

All CBs use tile-sized pages (bf16 = 2048 bytes). CB indices follow convention: 0–7 inputs, 8–15 special, 16–23 outputs, 24–31 intermediates.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| cb_q | 0 | tile_size(bf16) | B_q_t (4) | bf16 | Reader | Compute (QK^T matmul in0, retained across KV-blocks) | One Q-block; reused across all KV-blocks for this Q-block |
| cb_k | 1 | tile_size(bf16) | 2 * B_kv_t (8) | bf16 | Reader | Compute (QK^T matmul in1, consumed per KV-block) | One KV-block; double-buffered |
| cb_v | 2 | tile_size(bf16) | 2 * D_t | bf16 | Reader | Compute (PV matmul in1, consumed per KV-block) | One KV-block of V; double-buffered |
| cb_mask | 3 | tile_size(bf16) | 2 * B_kv_t (8) | bf16 | Reader | Compute (mask-add eltwise) | One KV-block of mask; double-buffered. Unused when no mask. |
| cb_scaler_reduce | 4 | tile_size(bf16) | 1 | bf16 | Reader | Compute (reduce scaler for MAX and SUM) | Entire kernel; scaler = 1.0 |
| cb_scale_factor | 5 | tile_size(bf16) | 1 | bf16 | Reader | Compute (scale broadcast for QK^T) | Entire kernel; holds scale value |
| cb_scores | 24 | tile_size(bf16) | B_q_t * B_kv_t (16) | bf16 | Compute (QK^T matmul) | Compute (scale, mask-add) | One KV-block's score matrix; full block for sequential helper intermediates |
| cb_scores_masked | 25 | tile_size(bf16) | B_q_t * B_kv_t (16) | bf16 | Compute (mask-add eltwise or scale-only passthrough) | Compute (row-max, sub, exp) | Intermediate; holds scores after mask/scale |
| cb_max_new | 26 | tile_size(bf16) | B_q_t (4) | bf16 | Compute (reduce MAX REDUCE_ROW) | Compute (alpha computation, update m_i) | One KV-block; per-row max |
| cb_max_old | 27 | tile_size(bf16) | B_q_t (4) | bf16 | Compute (running max) | Compute (rescale alpha computation) | Persistent across KV-blocks; holds m_i |
| cb_exp_scores | 28 | tile_size(bf16) | B_q_t * B_kv_t (16) | bf16 | Compute (exp eltwise) | Compute (row-sum, PV matmul in0) | One KV-block; full block for sequential intermediates |
| cb_sum_new | 29 | tile_size(bf16) | B_q_t (4) | bf16 | Compute (reduce SUM REDUCE_ROW) | Compute (update l_i) | One KV-block; per-row sum |
| cb_sum_old | 30 | tile_size(bf16) | B_q_t (4) | bf16 | Compute (running sum l_i) | Compute (rescale, update) | Persistent across KV-blocks; holds l_i |
| cb_o | 16 | tile_size(bf16) | B_q_t * D_t | bf16 | Compute (PV matmul out) | Writer | One Q-block's output; persistent across KV-blocks, written at Q-block end |
| cb_o_accum | 31 | tile_size(bf16) | B_q_t * D_t | bf16 | Compute (PV matmul interm) | Compute (rescale + accumulate into cb_o) | Scratch for PV accumulation with L1 acc |
| cb_alpha | 8 | tile_size(bf16) | B_q_t (4) | bf16 | Compute (eltwise_chain sub+exp) | Compute (rescale O and l) | One KV-block; alpha = exp(m_old - m_new) |

### CB sizing rationale

- **cb_q (4 pages)**: One Q-block = 4 tiles. Loaded once per Q-block, retained for all KV iterations. Reader writes, compute reads with `NoWaitNoPop` policy (Q is re-read each KV-block for QK^T matmul without popping).
- **cb_k, cb_v, cb_mask (8 pages each, double-buffered)**: Reader streams KV-blocks. Double-buffering allows reader to fetch the next block while compute processes the current. 2 × B_kv_t = 2 × 4 = 8 pages. cb_v has 2 × D_t pages (V is D-wide, not B_kv-wide).
- **cb_scores, cb_exp_scores (16 pages each)**: Full B_q × B_kv = 4 × 4 = 16 tile score block. These are sequential-helper intermediates (matmul → eltwise → reduce all on TRISC), so the CB must hold the entire block per the CB sync rule for sequential helpers.
- **cb_max_old, cb_sum_old (4 pages each)**: Persistent running state (m_i, l_i). B_q_t = 4 rows. Held across all KV-blocks for a Q-block.
- **cb_o (4 × D_t pages)**: Output accumulator. B_q_t × D_t tiles. For D=128 (D_t=4), that's 16 tiles = 32 KB. Persistent across KV-blocks, drained by writer at Q-block end.
- **cb_scaler_reduce (1 page)**: Scaler tile for reduce (1.0 for MAX, 1.0 for SUM). Prepared once by reader.
- **cb_scale_factor (1 page)**: Holds the scale value (1/sqrt(D) or explicit). Used to scale QK^T scores.
- **cb_alpha (4 pages)**: Holds alpha = exp(m_old - m_new) per Q-block row. Produced by eltwise_chain, consumed by rescale phases.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------------|----------|-----------|--------------|
| QK^T score matmul | helper | `matmul_block` | `matmul_block_helpers.hpp:333-365` | `transpose=true` (K^T), `packer_l1_acc=false`, `last_block_target=Out`, `tile_order=SubblockMajor`, `init_mode=Short`, `in0_policy=NoWaitNoPop` (Q retained), `in1_policy=WaitAndPopPerKBlock`, `reconfig=INPUT_AND_OUTPUT` | cb_q (in0), cb_k (in1) | cb_scores (out) | Boot with `mm_block_init(cb_q, cb_k, cb_scores, transpose=1, ct_dim, rt_dim, kt_dim)`. `MatmulBlockShape::of(in0_sb=1, in1_sb=1, sb_h=2, sb_w=2, k=D_t, num_k=1)`. |
| Scale scores | helper | `mul` (eltwise convenience) | `eltwise_convenience.hpp:82-99` | `BroadcastDim::Scalar`, `InputLifecycle::Streaming`, `InputLifecycle::HeldBulk` | cb_scores, cb_scale_factor | cb_scores (in-place) | `mul<cb_scores, cb_scale_factor, cb_scores, BroadcastDim::Scalar, ...>(shape)` |
| Mask add | helper | `add` (eltwise convenience) | `eltwise_convenience.hpp:44-61` | `BroadcastDim::None`, both `Streaming` | cb_scores, cb_mask | cb_scores_masked | `add<cb_scores, cb_mask, cb_scores_masked>(shape)`. Only when attn_mask is present. |
| Passthrough (no mask) | helper | `copy` (eltwise convenience) | `eltwise_convenience.hpp:172-182` | `Streaming` | cb_scores | cb_scores_masked | When no mask: `copy<cb_scores, cb_scores_masked>(shape)`. |
| Row-max of scores | helper | `reduce` | `reduce_helpers_compute.hpp:411-426` | `PoolType::MAX`, `ReduceDim::REDUCE_ROW`, `input_policy=WaitUpfrontNoPop`, `reconfig_mode=INPUT_AND_OUTPUT` | cb_scores_masked, cb_scaler_reduce | cb_max_new | `ReduceInputBlockShape::of(B_q_t, B_kv_t, 1)`. WaitUpfrontNoPop leaves scores in CB for phase 8. |
| Compute alpha = exp(m_old - m_new) | helper | `eltwise_chain` | `eltwise_chain.hpp:576-577` | `BinaryFpu<cb_max_old, cb_max_new, BinaryFpuOp::Sub, BroadcastDim::None>`, `Exp<>`, `PackTile<cb_alpha>` | cb_max_old, cb_max_new | cb_alpha | Fused chain: sub then exp. Shape: B_q_t tiles. |
| Rescale O_i (alpha * O) | helper | `mul` (eltwise convenience) | `eltwise_convenience.hpp:82-99` | `BroadcastDim::Col` (per-row alpha broadcast) | cb_o, cb_alpha | cb_o (in-place) | `mul<cb_o, cb_alpha, cb_o, BroadcastDim::Col, ...>(shape)` |
| Rescale l_i (alpha * l) | helper | `mul` (eltwise convenience) | `eltwise_convenience.hpp:82-99` | `BroadcastDim::Col` | cb_sum_old, cb_alpha | cb_sum_old (in-place) | Same alpha. |
| Subtract m_new from scores | helper | `sub` (eltwise convenience) | `eltwise_convenience.hpp:63-80` | `BroadcastDim::Col` (per-row max broadcast) | cb_scores_masked, cb_max_new | cb_scores_masked (in-place) | `sub<cb_scores_masked, cb_max_new, cb_scores_masked, BroadcastDim::Col, ..., Streaming, HeldStream>(shape)`. Reads scores with Streaming (wait+pop), reads m_new with HeldStream. |
| Exp of scores | helper | `unary` (eltwise convenience) | `eltwise_convenience.hpp:127-140` | `Exp<>` | cb_scores_masked | cb_exp_scores | `unary<Exp<>, cb_scores_masked, cb_exp_scores>(shape)` |
| Row-sum of exp scores | helper | `reduce` | `reduce_helpers_compute.hpp:411-426` | `PoolType::SUM`, `ReduceDim::REDUCE_ROW`, `input_policy=WaitAndPopPerTile` | cb_exp_scores, cb_scaler_reduce | cb_sum_new | `ReduceInputBlockShape::of(B_q_t, B_kv_t, 1)`. Produces B_q_t per-row sum tiles. |
| Update l_i = l_i + sum_new | helper | `add` (eltwise convenience) | `eltwise_convenience.hpp:44-61` | `BroadcastDim::None` | cb_sum_old, cb_sum_new | cb_sum_old (in-place) | Per-tile add. |
| PV matmul (accumulate O) | helper | `matmul_block` | `matmul_block_helpers.hpp:333-365` | `transpose=false`, `packer_l1_acc=true`, `last_block_target=Out`, `tile_order=SubblockMajor`, `init_mode=Short`, `in0_policy=WaitAndPopPerKBlock`, `in1_policy=WaitAndPopPerKBlock`, `reconfig=INPUT_AND_OUTPUT` | cb_exp_scores (in0, P), cb_v (in1, V) | cb_o (out, accumulated) | `MatmulBlockShape::of(in0_sb=1, in1_sb=1, sb_h=2, sb_w=min(D_t,2), k=B_kv_t, num_k=1)`. Boot with `mm_block_init(cb_exp_scores, cb_v, cb_o, transpose=0, ...)`. |
| Update m_i = m_new | helper | `copy` (eltwise convenience) | `eltwise_convenience.hpp:172-182` | `Streaming` | cb_max_new | cb_max_old (in-place) | `copy<cb_max_new, cb_max_old>(B_q_t)` |
| Scaler preparation (MAX) | helper | `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:94-101` | `<cb_scaler_reduce, PoolType::MAX, ReduceDim::REDUCE_ROW>` | — | cb_scaler_reduce | Called by reader. Scaler = 1.0 for MAX. |
| Scaler preparation (SUM) | helper | `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:94-101` | `<cb_scaler_reduce, PoolType::SUM, ReduceDim::REDUCE_ROW>` | — | cb_scaler_reduce | Called by reader. Scaler = 1.0 for SUM. Same CB — re-prepared before each reduce type. |
| Scale factor preparation | raw_api | reader fills cb_scale_factor | — | Reader writes scale value as a tile | — | cb_scale_factor | See justification below. |
| Init m_i, l_i, O_i | raw_api | reader/compute fills initial state | — | Compute initializes via FillScalar or reader writes constant tiles | — | cb_max_old, cb_sum_old, cb_o | See justification below. |

### Raw API justifications

**Scale factor tile (`cb_scale_factor`)**: The scale is a scalar (1/sqrt(D) or explicit). No helper exists for "fill a CB tile with a scalar value from a runtime arg." The `FillScalar` chain element (`eltwise_chain.hpp:544`) fills DST with a register-configured scalar, but the scale value comes from a runtime arg, not a compile-time register. The reader kernel writes the scale value into a 1-page CB tile using raw `cb_reserve_back` / write / `cb_push_back`. This is a data-movement operation (filling a CB with a constant), not a compute phase, so no compute helper applies.

**Init m_i / l_i / O_i**: Initializing running state to constants (-inf, 0, 0) is a one-time setup, not a compute phase. The reader fills cb_max_old with -inf and cb_sum_old / cb_o with 0 using raw CB writes (same pattern as the scale factor). Alternatively, the compute kernel can use `FillScalar` (`eltwise_chain.hpp:544`) to fill DST and pack into the CBs. Either is a constant-fill, not a reduction or eltwise operation.

## Compute Phases

The phases below describe one iteration of the inner KV-block loop. The outer Q-block loop and (B,H) work-unit loop wrap around these.

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|------------------------|-------------------|----------------|
| 0 | Init: load Q-block, zero O_i, init m_i=-inf, l_i=0 | raw_api (reader fills cb_q; compute/reader initializes m_i, l_i, O_i) | cb_q (4 tiles, loaded), cb_max_old (4 tiles, =-inf), cb_sum_old (4 tiles, =0), cb_o (4×D_t tiles, =0) | — | cb_q: full (retained). cb_max_old, cb_sum_old, cb_o: initialized. |
| 1 | QK^T score matmul: S = Q @ K^T (subblocked) | `matmul_block` | cb_q (4 tiles, NoWaitNoPop — retained), cb_k (4 tiles, WaitAndPopPerKBlock) | cb_scores (16 tiles) | cb_scores: full (16 tiles). cb_q: retained (not popped). cb_k: consumed (popped). |
| 2 | Scale: S *= scale | `mul` (eltwise) | cb_scores (16 tiles, Streaming), cb_scale_factor (1 tile, HeldBulk, Scalar broadcast) | cb_scores (in-place, 16 tiles) | cb_scores: scaled (16 tiles). cb_scale_factor: retained. |
| 3a | Mask add: S += mask (if attn_mask) | `add` (eltwise) | cb_scores (16 tiles, Streaming), cb_mask (4 tiles, Streaming) | cb_scores_masked (16 tiles) | cb_scores: consumed. cb_scores_masked: full (16 tiles). cb_mask: consumed. |
| 3b | Passthrough (no mask): copy S → cb_scores_masked | `copy` (eltwise) | cb_scores (16 tiles, Streaming) | cb_scores_masked (16 tiles) | cb_scores: consumed. cb_scores_masked: full (16 tiles). |
| 4 | Row-max: m_blk = max(S, dim=-1) | `reduce` MAX REDUCE_ROW, `WaitUpfrontNoPop` | cb_scores_masked (16 tiles, WaitUpfrontNoPop — NOT popped), cb_scaler_reduce (1 tile) | cb_max_new (4 tiles) | cb_scores_masked: still full (16 tiles, NOT popped). cb_max_new: full (4 tiles). cb_scaler_reduce: retained. |
| 5 | Compute alpha = exp(m_old - m_new) | `eltwise_chain` (sub + exp) | cb_max_old (4 tiles), cb_max_new (4 tiles) | cb_alpha (4 tiles) | cb_alpha: full (4 tiles). cb_max_old, cb_max_new: consumed. |
| 6 | Rescale O: O *= alpha (broadcast Col) | `mul` (eltwise) | cb_o (4×D_t tiles, Streaming), cb_alpha (4 tiles, HeldStream, Col broadcast) | cb_o (in-place) | cb_o: rescaled. cb_alpha: consumed. |
| 7 | Rescale l: l *= alpha (broadcast Col) | `mul` (eltwise) | cb_sum_old (4 tiles, Streaming), cb_alpha (4 tiles, held — consumed in phase 6) | cb_sum_old (in-place) | cb_sum_old: rescaled. |
| 8 | Subtract m_new: S -= m_new (broadcast Col) | `sub` (eltwise) | cb_scores_masked (16 tiles, Streaming — wait+pop), cb_max_new (4 tiles, HeldStream) | cb_scores_masked (in-place, 16 tiles) | cb_scores_masked: rescaled (16 tiles). cb_max_new: retained (for phase 13). |
| 9 | Exp: P = exp(S - m_new) | `unary` (eltwise Exp) | cb_scores_masked (16 tiles, Streaming) | cb_exp_scores (16 tiles) | cb_exp_scores: full (16 tiles). cb_scores_masked: consumed. |
| 10 | Row-sum: l_blk = sum(P, dim=-1) | `reduce` SUM REDUCE_ROW | cb_exp_scores (16 tiles, WaitAndPopPerTile), cb_scaler_reduce (1 tile) | cb_sum_new (4 tiles) | cb_exp_scores: consumed. cb_sum_new: full (4 tiles). cb_scaler_reduce: retained. |
| 11 | Update l: l_i += l_blk | `add` (eltwise) | cb_sum_old (4 tiles), cb_sum_new (4 tiles) | cb_sum_old (in-place) | cb_sum_old: updated. cb_sum_new: consumed. |
| 12 | PV matmul: O += P @ V (subblocked, L1 acc) | `matmul_block` (packer_l1_acc=true) | cb_exp_scores (16 tiles, WaitAndPopPerKBlock), cb_v (D_t tiles, WaitAndPopPerKBlock) | cb_o (accumulated, 4×D_t tiles) | cb_o: accumulated. cb_exp_scores: consumed. cb_v: consumed. |
| 13 | Update m: m_i = m_new | `copy` (eltwise) | cb_max_new (4 tiles, Streaming) | cb_max_old (in-place) | cb_max_old: updated. cb_max_new: consumed. |
| 14 | (After all KV-blocks) Write O to output | writer | cb_o (4×D_t tiles) | DRAM output | cb_o: drained. |

### Phase Splitting — the scores re-use problem

**Problem**: Phases 4 (row-max) and 8 (subtract m_new) both need the score block, but reduce with `WaitAndPopPerTile` consumes the input. The score block must survive the reduce so it can be used for the subtract + exp.

**Solution**: Use `ReduceInputPolicy::WaitUpfrontNoPop` for the row-max reduce (phase 4). This waits for all tiles but does NOT pop them, leaving `cb_scores_masked` intact for phase 8. The subsequent `sub` (phase 8) uses `Streaming` lifecycle which re-waits and pops per-tile. The CB sync is: reduce pushes nothing (just reads), sub reads with wait+pop. This works because `WaitUpfrontNoPop` does a bulk wait without pop, and the subsequent `Streaming` eltwise does per-tile wait+pop.

### Stage checkpoints

Each phase declares a checkpoint for TDD:

| Phase | Checkpoint (CB, tile, slice) | What to verify |
|-------|------------------------------|----------------|
| 0 | cb_max_old, tile 0, first 4×4 elements | All -inf |
| 1 | cb_scores, tile 0 (first Q-row × first K-col), first 4×4 elements | Q @ K^T values match reference (pre-scale) |
| 2 | cb_scores, tile 0, first 4×4 | Scaled by scale factor |
| 3 | cb_scores_masked, tile 0, first 4×4 | Mask added (0 where unmasked, -inf where masked) |
| 4 | cb_max_new, tile 0, first 4×4 | Row-max of first 4 score rows |
| 5 | cb_alpha, tile 0, first 4×4 | exp(m_old - m_new) values |
| 6 | cb_o, tile 0, first 4×4 | O rescaled by alpha |
| 8 | cb_scores_masked, tile 0, first 4×4 | Scores minus m_new |
| 9 | cb_exp_scores, tile 0, first 4×4 | exp of rescaled scores |
| 10 | cb_sum_new, tile 0, first 4×4 | Row-sum of exp scores |
| 11 | cb_sum_old, tile 0, first 4×4 | Updated l_i |
| 12 | cb_o, tile 0, first 4×4 | O after P @ V accumulation |
| 14 | output DRAM tile 0, first 4×4 | Final output matches reference |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| 2 | mul (scale) | cb_scores: 2D [B_q_t, B_kv_t] | cb_scale_factor: Scalar (1 tile) | Scalar |
| 3a | add (mask) | cb_scores: 2D [B_q_t, B_kv_t] | cb_mask: 2D [B_q_t, B_kv_t] | None |
| 5 | sub (m_old - m_new) | cb_max_old: Col0 [B_q_t, 1] | cb_max_new: Col0 [B_q_t, 1] | None (same shape) |
| 6 | mul (O *= alpha) | cb_o: 2D [B_q_t, D_t] | cb_alpha: Col0 [B_q_t, 1] | Col |
| 7 | mul (l *= alpha) | cb_sum_old: Col0 [B_q_t, 1] | cb_alpha: Col0 [B_q_t, 1] | None |
| 8 | sub (S -= m_new) | cb_scores_masked: 2D [B_q_t, B_kv_t] | cb_max_new: Col0 [B_q_t, 1] | Col |
| 11 | add (l += l_blk) | cb_sum_old: Col0 [B_q_t, 1] | cb_sum_new: Col0 [B_q_t, 1] | None |

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | EltwiseShape |
|-------------|----------------|--------------------|--------------|-----------------------|--------------|
| Row-max of scores (dim=-1, over KV cols) | REDUCE_ROW | Col0 [B_q_t, 1] | Col (for broadcast back) | of(B_q_t, B_kv_t, 1) | of(B_q_t, B_kv_t) |
| Row-sum of exp scores (dim=-1, over KV cols) | REDUCE_ROW | Col0 [B_q_t, 1] | Col (for broadcast back) | of(B_q_t, B_kv_t, 1) | of(B_q_t, B_kv_t) |

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB
  - Reader pushes B_kv_t tiles of K per KV-block; compute waits B_kv_t (via matmul_block in1_policy=WaitAndPopPerKBlock).
  - Reader pushes 1 scaler tile; compute waits 1 (reduce scaler).
  - Compute pushes 16 score tiles (matmul); eltwise waits 16 (Streaming).
  - Reduce with WaitUpfrontNoPop waits all 16 but pops 0; subsequent sub waits 16 and pops 16.
- [x] Reduce scaler CB is bfloat16 (scaler tiles prepared by reader via `calculate_and_prepare_reduce_scaler`)
- [x] Reduce scaler uses pool-type-aware API: `calculate_and_prepare_reduce_scaler<cb_scaler_reduce, PoolType::MAX, ReduceDim::REDUCE_ROW>()` for max, `calculate_and_prepare_reduce_scaler<cb_scaler_reduce, PoolType::SUM, ReduceDim::REDUCE_ROW>()` for sum
- [x] DEST: max 8 tiles (fp32 dest acc). Subblock sizing (2×2 = 4 tiles) is within limit.
- [x] Sequential helper intermediates sized to full block: cb_scores = 16 tiles, cb_exp_scores = 16 tiles
- [x] Page sizes aligned to tile size (2048 bytes for bf16)
- [x] All cb_wait_front calls on same CB use same page count
- [x] Helpers are not wrapped with extra CB operations
- [x] Every compute phase uses a helper; raw API only for reader-side constant fills
- [x] `compute_kernel_hw_startup()` called before any helper usage (at kernel boot)

## Key Risks and Gotchas

1. **Score block re-use after reduce**: The row-max reduce (phase 4) must use `WaitUpfrontNoPop` so the score block survives for the subtract+exp (phases 8-9). Forgetting this causes a deadlock: the reduce pops the tiles, then the sub waits for tiles that were already consumed.

2. **m_i / l_i persistence**: `cb_max_old` and `cb_sum_old` persist across all KV-blocks for a Q-block. They must NOT be popped between KV-block iterations. The compute kernel manually manages their lifecycle: read with `HeldBulk`/`HeldStream`, write in-place, pop only at Q-block end.

3. **O_i accumulation**: `cb_o` accumulates across KV-blocks via `packer_l1_acc=true` in the PV matmul. The first PV matmul (first KV-block) must initialize O to the rescaled value; subsequent blocks accumulate. The rescale (phase 6) must happen BEFORE the PV matmul so the accumulated O is correctly rescaled before adding the new block's contribution.

4. **Q retention**: `cb_q` is loaded once per Q-block and reused for every KV-block's QK^T matmul. Use `in0_policy=NoWaitNoPop` so the matmul does not pop Q. The reader must not overwrite cb_q until the Q-block is done.

5. **Matmul init state**: Two different matmuls run in the inner loop (QK^T with transpose=true, PV with transpose=false). Each `matmul_block` call with `InitMode::Short` issues `mm_block_init_short`, restoring the matmul state for its configuration. The kernel boots with `mm_block_init` for the first matmul; the second matmul's `InitMode::Short` reconfigures. Both matmuls have different transpose settings — `mm_block_init_short` handles the transpose reconfig.

6. **Scale application**: Scale is applied as a separate eltwise mul after the QK^T matmul (phase 2), not fused into the matmul. The matmul helper does not support a post-matmul scalar multiply. The scale is a scalar broadcast (`BroadcastDim::Scalar`) over all score tiles.

7. **Mask block streaming**: The attn_mask (B, 1, S_q, S_kv) or (B, H, S_q, S_kv) is streamed block-by-block alongside K/V. The reader fetches the mask block for the current (Q-block, KV-block) pair. When no mask is provided, phase 3b (copy passthrough) is used instead of phase 3a (mask add).

8. **L1 memory budget**: Total L1 for CBs per core (D=128, D_t=4):
   - cb_q: 4 × 2KB = 8KB; cb_k: 8 × 2KB = 16KB; cb_v: 2×4 × 2KB = 16KB; cb_mask: 8 × 2KB = 16KB
   - cb_scaler_reduce: 2KB; cb_scale_factor: 2KB
   - cb_scores: 16 × 2KB = 32KB; cb_scores_masked: 16 × 2KB = 32KB
   - cb_max_new: 4 × 2KB = 8KB; cb_max_old: 4 × 2KB = 8KB
   - cb_exp_scores: 16 × 2KB = 32KB; cb_sum_new: 4 × 2KB = 8KB; cb_sum_old: 4 × 2KB = 8KB
   - cb_o: 4×4 × 2KB = 32KB; cb_o_accum: 4×4 × 2KB = 32KB; cb_alpha: 4 × 2KB = 8KB
   - Total: ~264KB — well within 1.5MB L1 budget.

9. **Causal masking (refinement)**: When `is_causal=True` is added to SUPPORTED, the mask is generated on-device per (Q-block, KV-block) pair. Three regions: fully-past (no mask), fully-future (skip entire block — don't run QK^T/softmax/PV), diagonal-straddling (per-element triangular -inf mask). The block-skip for fully-future blocks is the causal perf win. `is_causal=True` + `attn_mask is not None` is a ValueError. `is_causal=True` + cross-attention (S_q != S_kv) is an EXCLUSION (NotImplementedError).

10. **Block size selection**: B_q_t=4 tiles (128 rows) and B_kv_t=4 tiles (128 cols) are the recommended defaults. These keep the score block at 16 tiles (32KB) and the subblocks within DEST. For very large head dims (D=256, D_t=8), the PV matmul subblock width must be ≤2 to stay within 8-tile DEST: `out_subblock_w=min(D_t, 2)`.

11. **GQA/MQA head replication**: The reader must map Q-head indices to KV-head indices via `kv_head_idx = q_head_idx // (H_q // H_kv)`. For MQA (H_kv=1), all Q-heads read from the same KV-head. This is purely a reader address mapping — no compute kernel changes needed.

12. **compute_kernel_config honoring**: The entry point must read `fp32_dest_acc_en` from the config (resolving `None` through `default_compute_kernel_config()`) and pass it to the compute kernel. Phase 0 supports only `fp32_dest_acc_en=True`; `False` is rejected via SUPPORTED. Never silently force `True` when the caller passes `False`.

## Structural impossibilities

No INVALID cells are declared for this op in `feature_spec.py`. SDPA is TILE-only (no ROW_MAJOR in TARGET), so the canonical `{dtype: bfloat8_b, layout: ROW_MAJOR}` rule is vacuous. `INVALID = []` is the expected baseline.

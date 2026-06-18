# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul → online-softmax → matmul) |
| Goal | Compute `softmax(Q·Kᵀ·scale + mask)·V` using the **Flash Attention** algorithm: tile over the KV sequence and accumulate the weighted output with an online (running) softmax so the full `S_q × S_kv` score matrix is never materialized. |
| Math | `O[b,h,i,:] = Σ_j softmax_j( (Q[b,h,i,:]·K[b,h,j,:]ᵀ)·scale + mask[b,h,i,j] ) · V[b,h,j,:]` |
| Mode | Hybrid (helper-library compute kernel built on `matmul_block`, `reduce`, `eltwise_chain`) |
| References | Tri Dao, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (online-softmax recurrence). Existing in-tree op: `ttnn/cpp/ttnn/operations/transformer/sdpa/` (algorithm reference, NOT helper-based). Helper headers: `ttnn/cpp/ttnn/kernel_lib/{matmul_block_helpers,reduce_helpers_compute,reduce_helpers_dataflow,eltwise_convenience,eltwise_chain,eltwise_math,eltwise_scalar,eltwise_binary_sfpu,dest_helpers}.hpp` |

### The load-bearing constraint

The `S_q × S_kv` attention-score matrix **must never be materialized** in DRAM or in any L1 circular buffer sized to hold it whole. Every score-bearing CB in this design is sized to a single `B_q × B_kv` block (`cb_qk_scores`, `cb_p`), never to `Sq_t × Sk_t`. The mask is applied **block-by-block** during the online softmax, not as a post-hoc full-matrix add. This is what makes the op Flash Attention and not plain SDPA.

### Online-softmax recurrence (per Q-block `i`, streamed over KV-blocks `j = 0 … n_kv-1`)

State carried across `j` (all per-row of the Q-block, fp32-accumulated semantics in the helper DEST):
- `m` — running row-max (one value per row → a column vector, `B_q` tiles, Col-0 valid)
- `l` — running row-sum of exponentials (`B_q` tiles, Col-0 valid)
- `O` — running weighted output (`B_q × DHt` tiles)

```
S_j   = (Q_i · scale) · K_jᵀ           # B_q × B_kv block, scale pre-folded into Q
S_j  += mask_ij                        # only if attention_mask present
m_j   = max(m_{j-1}, rowmax(S_j))      # running max
P_j   = exp(S_j - m_j)                 # B_q × B_kv, broadcast-subtract column max
α_j   = exp(m_{j-1} - m_j)             # correction factor (j>0; α_0 unused)
l_j   = α_j · l_{j-1} + rowsum(P_j)
O_j   = α_j · O_{j-1} + (P_j · V_j)
```
After the last KV-block: `O_i = O_{n_kv-1} / l_{n_kv-1}` (single reciprocal + broadcast multiply). This is mathematically exact softmax (the recurrence equals the two-pass formulation).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `Q` | `ttnn.Tensor` | yes | 4D `(B, H_q, S_q, D)`, bf16, TILE | — | tensor |
| `K` | `ttnn.Tensor` | yes | 4D `(B, H_kv, S_kv, D)`, bf16, TILE | — | tensor |
| `V` | `ttnn.Tensor` | yes | 4D `(B, H_kv, S_kv, D)`, bf16, TILE | — | tensor |
| `attention_mask` | `ttnn.Tensor` | no | `(B, 1, S_q, S_kv)` or `(B, H_q, S_q, S_kv)`, additive (0=attend, -inf=mask), bf16, TILE | `None` | tensor (optional) |
| `scale` | `float` | no | finite > 0 | `None` → `1/sqrt(D)` | host-resolved → CT `scale_bits` (fp32 bit-pattern) |

`scale` is resolved on host (`scale if scale is not None else 1.0/math.sqrt(D)`), reinterpreted to its 32-bit pattern, and passed to the compute kernel as a compile-time arg consumed by `MulUnary<>{scale_bits}`. `D` is compile-time per program, so the resolved scale is constant per launch.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q `(B, H_q, S_q, D)`; K,V `(B, H_kv, S_kv, D)`. `H_q % H_kv == 0` (MHA: `H_q==H_kv`; GQA: `1<H_kv<H_q`; MQA: `H_kv==1`). Cross-attn: `S_q != S_kv` allowed. |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE_LAYOUT |
| Memory | DRAM or L1 interleaved |
| Alignment | Phase 0: `S_q`, `S_kv`, `D` divisible by 32 (tile-aligned) |

### Output

| Property | Value |
|----------|-------|
| Shape | `(B, H_q, S_q, D)` (same as Q) |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved (or caller `memory_config`) |

## Dataflow Strategy

All tensors arrive **already tiled** (TILE_LAYOUT) → **no tilize / untilize** is needed anywhere. The data path is tiles end-to-end.

```
DRAM (Q,K,V[,mask] tiles, interleaved)
  │  reader (NCRISC)
  ├─► cb_q_in     : one Q-block (B_q × DHt tiles), loaded once per Q-block, retained across the KV loop
  ├─► cb_k_in     : streamed K-blocks (B_kv × DHt), one per KV step (double-buffered)
  ├─► cb_v_in     : streamed V-blocks (B_kv × DHt), one per KV step (double-buffered)
  ├─► cb_mask_in  : streamed mask sub-blocks (B_q × B_kv), one per KV step (only if mask given)
  ├─► cb_max_scaler : 1 bf16 tile, MAX+REDUCE_ROW fill (row-0), written once
  └─► cb_sum_scaler : 1 bf16 tile, SUM+REDUCE_ROW fill (col-0), written once

COMPUTE (TRISC unpack/math/pack) — per Q-block:
  pre-scale Q (× scale)  → online-softmax KV loop  → normalize  → cb_out
        (intermediates cb_qk_scores, cb_p, cb_o_blk and persistent
         accumulators cb_m_run/cb_m_prev/cb_l_run/cb_l_blk/cb_o_run/cb_alpha
         never exceed one B_q × B_kv (scores) or B_q × DHt (output) block)

  │  cb_out (B_q × DHt tiles, normalized)
  ▼  writer (BRISC)
DRAM (O tiles, interleaved)
```

**No inter-Tensix communication.** Each core owns whole `(batch, q-head, q-block)` work items and streams its own K/V/mask straight from DRAM. There are no multicasts, semaphores, or ring topologies — every core is independent.

**GQA/MQA** is a reader-side address remap only: for a query head `h_q`, the KV head is `h_kv = h_q / (H_q / H_kv)`. The compute kernel is identical regardless of `kv_heads_mode`; only the DRAM base offset the reader computes for K/V (and per-head mask) changes.

**Mask** is treated as a generic additive tensor read block-by-block. For a `(B,1,S_q,S_kv)` mask the reader uses head 0; for `(B,H_q,S_q,S_kv)` it uses `h_q`. Because no causal geometry is generated internally, causal masking works for both self- and cross-attention (a rectangular additive mask block is just added) — there is no `S_q==S_kv` assumption and hence no causal/cross exclusion.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one `(batch b, query-head h_q, q-block qb)` triple — produces one `B_q × DHt` output block by streaming all KV-blocks |
| Total work | `B · H_q · n_q` where `n_q = S_q / (B_q·32)` (q-blocks per head) |
| Grid | `ttnn.split_work_to_cores(device.compute_with_storage_grid_size(), total_work)` → `(num_cores, all_cores, core_group_1, core_group_2, items_g1, items_g2)` (`ttnn-python-utility-bindings.md:191-196`) |
| Per-core work | contiguous range of work items; core decodes each flat index → `(b, h_q, qb)`; inner loop over `n_kv = S_kv/(B_kv·32)` KV-blocks |
| Remainder | `split_work_to_cores` assigns `items_g1` to `core_group_1` and `items_g2` (= `items_g1-1`) to `core_group_2`; no manual remainder handling |

### Block sizing

| Symbol | Meaning | Value |
|--------|---------|-------|
| `DHt` | head dim in tiles | `D / 32` |
| `Sq_t`, `Sk_t` | seq lengths in tiles | `S_q/32`, `S_kv/32` |
| `B_q` (`Sq_chunk_t`) | Q-block height in tiles | host-chosen divisor of `Sq_t`, preferably the largest power-of-2 ≤ 4 that divides `Sq_t` (recommended 2 = 64 rows, up to 4 = 128) |
| `B_kv` (`Sk_chunk_t`) | KV-block height in tiles | host-chosen divisor of `Sk_t`, same rule (recommended 2 = 64) |
| `vDHt` | V head dim in tiles | `= DHt` (V head dim == D) |

Choosing `B_q`/`B_kv` as **divisors** of `Sq_t`/`Sk_t` guarantees no Q/K padding for tile-aligned Phase-0 inputs (divisor 1 always works for `S=32`). The host computes matmul sub-block dims with the standard `determine_largest_subblock_size(...)` rule so that `out_subblock_h · out_subblock_w ≤ DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`; 8 for bf16 half-sync).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q_in` | 0 | `tile_size(bf16)` | `B_q · DHt · 2` | bf16 | reader | compute (QK matmul `in0`, retained across KV loop) | double-buffered; loaded once per Q-block, popped after KV loop |
| `cb_k_in` | 1 | `tile_size(bf16)` | `B_kv · DHt · 2` | bf16 | reader | compute (QK matmul `in1`, transposed) | streaming, popped per KV-block |
| `cb_v_in` | 2 | `tile_size(bf16)` | `B_kv · vDHt · 2` | bf16 | reader | compute (PV matmul `in1`) | streaming, popped per KV-block |
| `cb_mask_in` | 3 | `tile_size(bf16)` | `B_q · B_kv · 2` (only if mask) | bf16 | reader | compute (mask add) | streaming, popped per KV-block |
| `cb_max_scaler` | 8 | `tile_size(bf16)` | 1 | bf16 | reader | compute (MAX reduce) | persistent constant |
| `cb_sum_scaler` | 9 | `tile_size(bf16)` | 1 | bf16 | reader | compute (SUM reduce) | persistent constant |
| `cb_alpha` | 10 | `tile_size(bf16)` | `B_q` | bf16 | compute | compute (rescale l, O) | per-KV-block scratch |
| `cb_l_recip` | 11 | `tile_size(bf16)` | `B_q` | bf16 | compute | compute (final normalize) | post-loop scratch |
| `cb_out` | 16 | `tile_size(bf16)` | `B_q · vDHt · 2` | bf16 | compute | writer | double-buffered output block |
| `cb_qk_scores` | 24 | `tile_size(bf16)` | `B_q · B_kv` | bf16 | compute (QK matmul) | compute (max reduce, sub-exp) | full block; freed each KV-block |
| `cb_p` | 25 | `tile_size(bf16)` | `B_q · B_kv` | bf16 | compute (sub-exp) | compute (sum reduce, PV matmul) | full block; freed each KV-block |
| `cb_o_blk` | 26 | `tile_size(bf16)` | `B_q · vDHt` | bf16 | compute (PV matmul) | compute (accumulate into `cb_o_run`) | full block; freed each KV-block |
| `cb_o_run` | 27 | `tile_size(bf16)` | `B_q · vDHt` | bf16 | compute | compute (persistent O accumulator) | persistent across KV loop |
| `cb_m_prev` | 28 | `tile_size(bf16)` | `B_q` | bf16 | compute | compute (MAX-reduce accumulator = `m_{j-1}`) | persistent across KV loop |
| `cb_m_run` | 29 | `tile_size(bf16)` | `B_q` | bf16 | compute (MAX reduce) | compute (sub-exp, α) | persistent across KV loop |
| `cb_l_run` | 30 | `tile_size(bf16)` | `B_q` | bf16 | compute | compute (persistent l accumulator) | persistent across KV loop |
| `cb_l_blk` | 31 | `tile_size(bf16)` | `B_q` | bf16 | compute (SUM reduce) | compute (accumulate into `cb_l_run`) | per-KV-block scratch |

**Score CBs are block-sized (`B_q·B_kv`), never `Sq_t·Sk_t`** — this is the Flash Attention constraint, enforced in the CB table. 17 CB slots used; all within the 0–31 budget.

CB sync (producer push == consumer wait) is satisfied per CB: streaming inputs (`cb_k_in`/`cb_v_in`/`cb_mask_in`) are pushed once and popped once per KV-block; `cb_q_in` is pushed once per Q-block and popped once after the KV loop; scalers pushed once, waited (no pop) every reduce; intermediates pushed and consumed within each KV step; `cb_out` pushed once per Q-block and consumed once by the writer.

## API Mapping

Every mechanism has a verified file:line reference. All compute phases use helpers; there are **no raw-API fallbacks**.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| Boot | helper | `compute_kernel_hw_startup()` + `mm_block_init()` | `reduce_helpers_compute.hpp:296-300`; `matmul_block_helpers.hpp:425-428` | — | — | — | once at top of `kernel_main`; the only hw_configure-bearing init |
| Pre-scale Q | helper | `transform_in_place<cb_q_in>(B_q·DHt, MulUnary<>{scale_bits})` | `eltwise_convenience.hpp:227-242`; `eltwise_scalar.hpp:32` (`MulUnary`) | `Cb=cb_q_in` | `cb_q_in` | `cb_q_in` | in-place; once per Q-block before KV loop; owns its own wait/pop/reserve/push |
| 1. `S = Q·Kᵀ` | helper | `matmul_block<transpose=true, …, tile_order=TileRowMajor, in0_policy=WaitAndRetainOnLastBlock, in1_policy=WaitAndPopPerKBlock>` | `matmul_block_helpers.hpp:790-823`; transpose-B `:448`; `MatmulBlockShape::of` `:232-241`; `OutputCBLayout` `:50`; `InputPolicy` `:97`; SDPA `@example` `:672-687` | `cb_q_in` (in0), `cb_k_in` (in1) | `cb_qk_scores` | `transpose=true` gives Kᵀ; `M=B_q, N=B_kv, Kc=DHt`; retains Q across KV loop; manages its own out reserve/push and in1 pop |
| 2. mask add | helper | `add<cb_qk_scores, cb_mask_in, cb_qk_scores>(EltwiseShape::grid(B_q,B_kv))` | `eltwise_convenience.hpp:56-61`; `EltwiseShape::grid` `:105-122` | `cb_qk_scores`, `cb_mask_in` | `cb_qk_scores` (in-place) | `BroadcastDim::None`; only emitted when mask present |
| 3. running max | helper | `reduce<MAX, REDUCE_ROW, cb_qk_scores, cb_max_scaler, cb_m_run, WaitUpfrontNoPop, …, Accumulate>(ReduceInputBlockShape::of(B_q,B_kv), …, Accumulate(cb_m_prev, j))` | `reduce_helpers_compute.hpp:411-426`; `WaitUpfrontNoPop` `:372-378`; `Accumulate` `:224-238`; `ReduceInputBlockShape` `:138-147` | `cb_qk_scores`, `cb_max_scaler` | `cb_m_run` | `cb_m_run = max(cb_m_prev, rowmax(S_j))`; `j=0` skips reload; **does not pop** `cb_qk_scores` (reused by phase 5). Accumulator CB (`cb_m_prev`) ≠ output CB (`cb_m_run`) so `m_{j-1}` survives for phase 4 |
| 4. correction α | helper | `sub<cb_m_prev, cb_m_run, cb_alpha>(B_q)` → `unary<Exp<>, cb_alpha, cb_alpha>(B_q)` (or one fused `eltwise_chain(B_q, BinaryFpu<Sub>, Exp<>, PackTile<cb_alpha>)`) | `eltwise_convenience.hpp:75-80` (`sub`), `:136-140` (`unary`); `eltwise_chain.hpp:577`, `BinaryFpu` `:500-513`, `PackTile` `:535-541`; `Exp<>` `eltwise_math.hpp:20-21` | `cb_m_prev`, `cb_m_run` | `cb_alpha` | `α_j = exp(m_{j-1} − m_j)`; both operands Col-0 column vectors, `BroadcastDim::None`; emitted only for `j>0` |
| 5. `P = exp(S − m)` | helper | `eltwise_chain(EltwiseShape::grid(B_q,B_kv), BinaryFpu<cb_qk_scores,cb_m_run,Sub,BroadcastDim::Col, …, AIdx=Block,BIdx=Col>, Exp<>, PackTile<cb_p>)` (equiv. `sub<…,BroadcastDim::Col>` then `unary<Exp<>>`) | `eltwise_chain.hpp:577`; `BinaryFpu` `:500-513`; `Exp<>` `eltwise_math.hpp:20-21`; `BroadcastDim` `eltwise_convenience.hpp:425-430`; `OperandKind` `eltwise_chain.hpp:253-258` | `cb_qk_scores`, `cb_m_run` | `cb_p` | broadcast-subtract per-row max (Col-bcast from Col-0), then exp; **pops** `cb_qk_scores` (the only reader to pop it) |
| 6. block sum | helper | `reduce<SUM, REDUCE_ROW, cb_p, cb_sum_scaler, cb_l_blk, WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q,B_kv))` | `reduce_helpers_compute.hpp:411-426`; `WaitUpfrontNoPop` `:372-378` | `cb_p`, `cb_sum_scaler` | `cb_l_blk` | `rowsum(P_j)`; **does not pop** `cb_p` (reused by phase 8) |
| 7a. rescale l | helper | `mul<cb_l_run, cb_alpha, cb_l_run>(B_q)` | `eltwise_convenience.hpp:94-99` | `cb_l_run`, `cb_alpha` | `cb_l_run` (in-place) | `l ← α·l`; `BroadcastDim::None`; `j>0` only |
| 7b. accumulate l | helper | `add<cb_l_run, cb_l_blk, cb_l_run>(B_q)` (`j>0`); `copy<cb_l_blk, cb_l_run>(B_q)` (`j==0`) | `eltwise_convenience.hpp:56-61`, `:180-182` (`copy`) | `cb_l_run`, `cb_l_blk` | `cb_l_run` | `l ← l + rowsum(P_j)` |
| 8. `P·V` | helper | `matmul_block<transpose=false, …, tile_order=TileRowMajor, in0_policy=WaitAndPopPerKBlock, in1_policy=WaitAndPopPerKBlock>` | `matmul_block_helpers.hpp:790-823`; `MatmulBlockShape::of` `:232-241` | `cb_p` (in0), `cb_v_in` (in1) | `cb_o_blk` | `M=B_q, N=vDHt, Kc=B_kv`; pops `cb_p` and `cb_v_in`; manages out reserve/push |
| 9a. rescale O | helper | `mul<cb_o_run, cb_alpha, cb_o_run, BroadcastDim::Col, …, AIdx=Block,BIdx=Col>(EltwiseShape::grid(B_q,vDHt))` | `eltwise_convenience.hpp:94-99`; `BroadcastDim` `:425-430` | `cb_o_run`, `cb_alpha` | `cb_o_run` (in-place) | `O ← α·O`; α (Col-0) broadcast across vDHt columns; `j>0` only |
| 9b. accumulate O | helper | `add<cb_o_run, cb_o_blk, cb_o_run>(EltwiseShape::grid(B_q,vDHt))` (`j>0`); `copy<cb_o_blk, cb_o_run>` (`j==0`) | `eltwise_convenience.hpp:56-61`, `:180-182` | `cb_o_run`, `cb_o_blk` | `cb_o_run` | `O ← O + P_j·V_j`; `BroadcastDim::None` |
| 9c. commit max | helper | `copy<cb_m_run, cb_m_prev>(B_q)` | `eltwise_convenience.hpp:180-182` | `cb_m_run` | `cb_m_prev` | makes `m_j` the accumulator (`m_{j-1}`) for the next KV-block |
| 10a. reciprocal | helper | `unary<Recip<>, cb_l_run, cb_l_recip>(B_q)` | `eltwise_convenience.hpp:136-140`; `Recip<>` `eltwise_math.hpp:32-33` | `cb_l_run` | `cb_l_recip` | `1/l_final` (per row, Col-0) |
| 10b. normalize | helper | `mul<cb_o_run, cb_l_recip, cb_out, BroadcastDim::Col, …, AIdx=Block,BIdx=Col>(EltwiseShape::grid(B_q,vDHt))` | `eltwise_convenience.hpp:94-99` | `cb_o_run`, `cb_l_recip` | `cb_out` | `O_i = O/l_final`; recip (Col-0) broadcast across vDHt |
| Scaler setup (reader) | helper | `prepare_reduce_scaler<cb_max_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>(1.0f)` and `prepare_reduce_scaler<cb_sum_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | `reduce_helpers_dataflow.hpp:65-67`; fill-pattern note `:45-47` | — | `cb_max_scaler`, `cb_sum_scaler` | **pool-type-aware overloads** (NOT legacy `prepare_reduce_scaler<cb>`): MAX+REDUCE_ROW → row-0 fill, SUM+REDUCE_ROW → col-0 fill. Two distinct scaler CBs because the fills differ |
| Reader Q/K/V/mask | helper | `TensorAccessor` reads | `tech_reports/tensor_accessor/tensor_accessor.md` | DRAM | `cb_q_in`,`cb_k_in`,`cb_v_in`,`cb_mask_in` | interleaved tile reads; per-(b,h_q,qb,j) page-id math, GQA head remap `h_kv=h_q/(H_q/H_kv)` |
| Writer O | helper | `TensorAccessor` writes | `tech_reports/tensor_accessor/tensor_accessor.md` | `cb_out` | DRAM | one `B_q × vDHt` block per work item |

> **No helper is bypassed.** Every compute phase maps to `matmul_block`, `reduce`, or an `eltwise_convenience`/`eltwise_chain` element. There are no hand-rolled FPU/SFPU sequences, so the "Helpers considered and rejected" sub-entry is not required for any phase.

## Compute Phases

Sequential execution for one work item `(b, h_q, qb)`. Phases 1–9 repeat per KV-block `j`; pre-scale runs once before the loop; phases 10a/10b run once after.

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 0 | pre-scale `Q ← Q·scale` | `transform_in_place`+`MulUnary` | `cb_q_in` (`B_q·DHt`, fresh) | `cb_q_in` (scaled) | `cb_q_in` holds scaled Q, retained for whole KV loop |
| 1 | `S_j = Q·Kᵀ` | `matmul_block` (transpose) | `cb_q_in` (retained), `cb_k_in` (`B_kv·DHt`) | `cb_qk_scores` (`B_q·B_kv`) | `cb_q_in` retained; `cb_k_in` popped; scores ready |
| 2 | `S_j += mask` | `add` | `cb_qk_scores`, `cb_mask_in` (`B_q·B_kv`) | `cb_qk_scores` | mask popped (if present); scores updated in place |
| 3 | `m_j = max(m_{j-1}, rowmax S_j)` | `reduce<MAX>`+`Accumulate` | `cb_qk_scores` (no-pop), `cb_max_scaler`, accumulator `cb_m_prev` | `cb_m_run` (`B_q`) | scores still present; `cb_m_run`=m_j, `cb_m_prev`=m_{j-1} |
| 4 | `α_j = exp(m_{j-1}-m_j)` (j>0) | `eltwise_chain`(Sub,Exp) | `cb_m_prev`, `cb_m_run` | `cb_alpha` (`B_q`) | α ready |
| 5 | `P_j = exp(S_j - m_j)` | `eltwise_chain`(Sub Col-bcast, Exp) | `cb_qk_scores` (pop), `cb_m_run` | `cb_p` (`B_q·B_kv`) | `cb_qk_scores` freed; `cb_p` ready |
| 6 | `s_j = rowsum(P_j)` | `reduce<SUM>` | `cb_p` (no-pop), `cb_sum_scaler` | `cb_l_blk` (`B_q`) | `cb_p` still present |
| 7 | `l_j = α_j·l_{j-1}+s_j` | `mul`(j>0) + `add`/`copy` | `cb_l_run`, `cb_alpha`, `cb_l_blk` | `cb_l_run` (`B_q`) | running sum updated |
| 8 | `O_blk = P_j·V_j` | `matmul_block` | `cb_p` (pop), `cb_v_in` (`B_kv·vDHt`) | `cb_o_blk` (`B_q·vDHt`) | `cb_p`,`cb_v_in` freed |
| 9 | `O_j = α_j·O_{j-1}+O_blk`; commit `m_prev←m_run` | `mul` Col-bcast (j>0) + `add`/`copy` + `copy` | `cb_o_run`,`cb_alpha`,`cb_o_blk`; `cb_m_run` | `cb_o_run`; `cb_m_prev` | accumulators advanced for `j+1` |
| 10a | `r = 1/l_final` | `unary<Recip>` | `cb_l_run` | `cb_l_recip` (`B_q`) | reciprocal ready |
| 10b | `O_i = O·r` | `mul` Col-bcast | `cb_o_run`, `cb_l_recip` | `cb_out` (`B_q·vDHt`) | output block pushed to writer; pop `cb_q_in` |

**First-block specialization (`j==0`):** phases 4, 7a, 9a (the α-correction) are skipped; phases 7b and 9b become `copy` (l ← block sum, O ← O_blk); phase 3 skips the accumulator reload (`Accumulate(cb_m_prev, 0)` → pure `rowmax(S_0)`).

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 2 mask add | `add` | `cb_qk_scores` All `[B_q,B_kv]` | `cb_mask_in` All `[B_q,B_kv]` | None |
| 4 α | `sub` | `cb_m_prev` Col0 `[B_q,1]` | `cb_m_run` Col0 `[B_q,1]` | None |
| 5 P sub | `sub` | `cb_qk_scores` All `[B_q,B_kv]` | `cb_m_run` Col0 `[B_q,1]` (per-row max) | **Col** (broadcast column-vector across columns) |
| 7a rescale l | `mul` | `cb_l_run` Col0 `[B_q,1]` | `cb_alpha` Col0 `[B_q,1]` | None |
| 7b accumulate l | `add` | `cb_l_run` Col0 `[B_q,1]` | `cb_l_blk` Col0 `[B_q,1]` | None |
| 9a rescale O | `mul` | `cb_o_run` All `[B_q,vDHt]` | `cb_alpha` Col0 `[B_q,1]` | **Col** |
| 9b accumulate O | `add` | `cb_o_run` All `[B_q,vDHt]` | `cb_o_blk` All `[B_q,vDHt]` | None |
| 10b normalize | `mul` | `cb_o_run` All `[B_q,vDHt]` | `cb_l_recip` Col0 `[B_q,1]` | **Col** |

`REDUCE_ROW` outputs land in **Col0** (per-row scalar in column 0 of each tile), so every per-row vector (`m`, `l`, `α`, `recip`) is a Col0 column-vector and is broadcast with `BroadcastDim::Col` (operand kind `Col`) across the free columns. This matches the template valid-region rule "REDUCE_ROW out → Col0".

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim (downstream) | ReduceInputBlockShape |
|-------------|----------------|---------------------|---------------------------|-----------------------|
| row-max over `S_kv` (free/W axis of score block) | `REDUCE_ROW` | Col0, `B_q` tiles | `Col` (phase 5) | `of(B_q, B_kv)` |
| row-sum over `S_kv` of `P` | `REDUCE_ROW` | Col0, `B_q` tiles | `Col` (phases 7/9/10) | `of(B_q, B_kv)` |

Softmax reduces along the KV (free) axis of each score block — always `REDUCE_ROW`. The op never reduces over the Q axis, so only one reduce direction is used.

## Validation Contract (for the op file — implementer deliverable)

Recommended `INPUT_TAGGERS` (op-local, matching the shared golden suite):

- `tag_alignment(inputs, axes)` — examine Q's `(S_q, D)` = `inputs[0][-2:]`: both %32==0 → `"tile_aligned"`; `D%32!=0` → `"w_non_aligned"`; else (`S_q%32!=0`, D aligned) → `"h_non_aligned"`.
- `tag_attention_kind(inputs, axes)` — `inputs[0][-2]` (`S_q`) vs `inputs[1][-2]` (`S_kv`): equal → `"self"`, else `"cross"`.
- `tag_kv_heads(inputs, axes)` — `inputs[0][1]` (`H_q`) vs `inputs[1][1]` (`H_kv`): equal → `"mha"`, `H_kv==1` → `"mqa"`, else `"gqa"`.

Recommended Phase-0 `SUPPORTED`:

```python
SUPPORTED = {
    "dtype":          [ttnn.bfloat16],
    "layout":         [ttnn.TILE_LAYOUT],
    "alignment":      ["tile_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode":  ["mha", "gqa", "mqa"],   # head remap is reader-side only — all three work in Phase 0
    "mask_mode":      ["none", "causal"],
    "scale_mode":     ["auto", "explicit"],
}
EXCLUSIONS = []   # generic additive-mask path handles causal+cross; no S_q==S_kv assumption
```

> **Note on `kv_heads_mode`:** the baseline-SDPA Phase-0 list in the prompt omitted this axis, but the shared golden suite's `tag_kv_heads` tagger emits it and the cartesian iterates it. Because GQA/MQA require nothing beyond the reader's `h_kv = h_q/(H_q/H_kv)` address remap (identical compute kernel), Phase 0 supports all three values; declaring them keeps the harness gate consistent rather than silently running ungated cells.

`validate()` (entry point's first line) raises the typed `UnsupportedAxisValue`/`ExcludedCell` for out-of-`SUPPORTED` / `EXCLUSIONS` cells. **Beyond axis validation**, raise `ValueError`/`RuntimeError` for tensor-shape contract violations *before* the axis gate:
- any of Q/K/V not 4D;
- `Q.shape[-1] != K.shape[-1]` (head_dim mismatch Q vs K);
- `K.shape != V.shape` (K/V seq_len or head_dim mismatch);
- `Q.shape[0] != K.shape[0]` (batch mismatch) or `H_q % H_kv != 0` (incompatible head grouping);
- mask present and `mask.shape[-2:] != (S_q, S_kv)`, or `mask.shape[1] ∉ {1, H_q}`, or `mask.shape[0] != B`.

There is no `dim`/`axis` index parameter, so no sign-canonicalization is required.

### Structural impossibilities (INVALID)

Pipeline mode: `eval/golden_tests/scaled_dot_product_attention/feature_spec.py` already declares `INVALID = []` (SDPA is TILE-only, so the canonical bf8b+ROW_MAJOR rule is vacuous). No op-specific structural impossibilities to add. The op file must **not** declare `INVALID`.

## Key Risks and Gotchas

- **Flash constraint (load-bearing):** `cb_qk_scores` and `cb_p` MUST be sized `B_q·B_kv`, never `Sq_t·Sk_t`. The mask is added per KV-block from `cb_mask_in` (`B_q·B_kv`), never assembled whole. Sizing either score CB to the full matrix breaks the spec.
- **Persistent accumulators across the KV loop:** `cb_o_run` (`B_q·vDHt`), `cb_m_run`/`cb_m_prev`/`cb_l_run` (`B_q`) hold running state and are not popped between KV-blocks. They are full-block CBs (not streaming double-buffers); in-place `mul`/`add` reuse the slot under per-tile streaming (pop-before-reserve), so `B_q`-/`B_q·vDHt`-page sizing is sufficient.
- **`m_{j-1}` must survive to compute α:** the running-max `reduce<MAX>` uses a **distinct** accumulator CB (`cb_m_prev`) and output CB (`cb_m_run`). Phase 9c copies `cb_m_run → cb_m_prev` to advance state. Do not collapse these into one CB or α (phase 4) loses its `m_{j-1}` operand.
- **Two scaler CBs, pool-type-aware fills:** `cb_max_scaler` (MAX+REDUCE_ROW → row-0 fill) and `cb_sum_scaler` (SUM+REDUCE_ROW → col-0 fill) are different tiles. Use `prepare_reduce_scaler<cb, PoolType, ReduceDim>(1.0f)` (the pool-type-aware overload), never the legacy `prepare_reduce_scaler<cb>`. Both are bf16, value 1.0 (plain max / plain sum, not average).
- **Reduce does not pop the score block:** phases 3 (max) and 6 (sum) use `WaitUpfrontNoPop` so `cb_qk_scores`/`cb_p` remain for the subsequent sub-exp (phase 5) / PV matmul (phase 8). Exactly one consumer pops each (phase 5 pops `cb_qk_scores`, phase 8 pops `cb_p`).
- **MAX+REDUCE_ROW accumulation arch note:** `reduce_helpers_compute.hpp:210-213` rejects this combo on *Quasar* (needs a within-16×16-face transpose). Valid on Wormhole/Blackhole (the test targets). If ported to Quasar, replace phase 3's accumulation with `reduce<MAX>` (no accumulate) → `binary_sfpu<BinaryMax<>>` (`eltwise_binary_sfpu.hpp:72-77`).
- **Matmul `OutputCBLayout::TileRowMajor`** for both matmuls — downstream `reduce`/`eltwise` read the block in tile-row order. `SubblockMajor` (the default) would scramble the row/col mapping the reduce assumes.
- **Q transpose orientation:** QK uses `transpose=true` on `in1=K` (`matmul_block_helpers.hpp:448`) so stored `K[B_kv,DHt]` is consumed as `Kᵀ[DHt,B_kv]`; contraction tiles `in0_block_k = DHt`. PV uses `transpose=false`, contraction `in0_block_k = B_kv`.
- **DEST limit:** `out_subblock_h·out_subblock_w ≤ DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`; 8 for bf16). Host derives sub-block dims; large `B_q·B_kv`/`B_q·vDHt` blocks are sub-blocked by the helper.
- **Scale folded into Q:** pre-scaling Q once per Q-block (phase 0) makes scores pre-scaled, so no per-block score scaling is needed (Q is reused across all KV-blocks; `B_q·DHt ≪ Σ B_kv·DHt`). `scale_bits` is the fp32 bit-pattern of the host-resolved scale.
- **Fully-masked blocks:** with the additive mask, a fully `-inf` block contributes `P=0`, `rowsum=0`, `O_blk=0`, `α=1` — harmless as long as `j=0` for each row is not fully masked. For standard causal (column 0 attended by every row) and the no-mask case this never occurs; the golden suite uses exactly these.
- **GQA/MQA correctness:** the only difference is the reader's KV head index `h_kv = h_q/(H_q/H_kv)` and per-head mask offset; the compute kernel is unchanged.
- **Boot init:** `compute_kernel_hw_startup()` then one `mm_block_init()` at the top of the compute kernel — the only hw_configure-bearing call. `matmul_block` (`InitMode::Short` default) and the reduce/eltwise helpers issue their own per-call short-inits/reconfigs.

# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + online softmax + matmul) |
| Goal | Compute `softmax(Q @ K^T * scale + mask) @ V` with the Flash Attention algorithm: tile over S_kv, maintain running max / running sum / running output per Q block. The full S_q × S_kv score matrix is NEVER materialized — only one B_q × B_kv score block lives in L1 at a time. |
| Math | `O = softmax(Q @ K^T * scale + mask, dim=-1) @ V`, online recurrence: `m_new = max(m_prev, rowmax(S_blk))`, `α = exp(m_prev − m_new)`, `P = exp(S_blk − m_new)`, `l_new = α·l_prev + rowsum(P)`, `O_new = α·O_prev + P @ V`, final `O = O / l` |
| Mode | Hybrid (helper-composed compute) |
| References | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `reduce_helpers_compute.hpp`, `streaming_reduce_helpers.hpp`, `eltwise_chain.hpp`, `eltwise_math.hpp`, `eltwise_scalar.hpp`, `eltwise_fill.hpp`, `reduce_helpers_dataflow.hpp`, `tech_reports/tensor_accessor/tensor_accessor.md` |

The load-bearing constraint: per-block CBs are sized `c_q × c_kv` tiles (≤ 16 tiles), independent of S_kv. Long-context shapes (S = 8192 → 256 KV blocks) stream through the same fixed CB layout.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `attention_mask` | ttnn.Tensor / None | No | (B,1,S_q,S_kv) or (B,H,S_q,S_kv), additive, bf16, TILE | None | CT flag `HAS_MASK` + mask accessor RT args |
| `scale` | float / None | No | any finite float | None → 1/√D | RT (float bits as uint32, fed to `MulUnary`) |

Shape-derived constants (host computes, passes as CT args): `Dt = D/32`, `Sq_t = S_q/32`, `Skv_t = S_kv/32`, chunk size `c = clamp(16 / Dt, 1, 4)` tiles (=> B_q = B_kv = 32·c rows, 128 for D ≤ 128, 64 for D = 256, 32 for D ≥ 512 — keeps total L1 footprint < ~900 KB for all Phase-0 shapes incl. D = 1024), `c_q = min(c, Sq_t)`, `c_kv = min(c, Skv_t)`. `mask_is_per_head` (mask H dim == H) is a CT flag for the reader. GQA/MQA ready: reader maps Q head `h` → KV head `h / (H_q / H_kv)`; compute is head-agnostic.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q (B,H,S_q,D), K (B,H_kv,S_kv,D), V (B,H_kv,S_kv,D), H % H_kv == 0; Phase 0: H_kv == H |
| Dtype | bfloat16 (all, incl. mask) |
| Layout | TILE |
| Memory | interleaved, DRAM or L1 |
| Alignment | S_q, S_kv, D divisible by 32 (Phase 0) |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H, S_q, D) — same as Q |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved DRAM |

## Dataflow Strategy

No inter-Tensix communication. Each core independently owns a set of (b, h, q_chunk) work units.

```
DRAM(Q,K,V,mask) → reader(NCRISC, tiles) → CBs → compute(unpack/math/pack) → cb_out_tiles → writer(BRISC) → DRAM(O)
```

Per work unit (one Q chunk of c_q tile-rows for one (b,h)):

1. Reader pushes the Q chunk once: `c_q·Dt` tiles, retained in L1 across the whole KV loop (`InputPolicy::WaitAndRetainOnLastBlock`, popped by compute after the final KV block).
2. Reader streams `Nkv = ceil(Skv_t / c_kv)` KV blocks; per block kb it pushes K-transposed tiles (`Dt × c_kv`, tile order `(d, n) → K[b, h_kv, n, d]` — tile-order transpose; intra-tile transpose handled by matmul `transpose=true`), V tiles (`c_kv × Dt`, row-major), and mask tiles (`c_q × c_kv`, when HAS_MASK).
3. Compute runs the online-softmax recurrence per block (Compute Phases below); statistics (m, l, α) are per-Q-row column tiles (`c_q` tiles each, Col0 valid). O accumulates in fp32 in `cb_o_acc`. Everything per-block, nothing sized by S_kv.
4. After the last KV block, compute normalizes `O / l` and pushes `c_q·Dt` bf16 tiles; writer streams them to DRAM.

Numerics: fp32 throughout the accumulation path — `FP32_DEST_ACC_EN = true` (DEST limit 4 tiles), MathFidelity HiFi2 (bf16 inputs; HiFi4+fp32-DEST with bf16 is known-bad on WH, matmul_block_helpers.hpp:415-419). Score/stat/output-accumulator CBs are Float32; only matmul inputs (Q, K^T, P, V) and final output are bf16. m is initialized to −1e9 (not −inf, avoids inf−inf NaN); l, O initialized to 0. First block: `Accumulate` skips reload at iteration 0, so m = rowmax(block 0) exactly; α₀ = exp(−1e9 − m) = 0 ⇒ O₀ = P@V exactly. Mask is applied block-by-block (`scale·S + mask` before the running-max update) — never as a full-matrix post-hoc add.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one (b, h, q_chunk): c_q tile-rows of Q against all of K/V |
| Grid | full compute grid via `ttnn.split_work_to_cores(grid, B·H·Nq)` where `Nq = ceil(Sq_t / c_q)` |
| Per-core work | contiguous range of flattened (b·H + h)·Nq + q_chunk indices; reader/compute/writer all loop over the same count |
| Remainder | core groups from split_work_to_cores; tail q-chunk uses `c_q_last = Sq_t − (Nq−1)·c_q`, tail kv-block uses `c_kv_last = Skv_t − (Nkv−1)·c_kv` — both ≥ 1 tile, passed via CT args, kernels select per-iteration chunk size |

CBs are sized for full `c_q`/`c_kv`; tail chunks just push/wait fewer tiles (all wait/pop counts derived from the same per-iteration size in all three kernels).

## Circular Buffers

`T_bf = tile_size(bf16) = 2080 B (untilized payload 2048)`, `T_f32 = 4096 B`. Sizes are per page = 1 tile.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q_tiles` | 0 | T_bf | c_q·Dt | bf16 | reader | compute (QK^T) | per Q chunk, retained across KV loop, popped after last block |
| `cb_kt_tiles` | 1 | T_bf | 2·c_kv·Dt | bf16 | reader | compute (QK^T) | per KV block, double-buffered |
| `cb_v_tiles` | 2 | T_bf | 2·c_kv·Dt | bf16 | reader | compute (P@V) | per KV block, double-buffered |
| `cb_mask_tiles` | 3 | T_bf | 2·c_q·c_kv | bf16 | reader | compute (scale+mask) | per KV block, double-buffered; only allocated when HAS_MASK |
| `cb_scaler_max` | 8 | T_bf | 1 | bf16 | reader (once) | reduce MAX | whole program |
| `cb_scaler_sum` | 9 | T_bf | 1 | bf16 | reader (once) | reduce SUM | whole program |
| `cb_cur_sum` | 10 | T_f32 | c_q | f32 | compute (rowsum P) | compute (l update) | per KV block |
| `cb_prev_max` | 11 | T_f32 | c_q | f32 | compute (init / copy) | compute (α) | per KV block |
| `cb_running_max` | 12 | T_f32 | 2·c_q | f32 | compute (accum. reduce) | compute (α held, sub-exp held, copy) | per Q chunk |
| `cb_alpha` | 13 | T_f32 | c_q | f32 | compute | compute (l update held, O rescale pop) | per KV block |
| `cb_running_sum` | 14 | T_f32 | 2·c_q | f32 | compute (l update) | compute (recip at end) | per Q chunk |
| `cb_inv_sum` | 15 | T_f32 | c_q | f32 | compute | compute (final mul) | per Q chunk |
| `cb_out_tiles` | 16 | T_bf | 2·c_q·Dt | bf16 | compute | writer | per Q chunk, double-buffered |
| `cb_scores` | 24 | T_f32 | c_q·c_kv | f32 | compute (QK^T) | compute (scale+mask) | per KV block, full block |
| `cb_scores_scaled` | 25 | T_f32 | c_q·c_kv | f32 | compute | compute (max reduce held + sub-exp pop) | per KV block, full block |
| `cb_probs` | 26 | T_bf | c_q·c_kv | bf16 | compute (sub-exp) | compute (rowsum held + P@V pop) | per KV block, full block |
| `cb_pv` | 27 | T_f32 | c_q·Dt | f32 | compute (P@V) | compute (O update) | per KV block, full block |
| `cb_o_acc` | 28 | T_f32 | 2·c_q·Dt | f32 | compute | compute (read-front + write-back, AtEnd pop) | per Q chunk; 2 blocks for same-CB read-write |

L1 worst cases: D ≤ 128 (Dt ≤ 4, c = 4): ≈ 640 KB. D = 1024 (Dt = 32, c = 1): ≈ 880 KB. Fits 1.5 MB.

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------------|----------|-----------|--------------|
| boot | raw_api | `compute_kernel_hw_startup` + `mm_init` | matmul_block_helpers.hpp:356-361,425-429 | (cb_q_tiles, cb_kt_tiles, cb_scores) | — | — | once, first lines of MAIN |
| scalers | helper | `calculate_and_prepare_reduce_scaler<cb, MAX, REDUCE_ROW>()` / `<cb, SUM, REDUCE_ROW>()` | reduce_helpers_dataflow.hpp:94-101 | pool-type-aware; row0 fill for MAX, col0 (matmul path) for SUM | — | cb_scaler_max, cb_scaler_sum | reader, once |
| init m,l,O | helper | `eltwise_chain(c_q, FillScalar<D0>{-1e9f}, PackTile<cb_prev_max>)` (+0.0f→cb_running_sum c_q tiles, 0.0f→cb_o_acc c_q·Dt tiles) | eltwise_fill.inl:14-21, eltwise_chain.hpp:576 | `PackTile<cb, OutputLifecycle::Streaming, PackTileReconfig::Output>` | — | cb_prev_max, cb_running_sum, cb_o_acc | per Q chunk |
| 1 QK^T | helper | `matmul_block<true, false, Out, TileRowMajor, Short, WaitAndRetainOnLastBlock, WaitAndPopPerKBlock>` | matmul_block_helpers.hpp:790-823; transpose:791; retain:82-87; TileRowMajor:37-45 | `MatmulBlockShape::of(c_q, 1, 1, c_kv, Dt, 1)`; interm = cb_q_tiles (unused, num_k_blocks=1, :605-612) | cb_q_tiles, cb_kt_tiles | cb_scores | Q never popped (every call is last block); subblock 1×c_kv ≤ 4 fp32-DEST tiles |
| 2 scale+mask | helper | `eltwise_chain(grid(c_q,c_kv), CopyTile<cb_scores,D0,Streaming>, MulUnary<D0>{scale_bits}, [DestReuseBinary<cb_mask_tiles,Add,DEST_TO_SRCB,Streaming>], PackTile<cb_scores_scaled>)` | eltwise_chain.hpp:491-541, eltwise_scalar.inl:46-58 | mask element compiled in iff HAS_MASK | cb_scores (+cb_mask_tiles) | cb_scores_scaled | block-wise mask — load-bearing |
| 3 save m_prev | helper | `eltwise_chain(c_q, CopyTile<cb_running_max,D0,HeldStream>, PackTile<cb_prev_max>)` | eltwise_chain.hpp:181-182,491-541 | kb > 0 only; no pop (Accumulate reload pops) | cb_running_max | cb_prev_max | |
| 4 m update | helper | `accumulate_reduce_block<MAX, REDUCE_ROW, WaitUpfrontNoPop>(cb_scores_scaled, cb_scaler_max, cb_running_max, of(c_q,c_kv), kb, Nkv)` | streaming_reduce_helpers.hpp:47-61; reduce policy:104-107 | iter 0 skips reload ⇒ m₀ = rowmax(blk0) | cb_scores_scaled (held), cb_scaler_max | cb_running_max | m_new col0-valid |
| 5 α | helper | `eltwise_chain(c_q, CopyTile<cb_prev_max,D0,Streaming>, DestReuseBinary<cb_running_max,Sub,DEST_TO_SRCB,HeldStream>, Exp<>, PackTile<cb_alpha>)` | eltwise_chain.hpp:515-525, eltwise_math.inl:28-31 | running_max held for phase 6 | cb_prev_max, cb_running_max | cb_alpha | α = exp(m_prev−m_new) |
| 6 P = exp(S−m) | helper | `eltwise_chain(grid(c_q,c_kv), BinaryFpu<cb_scores_scaled, cb_running_max, Sub, BroadcastDim::Col, DeferredPop, HeldBulk, …, Block, Col>, Exp<>, PackTile<cb_probs, Bulk>)` | eltwise_chain.hpp:500-513; Col bcast:425-430; lifecycles:183-186 | pops scores_scaled at end; running_max stays for next kb | cb_scores_scaled, cb_running_max | cb_probs (bf16) | exact softmax numerator |
| 7 rowsum P | helper | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(cb_probs, cb_scaler_sum, cb_cur_sum, of(c_q, c_kv))` | reduce_helpers_compute.hpp:400-415; policy:104-107 | probs retained for P@V | cb_probs, cb_scaler_sum | cb_cur_sum | SUM/ROW → matmul path, col0 scaler |
| 8 l update | helper | `eltwise_chain(c_q, CopyTile<cb_running_sum,D0,Streaming>, DestReuseBinary<cb_alpha,Mul,DEST_TO_SRCB,HeldStream>, DestReuseBinary<cb_cur_sum,Add,DEST_TO_SRCB,Streaming>, PackTile<cb_running_sum>)` | eltwise_chain.hpp:515-525 | α held for phase 10 | cb_running_sum, cb_alpha, cb_cur_sum | cb_running_sum | 2-block CB absorbs pop/push |
| 9 P@V | helper | `matmul_block<false, false, Out, TileRowMajor, Short, WaitAndPopPerKBlock, WaitAndPopPerKBlock>` | matmul_block_helpers.hpp:790-823 | `MatmulBlockShape::of(c_q, Dt/sw, 1, sw, c_kv, 1)`, sw = max divisor of Dt ≤ 4; interm = cb_pv (unused); pops probs + V | cb_probs, cb_v_tiles | cb_pv | |
| 10 O update | helper | `eltwise_chain(grid(c_q,Dt), BinaryFpu<cb_o_acc, cb_alpha, Mul, BroadcastDim::Col, Bulk, Bulk, …, Block, Col>, DestReuseBinary<cb_pv,Add,DEST_TO_SRCB,Streaming>, PackTile<cb_o_acc, Bulk>)` | eltwise_chain.hpp:500-525 | reads front block, packs back block, pops AtEnd; pops α | cb_o_acc, cb_alpha, cb_pv | cb_o_acc | 2-block cb_o_acc required |
| 11 1/l | helper | `eltwise_chain(c_q, CopyTile<cb_running_sum,D0,Streaming>, Recip<>, PackTile<cb_inv_sum>)` | eltwise_math.hpp:32-33 | after last kb | cb_running_sum | cb_inv_sum | |
| 12 O/l | helper | `eltwise_chain(grid(c_q,Dt), BinaryFpu<cb_o_acc, cb_inv_sum, Mul, BroadcastDim::Col, Bulk, Bulk, …, Block, Col>, PackTile<cb_out_tiles, Streaming, Output>)` | eltwise_chain.hpp:500-541 | pops o_acc + inv_sum; packs bf16 | cb_o_acc, cb_inv_sum | cb_out_tiles | |
| Q pop | raw_api | `cb_pop_front(cb_q_tiles, c_q·Dt)` | api/compute/cb_api.h | after last kb | cb_q_tiles | — | pairs with WaitAndRetainOnLastBlock |
| read/write | raw_api | `TensorAccessor` + `noc_async_read/write_tile` | tech_reports/tensor_accessor.md | TensorAccessorArgs last in CT args | — | — | reader streams K in transposed tile order |

Helpers considered and rejected — none. Every compute phase maps to a helper. The only raw compute API is the boot init + the single `cb_pop_front` releasing retained Q (the documented caller-side counterpart of `WaitAndRetainOnLastBlock`, matmul_block_helpers.hpp:82-87). `reduce_helpers_dataflow.hpp:65-67` (`prepare_reduce_scaler` legacy overload) rejected per scaler rule — pool-type-aware `calculate_and_prepare_reduce_scaler` used.

## Compute Phases

KV loop = phases 1–10 per kb, statistics persist; 11–12 finalize. Push = wait counts per CB hold per kb (scores: push c_q·c_kv, held wait then pop c_q·c_kv; running_max: 1 push per kb, popped by next kb's reload, final pop after phase 6 of last kb — implementer pops 1 block at chunk end; same for running_sum after phase 11).

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| 0 | init m=−1e9, l=0, O=0 | chain Fill | — | cb_prev_max c_q; cb_running_sum c_q; cb_o_acc c_q·Dt | |
| 1 | S = Q·K^T (transpose-B) | matmul_block | cb_q_tiles retained; cb_kt_tiles popped | cb_scores c_q·c_kv | Q stays |
| 2 | S′ = scale·S (+ mask) | chain | cb_scores popped; cb_mask popped | cb_scores_scaled c_q·c_kv | |
| 3 | m_prev ← m (kb>0) | chain | cb_running_max held | cb_prev_max c_q | |
| 4 | m = max(m, rowmax S′) | accumulate_reduce_block | cb_scores_scaled held; old m popped by reload | cb_running_max c_q | scores still fronted |
| 5 | α = exp(m_prev − m) | chain | cb_prev_max popped; m held | cb_alpha c_q | |
| 6 | P = exp(S′ − m) bcast-Col | chain | cb_scores_scaled popped; m held | cb_probs c_q·c_kv | m survives to next kb |
| 7 | r = rowsum P | reduce | cb_probs held | cb_cur_sum c_q | |
| 8 | l = α·l + r | chain | cb_running_sum popped; α held; cur_sum popped | cb_running_sum c_q | |
| 9 | PV = P·V | matmul_block | cb_probs popped; cb_v popped | cb_pv c_q·Dt | |
| 10 | O = α⊙O + PV | chain | cb_o_acc popped AtEnd; α popped; pv popped | cb_o_acc c_q·Dt | next kb or finalize |
| 11 | inv = 1/l | chain | cb_running_sum popped | cb_inv_sum c_q | |
| 12 | out = O⊙inv | chain | cb_o_acc, cb_inv_sum popped | cb_out_tiles c_q·Dt | all CBs empty; pop Q |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 6 | Sub | cb_scores_scaled [H,W] All | cb_running_max REDUCE_ROW out → Col0 | Col |
| 10 | Mul | cb_o_acc [H,W] All | cb_alpha Col0 (exp of Col0; junk cols never read by Col bcast) | Col |
| 12 | Mul | cb_o_acc All | cb_inv_sum Col0 | Col |

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| cb_o_acc same-CB read/write (phase 10) | Capacity 2 blocks; A lifecycle Bulk (pop AtEnd), pack Bulk — never use 1 block |
| Persistent stats popped exactly once per kb | running_max/running_sum 2·c_q pages; reload pop must match wait counts (always c_q) |
| K^T order | Reader does tile-order transpose; matmul `transpose=true` does intra-tile only — both required |
| Scaler fills differ | MAX/ROW row0, SUM/ROW col0 — two scaler CBs, pool-type-aware helper only |
| fp32 DEST | All subblocks ≤ 4 tiles; HiFi2, never HiFi4 with bf16 |
| Mask | Applied per-block before max — never post-hoc; rectangular causal mask is fully supported as plain additive tensor; no block-skip optimization in Phase 0 |
| −1e9 sentinel | exp arguments ≤ 0; m never NaN; α₀ = 0 makes O-init exact |
| Tail chunks | wait/pop counts derive from per-iter chunk size, identical in all three kernels |
| Structural impossibilities | None beyond feature_spec's `INVALID = []` (TILE-only universe — bf8b+ROW_MAJOR vacuous) |

Hardware checklist: CB sync verified per phase table ✓; scalers pool-type-aware bf16 ✓; DEST ≤ 4 (fp32) ✓; full-block CBs for sequential helpers (scores, probs, pv) ✓; same wait count per CB ✓; helpers not wrapped with extra CB ops ✓; hw_startup once at boot ✓.

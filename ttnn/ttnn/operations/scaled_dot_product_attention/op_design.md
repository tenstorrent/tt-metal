# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + softmax + matmul, tiled online-softmax) |
| Goal | Compute `softmax(Q·Kᵀ·scale + mask)·V` using the **Flash Attention** algorithm — tiled over the sequence dimension with an online-softmax recurrence so the full `S_q × S_kv` score matrix is **never** materialized in DRAM or in any L1 CB. |
| Math | `O[b,h,:,:] = softmax( (Q[b,h]·K[b,h]ᵀ)·scale + mask[b,·] , dim=-1 ) · V[b,h]` |
| Mode | Hybrid (tiled matmul + tiled reduce + tiled eltwise online recurrence) |
| References | Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*; production reference `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` (algorithm shape only — this design uses the `kernel_lib` helpers, not that op's private `compute_common.hpp` helpers) |

**Load-bearing constraint.** The score block held in L1 is sized **`q_chunk_t × kv_chunk_t` tiles per (Q-chunk, KV-chunk)** — i.e. `cb_scores` and `cb_p` hold one block at a time, never `Sq_t × Skv_t`. The running statistics (max `m`, sum `l`) and the running output `O` are kept per Q-chunk and updated incrementally as each KV block streams through. This is what makes it Flash Attention rather than plain SDPA.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `Q` | `ttnn.Tensor` | yes | `(B, H, S_q, D)`, bf16, TILE | — | tensor |
| `K` | `ttnn.Tensor` | yes | `(B, H_kv, S_kv, D)`, bf16, TILE | — | tensor |
| `V` | `ttnn.Tensor` | yes | `(B, H_kv, S_kv, D)`, bf16, TILE | — | tensor |
| `attention_mask` | `ttnn.Tensor` | no (kw-only) | `(B, 1, S_q, S_kv)` or `(B, H, S_q, S_kv)`, additive (0 / −inf), bf16, TILE | `None` | tensor |
| `scale` | `float` | no (kw-only) | finite > 0 | `None` → `1/sqrt(D)` | RT (bit-cast u32 to compute) |
| `q_chunk_t` | host const | derived | `1` (Phase 0) | `1` | CT |
| `kv_chunk_t` | host const | derived | largest divisor of `Skv_t` with `kv_chunk_t ≤ KV_CHUNK_MAX` | see below | CT |
| `use_mask` | host const | derived | `attention_mask is not None` | — | CT |

`Sq_t = S_q/32`, `Skv_t = S_kv/32`, `Dt = D/32` (all integers — Phase 0 is tile-aligned). `KV_CHUNK_MAX` is bounded so the QK output sub-block `q_chunk_t·kv_chunk_t` and the PV output sub-block `q_chunk_t·Dt` each fit `DEST_AUTO_LIMIT`; recommend `KV_CHUNK_MAX = 4`.

### Phase-0 design choice: `q_chunk_t = 1`

The Q chunk is **one tile-row (32 query rows)**. This makes every per-row broadcast operation use the *single-pinned-tile* broadcast pattern (`max` / `alpha` / `recip` are exactly **1 tile**, broadcast `BroadcastDim::Col` across the block with the B operand pinned at tile 0) — the exact `eltwise_chain` asymmetric-bcast pattern documented at `eltwise_chain.hpp:156-169`. No per-row loop, no runtime `Absolute` CB index. The Flash-Attention memory bound is fully satisfied (`cb_scores`/`cb_p` are `kv_chunk_t` tiles). Larger `q_chunk_t` (B_q = 64/128) is a performance refinement that adds a per-Q-tile-row loop with indexed broadcast.

## Tensors

### Input

| Property | Q | K / V | attention_mask (optional) |
|----------|---|-------|---------------------------|
| Shape | `(B, H, S_q, D)` | `(B, H_kv, S_kv, D)` | `(B, 1, S_q, S_kv)` or `(B, H, S_q, S_kv)` |
| Dtype | bfloat16 | bfloat16 | bfloat16 |
| Layout | TILE | TILE | TILE |
| Memory | DRAM or L1 interleaved | DRAM or L1 interleaved | DRAM or L1 interleaved |

Phase-0 `H_kv == H` (MHA). `S_kv` may differ from `S_q` (cross-attention). All of `S_q, S_kv, D` divisible by 32.

### Output

| Property | Value |
|----------|-------|
| Shape | `(B, H, S_q, D)` (same as Q) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved (or caller `memory_config`) |

## Registry contract (op file: `scaled_dot_product_attention.py`)

The op module must export `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` (imported by `eval/golden_tests/scaled_dot_product_attention/test_golden.py`) and a `validate()` called as the first line of the public entry point. `feature_spec.py` already exists (pipeline mode) and declares `TARGET`, `INPUTS`, `INVALID = []`.

### INPUT_TAGGERS (must match `feature_spec.py` docstring — three taggers)

| Tagger | Signature | Logic |
|--------|-----------|-------|
| `tag_alignment` | `(inputs, axes) -> "tile_aligned" \| "w_non_aligned" \| "h_non_aligned"` | `q = inputs[0]`; `S_q,D = q[-2],q[-1]`. Both %32==0 → `tile_aligned`; `D%32!=0` → `w_non_aligned`; else (`D` aligned, `S_q` not) → `h_non_aligned`. |
| `tag_attention_kind` | `(inputs, axes) -> "self" \| "cross"` | `"self"` if `inputs[0][-2] == inputs[1][-2]` (S_q == S_kv) else `"cross"`. |
| `tag_kv_heads` | `(inputs, axes) -> "mha" \| "gqa" \| "mqa"` | `H_q = inputs[0][1]`, `H_kv = inputs[1][1]`. `H_q == H_kv` → `"mha"`; `H_kv == 1` → `"mqa"`; else `"gqa"` (assumes `H_q % H_kv == 0`). |

### SUPPORTED (Phase 0 baseline)

| Axis | Values |
|------|--------|
| `dtype` | `[ttnn.bfloat16]` |
| `layout` | `[ttnn.TILE_LAYOUT]` |
| `alignment` | `["tile_aligned"]` |
| `attention_kind` | `["self", "cross"]` |
| `kv_heads_mode` | `["mha"]` |
| `mask_mode` | `["none", "causal"]` |
| `scale_mode` | `["auto", "explicit"]` |

`kv_heads_mode` **must** appear in `SUPPORTED` even though the task's Phase-0 enumeration omitted it: the tagger emits it and `feature_spec.TARGET` includes it, so `validate()` must gate on it. With `["mha"]`, GQA/MQA cells become refinement candidates (xfail with `UnsupportedAxisValue`) rather than silently producing wrong output. `mask_mode` is derived in `validate()`: `"none"` when `attention_mask is None`, else `"causal"`. `scale_mode`: `"auto"` when `scale is None`, else `"explicit"`.

### EXCLUSIONS

```python
EXCLUSIONS = []
```

The causal-mask path applies the **dense provided additive mask block-by-block** (the test builds the mask via `make_causal_mask`, including the rectangular `S_q × S_kv` form for cross-attention, and passes it as `attention_mask`). Because the kernel never assumes `S_q == S_kv` for masking — it simply adds whatever mask block the reader fetched — `{mask_mode: causal, attention_kind: cross}` is handled correctly and is **not** excluded.

### validate() order

1. Structural shape contract → raise `ValueError`/`RuntimeError`:
   - Q, K, V (and mask if present) rank == 4.
   - `Q.shape[-1] == K.shape[-1]` (head_dim match) and `V.shape[-1] == K.shape[-1]`.
   - `K.shape[-2] == V.shape[-2]` (S_kv match) and `K.shape[1] == V.shape[1]` (H_kv match).
   - `Q.shape[0] == K.shape[0]` (batch) and `Q.shape[1] % K.shape[1] == 0` (head divisibility).
   - mask (if present): `mask.shape[-2] == S_q`, `mask.shape[-1] == S_kv`, `mask.shape[0] == B`, `mask.shape[1] in (1, H)`.
2. Per-axis SUPPORTED check → `UnsupportedAxisValue` (NotImplementedError-derived). Build axes via `dtype/layout` from Q + the three taggers on `(Q.shape, K.shape, V.shape)` + derived `mask_mode`/`scale_mode`.
3. EXCLUSIONS check → `ExcludedCell` (currently empty).

## Dataflow Strategy

```
DRAM (tiled, interleaved)                         per Tensix core
─────────────────────────                         ───────────────
Q[b,h,q,:]   ── reader (NCRISC) ─► cb_q (bf16, Dt)         ┐
K[b,h,kvblk] ── reader ─────────► cb_k (bf16)              │ compute (UNPACK/MATH/PACK)
V[b,h,kvblk] ── reader ─────────► cb_v (bf16)              │  online-softmax recurrence
mask[b,0,..] ── reader ─────────► cb_mask (bf16, if mask)  │  (all fp32 stats/accumulators)
scaler 1.0   ── reader (dataflow scaler prep) ─► cb_scaler_max / cb_scaler_sum
                                                           │
                              cb_out (bf16, Dt) ──────────►┘ ── writer (BRISC) ─► DRAM O[b,h,q,:]
```

- **Format at each stage.** Inputs arrive **tiled bf16** (no tilize needed — TILE_LAYOUT). All compute intermediates that accumulate (`cb_scores`, `cb_pv`, `cb_out_accum`, running `m`/`l`/`alpha`/`recip`) are **fp32** for an exact online softmax; the exp-probabilities `cb_p` are bf16 (≤ 1.0, fed straight into the bf16 PV matmul). Output is packed **bf16, tiled** for the writer.
- **Within a Tensix.** Reader (NCRISC) streams the resident Q tile-row once, then streams each KV block (K, V, mask) in order; compute consumes them through the recurrence; writer (BRISC) drains the single normalized output tile-row. No tilize/untilize — everything is tile-format end to end.
- **Inter-Tensix.** None. Each work unit `(b, h, q_tile_row)` is fully independent; no multicast, semaphores, or ring topology. Cores share only DRAM.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one `(batch b, head h, query tile-row q)` triple → produces output tile-row `O[b,h,q,:]` (`Dt` tiles) |
| Total units | `B · H · Sq_t` (since `q_chunk_t = 1`, `num_q_chunks = Sq_t`) |
| Grid | full compute grid `device.compute_with_storage_grid_size()` |
| Split | `ttnn.split_work_to_cores(core_grid, B·H·Sq_t)` → `core_group_1`/`core_group_2`, per-core `units_g1`/`units_g2` |
| Per-core work | a contiguous range `[start_unit, start_unit+num_units)`; each core loops its units, and for each loops all `num_kv_chunks = Skv_t / kv_chunk_t` KV blocks |
| Index decode | `b = unit / (H·Sq_t)`; `h = (unit / Sq_t) % H`; `q = unit % Sq_t`. KV head `= h` (MHA). |
| Remainder | handled by `split_work_to_cores` (group_1 cores get one extra unit). Uniform KV blocks: `kv_chunk_t` is chosen to **divide** `Skv_t`, so no partial KV block. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q` | 0 | tile(bf16) | `Dt` | bf16 | reader | compute (QKᵀ in0, retained) | per work unit; popped after KV loop |
| `cb_k` | 1 | tile(bf16) | `2·kv_chunk_t·Dt` | bf16 | reader | compute (QKᵀ in1) | streaming, double-buffered; popped per KV block |
| `cb_v` | 2 | tile(bf16) | `2·kv_chunk_t·Dt` | bf16 | reader | compute (PV in1) | streaming, double-buffered; popped per KV block |
| `cb_mask` | 3 | tile(bf16) | `2·kv_chunk_t` | bf16 | reader | compute (mask add) | streaming, double-buffered; only if `use_mask`; popped per KV block |
| `cb_scaler_max` | 8 | tile(bf16) | 1 | bf16 | reader (scaler prep) | compute (MAX reduce) | filled once per core |
| `cb_scaler_sum` | 9 | tile(bf16) | 1 | bf16 | reader (scaler prep) | compute (SUM reduce) | filled once per core |
| `cb_mblock` | 10 | tile(fp32) | 1 | fp32 | compute (reduce MAX, j>0) | compute (max recurrence) | per KV block (j>0) |
| `cb_mnew` | 11 | tile(fp32) | 1 | fp32 | compute (max recurrence) | compute (alpha, copy-back) | per KV block (j>0) |
| `cb_lblock` | 12 | tile(fp32) | 1 | fp32 | compute (reduce SUM, j>0) | compute (sum recurrence) | per KV block (j>0) |
| `cb_out` | 16 | tile(bf16) | `2·Dt` | bf16 | compute (final normalize) | writer | streaming, double-buffered |
| `cb_scores` | 24 | tile(fp32) | `kv_chunk_t` | fp32 | compute (QKᵀ matmul) | compute (mask add, MAX reduce, exp) | per KV block; held across max+exp |
| `cb_p` | 25 | tile(bf16) | `kv_chunk_t` | bf16 | compute (exp) | compute (SUM reduce, PV matmul) | per KV block; held across sum+PV |
| `cb_pv` | 26 | tile(fp32) | `Dt` | fp32 | compute (PV matmul, j>0) | compute (O recurrence) | per KV block (j>0) |
| `cb_out_accum` | 27 | tile(fp32) | `Dt` | fp32 | compute (PV j=0 / O recurrence) | compute (final normalize) | **persistent across whole KV loop** |
| `cb_max` | 28 | tile(fp32) | 1 | fp32 | compute (reduce MAX j=0 / copy-back) | compute (exp, max recurrence) | **persistent across whole KV loop** (running `m`) |
| `cb_sum` | 29 | tile(fp32) | 1 | fp32 | compute (reduce SUM j=0 / sum recurrence) | compute (sum recurrence, recip) | **persistent across whole KV loop** (running `l`) |
| `cb_alpha` | 30 | tile(fp32) | 1 | fp32 | compute (alpha, j>0) | compute (sum + O recurrence) | per KV block (j>0) |
| `cb_recip` | 31 | tile(fp32) | 1 | fp32 | compute (recip after loop) | compute (final normalize) | once per work unit (after KV loop) |

**Sizing rationale.** Streaming reader/writer CBs (`cb_k`, `cb_v`, `cb_mask`, `cb_out`) are double-buffered so the reader/writer overlap compute. Sequential-helper intermediates between back-to-back full-thread helpers (`cb_scores`, `cb_p`, `cb_pv`) are sized to **one full block** because producer and consumer helpers each own all three TRISCs and cannot pipeline. Running state (`cb_q`, `cb_max`, `cb_sum`, `cb_out_accum`) is single-buffered and persists across the KV loop. Per-block scratch stats (`cb_mblock`, `cb_mnew`, `cb_lblock`, `cb_alpha`) and `cb_recip` are 1 tile. Scaler CBs are 1 tile. Total 18 CBs ≤ 32; L1 footprint ≈ a few hundred KB (well under 1.5 MB).

**Producer/consumer balance (`Skv_t/kv_chunk_t = num_kv_chunks` iterations).** `cb_q`: pushed 1×Dt by reader, scaled in-place (pop Dt + push Dt), read by every QKᵀ matmul without pop (retained), popped 1×Dt at end → balanced. `cb_max`: written once (j=0 reduce) then re-written each j>0 (pop 1 in α-step, push 1 in copy-back), read no-pop by exp every iter; net 1 tile resident throughout, popped by `recip`. `cb_sum`: written j=0, in-place each j>0, popped by `recip`. `cb_out_accum`: written j=0 (matmul push Dt), in-place each j>0, popped by final normalize. Every per-block CB (`cb_k/v/mask/scores/p/pv/mblock/mnew/lblock/alpha`) has push count == pop count per KV iteration.

## API Mapping

All helpers are in `compute_kernel_lib::` (compute) / `dataflow_kernel_lib::` (dataflow). `CbA/CbB/...` are CB indices.

| Phase | Type | Function | File:Line | Template / Args | Input CB | Output CB | Manages own CB ops? |
|-------|------|----------|-----------|------------------|----------|-----------|---------------------|
| scaler prep (max) | helper | `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:100` | `<cb_scaler_max, PoolType::MAX, ReduceDim::REDUCE_ROW>()` | — | `cb_scaler_max` (row-0 fill, 1.0) | yes (reserve+push) |
| scaler prep (sum) | helper | `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:100` | `<cb_scaler_sum, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | — | `cb_scaler_sum` (col-0 fill, 1.0) | yes |
| reader Q/K/V/mask | raw_api | `TensorAccessor` / `noc_async_read` | `tech_reports/tensor_accessor/tensor_accessor.md` | tiled interleaved addressing | DRAM | `cb_q/cb_k/cb_v/cb_mask` | reader does cb_reserve/push |
| 0. scale Q | helper | `eltwise_chain` (`CopyTile`+`MulUnary`+`PackTile`) | `eltwise_chain.hpp:734`; `MulUnary` `eltwise_scalar.hpp:61` | `CopyTile<cb_q,D0,WaitAndPop>`, `MulUnary<D0>{scale_u32}`, `PackTile<cb_q,…PerTileReserveAndPush>` over `Dt` | `cb_q` | `cb_q` (in-place) | yes |
| A. QKᵀ | helper | `matmul_block` | `matmul_block_helpers.hpp:352` | `<transpose=true, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, in0=WaitAndRetainOnLastBlock, in1=WaitAndPopPerKBlock>`; `MatmulBlockShape::of(in0_sb=1, in1_sb=kv_chunk_t/sbw, sbh=1, sbw≤DEST, in0_block_k=Dt, num_k=1)` | `cb_q`(in0,retained), `cb_k`(in1,pop) | `cb_scores` (interm = `cb_scores`) | yes |
| B. mask add | helper | `binary_add` | `eltwise_convenience.hpp:47` | `<cb_scores, cb_mask, cb_scores, BroadcastDim::None>(kv_chunk_t)` | `cb_scores`,`cb_mask` | `cb_scores` (in-place) | yes |
| C. row-max | helper | `reduce` | `reduce_helpers_compute.hpp:388` | `<MAX, REDUCE_ROW, cb_scores, cb_scaler_max, OUT, WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, kv_chunk_t))`; OUT = `cb_max` (j=0) or `cb_mblock` (j>0) | `cb_scores`(kept), `cb_scaler_max` | `cb_max`/`cb_mblock` | waits upfront, **no pop** of input (kept for exp); reserves+pushes output |
| D1. m_new=max | helper | `eltwise_chain` (`CopyTile`×2 + `BinaryMax` + `PackTile`) | `eltwise_chain.hpp:734`; `BinaryMax` `eltwise_binary_sfpu.hpp:58` | `CopyTile<cb_max,D0,WaitNoPop>`, `CopyTile<cb_mblock,D1,WaitAndPop>`, `BinaryMax<D0,D1,D0>`, `PackTile<cb_mnew>` over 1 | `cb_max`(kept), `cb_mblock`(pop) | `cb_mnew` | yes |
| D2. α=exp(m_prev−m_new) | helper | `eltwise_chain` (`BinaryFpu`+`Exp`+`PackTile`) | `eltwise_chain.hpp:734`; `Exp` `eltwise_math.hpp:37` | `BinaryFpu<cb_max,cb_mnew,Sub,None,Input,WaitAndPop,WaitNoPop>`, `Exp<>`, `PackTile<cb_alpha>` over 1 | `cb_max`(pop m_prev), `cb_mnew`(kept) | `cb_alpha` | yes |
| D3. m_prev←m_new | helper | `copy` | `eltwise_convenience.hpp:110` | `<cb_mnew, cb_max>(1)` | `cb_mnew`(pop) | `cb_max` | yes |
| E. P=exp(scores−m) | helper | `eltwise_chain` (`BinaryFpu` bcast-col + `Exp` + `PackTile`) | `eltwise_chain.hpp:734` (pattern `:156-169`) | `BinaryFpu<cb_scores, cb_max, Sub, BroadcastDim::Col, Input, A=WaitAndPop, B=WaitNoPop, AIdx=FirstTile, D0, BIdx=FirstTile>`, `Exp<>`, `PackTile<cb_p>` over `kv_chunk_t` | `cb_scores`(pop), `cb_max`(kept, pinned tile0) | `cb_p` | yes |
| F. row-sum | helper | `reduce` | `reduce_helpers_compute.hpp:388` | `<SUM, REDUCE_ROW, cb_p, cb_scaler_sum, OUT, WaitUpfrontNoPop>(of(1, kv_chunk_t))`; OUT = `cb_sum` (j=0) or `cb_lblock` (j>0) | `cb_p`(kept), `cb_scaler_sum` | `cb_sum`/`cb_lblock` | waits upfront, no pop (kept for PV); reserves+pushes output |
| G. l_new=α·l+l_blk | helper | `eltwise_chain` (`BinaryFpu` + `DestReuseBinary` + `PackTile`) | `eltwise_chain.hpp:734`; `DestReuseBinary` decl `eltwise_chain.hpp:612`; `AddBinary` semantics via `DestReuseBinary<...,Add,...>` | `BinaryFpu<cb_sum,cb_alpha,Mul,None,Input,WaitAndPop,WaitNoPop>` → D0, `DestReuseBinary<cb_lblock,Add,DEST_TO_SRCA,...,WaitAndPop>` → D0, `PackTile<cb_sum>` over 1 | `cb_sum`(pop), `cb_alpha`(kept), `cb_lblock`(pop) | `cb_sum` (in-place) | yes |
| H. PV matmul | helper | `matmul_block` | `matmul_block_helpers.hpp:352` | `<transpose=false, OutputCBLayout::TileRowMajor, InitMode::Short, in0=WaitAndPopPerKBlock, in1=WaitAndPopPerKBlock>`; `MatmulBlockShape::of(in0_sb=1, in1_sb=Dt/sbw, sbh=1, sbw≤DEST, in0_block_k=kv_chunk_t, num_k=1)`; OUT = `cb_out_accum` (j=0) or `cb_pv` (j>0) | `cb_p`(in0,pop), `cb_v`(in1,pop) | `cb_out_accum`/`cb_pv` | yes |
| I. O_new=α·O+PV | helper | `eltwise_chain` (`BinaryFpu` bcast-col + `DestReuseBinary` + `PackTile`) | `eltwise_chain.hpp:734` | `BinaryFpu<cb_out_accum, cb_alpha, Mul, BroadcastDim::Col, Input, A=WaitAndPop, B=WaitNoPop, FirstTile, D0, FirstTile>` → D0, `DestReuseBinary<cb_pv, Add, DEST_TO_SRCA, …, WaitAndPop>` → D0, `PackTile<cb_out_accum>` over `Dt` | `cb_out_accum`(pop), `cb_alpha`(kept, pinned tile0), `cb_pv`(pop) | `cb_out_accum` (in-place) | yes; then raw `cb_pop_front(cb_alpha,1)` |
| J. recip(l) | helper | `unary` | `eltwise_convenience.hpp:81`; `Recip` `eltwise_math.hpp:58` | `<Recip<>, cb_sum, cb_recip>(1)` | `cb_sum`(pop) | `cb_recip` | yes |
| K. O·recip | helper | `eltwise_chain` (`BinaryFpu` bcast-col + `PackTile`) | `eltwise_chain.hpp:734` | `BinaryFpu<cb_out_accum, cb_recip, Mul, BroadcastDim::Col, Input, A=WaitAndPop, B=WaitNoPop, FirstTile, D0, FirstTile>`, `PackTile<cb_out>` over `Dt` | `cb_out_accum`(pop), `cb_recip`(kept, pinned tile0) | `cb_out` (bf16) | yes; then raw `cb_pop_front(cb_recip,1)` |
| L. release Q | raw_api | `cb_pop_front` | `api/dataflow/circular_buffer.h` | `cb_pop_front(cb_q, Dt)` | `cb_q` | — | — |
| writer O | raw_api | `TensorAccessor` / `noc_async_write` | `tensor_accessor.md` | tiled interleaved write | `cb_out` | DRAM O | writer does cb_wait/pop |

### Helpers considered and rejected (raw-API fallbacks)

- **Reader/writer DRAM addressing (`TensorAccessor` raw API).** No `kernel_lib` helper covers DRAM↔L1 tiled NoC transfer of Q/K/V/mask/O sub-blocks. `tilize_helpers.hpp` / `untilize_helpers.hpp` only convert RM↔tiled within L1 (`tilize_helpers.hpp:153`, `untilize_helpers.hpp:145`); the tensors are already TILE_LAYOUT so no conversion is needed. `TensorAccessor` is the standard mechanism and is required for the tile-index computation per work unit. **Justification: no covering helper exists.**
- **`cb_pop_front(cb_q, Dt)` (phase L) and `cb_pop_front(cb_alpha/cb_recip, 1)` (phases I/K).** These release inputs that a helper *intentionally left fronted*. The QKᵀ matmul's `in0=WaitAndRetainOnLastBlock` (`matmul_block_helpers.hpp:76`) deliberately does not pop `cb_q` so it can be reused across KV blocks — the helper's contract delegates the final pop to the caller. Likewise the `B=WaitNoPop` operand policy (`eltwise_chain.hpp:335`) keeps `cb_alpha`/`cb_recip` resident for reuse across the block and delegates the pop. **Justification: these are the documented caller-owned tail of retain/no-pop policies, not a hand-rolled replacement of any helper.**

All other phases use helpers. No compute phase hand-rolls an LLK loop that a helper covers.

## Compute Phases

State carried across the KV loop (per work unit): `cb_q` (scaled Q, resident), `cb_max` (running `m`), `cb_sum` (running `l`), `cb_out_accum` (running `O`).

| # | Operation | Helper? | Input CB (state) | Output CB (tiles) | CB State After |
|---|-----------|---------|------------------|-------------------|----------------|
| boot | `compute_kernel_hw_startup` + `mm_block_init` once | — | — | — | engine + matmul state configured |
| 0 | scale Q in-place by `scale` | yes | `cb_q` (Dt, raw) | `cb_q` (Dt, scaled) | `cb_q` holds scaled Q |
| **per KV block j (0…num_kv−1)** | | | | | |
| A | `cb_scores = (Q·Kⱼᵀ)` (Q pre-scaled) | yes | `cb_q`(retained), `cb_k`(pop) | `cb_scores` (kv_chunk_t, fp32) | `cb_q` kept; `cb_k` freed |
| B | `cb_scores += maskⱼ` (if `use_mask`) | yes | `cb_scores`, `cb_mask`(pop) | `cb_scores` (in-place) | `cb_mask` freed |
| C | row-max | yes | `cb_scores`(kept) | `cb_max`(j=0) / `cb_mblock`(j>0) | `cb_scores` still resident (no pop) |
| D1 | `m_new = max(m_prev, m_blk)` (j>0) | yes | `cb_max`(kept),`cb_mblock`(pop) | `cb_mnew` | — |
| D2 | `α = exp(m_prev − m_new)` (j>0) | yes | `cb_max`(pop),`cb_mnew`(kept) | `cb_alpha` | `cb_max` freed |
| D3 | `m_prev ← m_new` (j>0) | yes | `cb_mnew`(pop) | `cb_max` | `cb_max` = m_new |
| E | `cb_p = exp(cb_scores − m)` (bcast col) | yes | `cb_scores`(pop),`cb_max`(kept) | `cb_p` (kv_chunk_t, bf16) | `cb_scores` freed; `cb_max` resident |
| F | row-sum | yes | `cb_p`(kept) | `cb_sum`(j=0) / `cb_lblock`(j>0) | `cb_p` still resident |
| G | `l_new = α·l_prev + l_blk` (j>0) | yes | `cb_sum`(pop),`cb_alpha`(kept),`cb_lblock`(pop) | `cb_sum` (in-place) | `cb_sum` = l_new; `cb_alpha` kept |
| H | `PV = cb_p · Vⱼ` | yes | `cb_p`(pop),`cb_v`(pop) | `cb_out_accum`(j=0) / `cb_pv`(j>0) | `cb_p`,`cb_v` freed |
| I | `O_new = α·O_prev + PV` (j>0, bcast col) | yes | `cb_out_accum`(pop),`cb_alpha`(kept),`cb_pv`(pop) | `cb_out_accum` (in-place) | `cb_pv` freed; then pop `cb_alpha` |
| **after KV loop** | | | | | |
| J | `cb_recip = 1/cb_sum` | yes | `cb_sum`(pop) | `cb_recip` | `cb_sum` freed |
| K | `cb_out = cb_out_accum · recip` (bcast col, → bf16) | yes | `cb_out_accum`(pop),`cb_recip`(kept) | `cb_out` (Dt, bf16) | `cb_out_accum` freed; then pop `cb_recip` |
| L | release retained Q | raw | `cb_q`(pop Dt) | — | `cb_q` freed → next work unit |

**Online-softmax equivalence.** For block j: `m_new = max(m_prev, rowmax(Sⱼ))`, `α = exp(m_prev − m_new)`, `l_new = α·l_prev + rowsum(exp(Sⱼ − m_new))`, `O_new = α·O_prev + exp(Sⱼ − m_new)·Vⱼ`, with `Sⱼ = scale·Q·Kⱼᵀ + maskⱼ` and the final `O = O_accum / l`. All of `m,l,O` accumulate in fp32 — mathematically identical to the two-pass `softmax(scale·Q·Kᵀ+mask)·V` (exp's `−inf` from masked entries yields 0 contributions). Scale is folded by **pre-scaling Q once per work unit** (phase 0), so `Q·Kⱼᵀ` already carries `scale` and the additive mask is added in the correctly-scaled space.

## Broadcast Verification

Binary ops with a row-vector operand (`q_chunk_t = 1` → the broadcast operand is exactly 1 tile, pinned at tile 0, broadcast across columns of the block). `REDUCE_ROW` produces an `(N,1)` column-shaped result → downstream broadcast is `BroadcastDim::Col` (`eltwise_chain.hpp:374-385`).

| Phase | Op | CB_A valid region | CB_B valid region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| B (mask add) | `cb_scores + cb_mask` | `[1, kv_chunk_t]` All | `[1, kv_chunk_t]` All | None (element-wise) |
| D1 (max) | `BinaryMax(cb_max, cb_mblock)` | `[1,1]` Col0 | `[1,1]` Col0 | None (both `(1,1)`) |
| D2 (α) | `cb_max − cb_mnew` | `[1,1]` Col0 | `[1,1]` Col0 | None |
| E (P=exp) | `cb_scores − cb_max` | `[1, kv_chunk_t]` All | `cb_max` `[1,1]` Col0 (REDUCE_ROW out) | **Col** |
| G (l_new) | `cb_sum · cb_alpha` (+ `cb_lblock`) | `[1,1]` Col0 | `[1,1]` Col0 | None |
| I (O_new) | `cb_out_accum · cb_alpha` (+ `cb_pv`) | `[1, Dt]` All | `cb_alpha` `[1,1]` Col0 | **Col** |
| K (normalize) | `cb_out_accum · cb_recip` | `[1, Dt]` All | `cb_recip` `[1,1]` Col0 | **Col** |

The `Col` broadcasts (E, I, K) subtract/multiply a per-row scalar (the row's max / α / 1-over-sum, held in column 0 of the single stat tile) across all columns of the score/output block — exactly the documented softmax broadcast (`eltwise_chain.hpp:382-385`).

## Reduce Direction Verification

Both reductions are over the **key axis** (`W` of the `[q_rows, kv_cols]` block) → `REDUCE_ROW`.

| Logical reduce | Tile ReduceDim | Output valid region | Downstream BroadcastDim | ReduceInputBlockShape | Scaler fill |
|----------------|----------------|---------------------|-------------------------|-----------------------|-------------|
| row-max over keys (phase C) | `REDUCE_ROW` (MAX) | `(1,1)` Col0 | `Col` (phase E) | `of(1, kv_chunk_t)` | row-0 (MAX, `cb_scaler_max`) |
| row-sum over keys (phase F) | `REDUCE_ROW` (SUM) | `(1,1)` Col0 | `Col` (phases G/I via α, K via recip) | `of(1, kv_chunk_t)` | col-0 (SUM+REDUCE_ROW matmul path, `cb_scaler_sum`) |

Two separate scaler CBs are mandatory: MAX+REDUCE_ROW uses **row-0** fill while SUM+REDUCE_ROW uses **col-0** (matmul-path) fill — the pool-type-aware `calculate_and_prepare_reduce_scaler` overload selects the correct layout (`reduce_helpers_dataflow.hpp:46-48, 100`). Both scalers are value 1.0.

## Key Risks and Gotchas

- **Flash-Attention invariant (the whole point).** `cb_scores` and `cb_p` are sized `kv_chunk_t` tiles (one `[q_chunk_t × kv_chunk_t]` block), never `Sq_t × Skv_t`. The running `m`/`l`/`O` (`cb_max`/`cb_sum`/`cb_out_accum`) are the only state that survives a KV block, and they are per-Q-chunk (`O(S)` memory). Do not add a CB that holds the whole attention matrix.
- **fp32 accumulators are mandatory** for exact softmax. `cb_scores`, `cb_pv`, `cb_out_accum`, `cb_max`, `cb_sum`, `cb_alpha`, `cb_recip` are fp32. Compute config: `fp32_dest_acc_en = true`, `HiFi2` (bf16 inputs, K-blocked). Sub-block sizing must satisfy `out_subblock_h·out_subblock_w ≤ DEST_AUTO_LIMIT` (4 fp32 half-sync / 8 fp32 full-sync — `dest_helpers.hpp`).
- **Persistent CBs survive the KV loop.** `cb_q`, `cb_max`, `cb_sum`, `cb_out_accum` are written once and read/updated each KV iteration. `cb_q` is retained by the QKᵀ matmul (`in0=WaitAndRetainOnLastBlock`, `num_k_blocks=1`) and the caller pops it after the loop (phase L). The exp (E) reads `cb_max` with `WaitNoPop` so it persists.
- **Two reduce scaler CBs with different fill layouts** — see Reduce Direction Verification. Using one scaler for both reduces produces wrong results.
- **`TileRowMajor` output layout for both matmuls** so `cb_scores`/`cb_p`/`cb_pv` are in tile-row-major order, which is what `reduce` (`ReduceInputBlockShape::of(rows, cols)` contiguous) and the column-broadcast eltwise chains expect.
- **Scale folded into Q** (phase 0) so the additive mask is added in the scaled space — adding scale and mask in the wrong order is a correctness bug. `scale = 1/sqrt(D)` when `scale is None`; passed to compute as a bit-cast `uint32` runtime arg for `MulUnary`.
- **Mask is dense and additive.** The op never *generates* a causal mask — it adds whatever `(B,1,S_q,S_kv)` (or `(B,H,…)`) additive block the reader fetched. Masked rows have ≥ 1 finite entry per row for causal (the diagonal), so `m` becomes finite on the first processed KV block; `exp(−inf − m) = 0` makes masked positions contribute nothing. Processing KV blocks in increasing order keeps the first block finite per row.
- **Boot sequence.** `compute_kernel_hw_startup(...)` then `mm_block_init(...)` once at the top of `kernel_main`. `matmul_block` uses `InitMode::Short` to restore matmul state after the intervening reduce/eltwise helpers; `reduce` and `eltwise_chain` own their own per-call init. Never call `compute_kernel_hw_startup`/`mm_block_init` mid-kernel.

## Structural impossibilities (feature_spec.py — pipeline mode)

`feature_spec.py` already exists with `INVALID = []`. SDPA is TILE-only in `TARGET` (no ROW_MAJOR), so the canonical `bf8b + ROW_MAJOR` rule is vacuous and there are no additional structural impossibilities to fold in. No change to `feature_spec.py` is requested.

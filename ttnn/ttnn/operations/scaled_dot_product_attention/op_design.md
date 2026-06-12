# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + softmax + matmul, online) |
| Goal | Compute `softmax(Q·Kᵀ·scale + mask)·V` using the Flash Attention algorithm — tiled over the sequence dimension with online softmax, never materializing the full S_q × S_kv score matrix. |
| Math | `O[b,h,:,:] = softmax( (Q·Kᵀ)·scale + M , dim=-1 ) · V`, computed block-by-block via the running-max / running-sum / running-output recurrence (Tri Dao, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). |
| Mode | Hybrid (multi-phase fused compute kernel; generic_op ProgramDescriptor) |
| References | METALIUM_GUIDE.md; `.claude/references/ttnn-cb-memory-fundamentals.md`; `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`; `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`; `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`; `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp`; `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp`; `tech_reports/tensor_accessor/tensor_accessor.md` |

### Algorithm (per (batch b, head h, Q-block i))

Let `Q_i` be a `B_q × D` block of Q; iterate over all `K_j, V_j` blocks of size `B_kv × D`:

```
m_i = -inf  (per row of Q_i, shape B_q × 1)
l_i = 0     (per row, B_q × 1)
O_i = 0     (B_q × D)

for each KV block j:
    S      = Q_i · K_jᵀ · scale            # B_q × B_kv
    S      = S + M_ij                      # additive mask (custom mask tile, or 0 if mask_mode=none)
    m_blk  = rowmax(S)                      # B_q × 1
    m_new  = max(m_i, m_blk)                # B_q × 1   (running max via accumulating MAX-reduce)
    corr   = exp(m_i - m_new)               # B_q × 1   (rescale factor for prior state)
    P      = exp(S - m_new)                 # B_q × B_kv (broadcast m_new down columns)
    l_blk  = rowsum(P)                      # B_q × 1
    l_i    = corr * l_i + l_blk             # B_q × 1
    O_i    = corr * O_i + P · V_j           # B_q × D    (broadcast corr down columns of O_i)
    m_i    = m_new

O_i = O_i / l_i                             # B_q × D    (broadcast 1/l_i down columns)
```

The recurrence is mathematically exact (equivalent to two-pass softmax) when accumulators
`m_i`, `l_i`, `O_i` are kept across KV blocks in their CBs. The `S` (score) and `P`
(probability) blocks are only ever `B_q × B_kv` tiles — the full S_q × S_kv matrix is never
allocated in DRAM or in any CB.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| query | ttnn.Tensor | yes | (B, H, S_q, D), bf16, TILE | — | tensor |
| key | ttnn.Tensor | yes | (B, H, S_kv, D), bf16, TILE | — | tensor |
| value | ttnn.Tensor | yes | (B, H, S_kv, D), bf16, TILE | — | tensor |
| attn_mask | ttnn.Tensor | no | (B, 1, S_q, S_kv) or (B, H, S_q, S_kv), bf16, TILE, additive (0 keep / −inf drop) | None | tensor |
| is_causal | bool | no | {False, True} — True is a refinement (not Phase 0) | False | host flag |
| scale | float | no | any finite > 0 | None → 1/√D | RT (CT-foldable) scalar |
| q_chunk_t | uint (CT) | derived | tiles per Q block (B_q/32); recommend 2–4 | min(S_q/32, 4) | CT |
| k_chunk_t | uint (CT) | derived | tiles per KV block (B_kv/32); recommend 2–4 | min(S_kv/32, 4) | CT |

`is_causal` and `attn_mask` are mutually exclusive (Torch's rule) — the entry point raises
`ValueError` if both are set. `mask_mode` and `scale_mode` are not tensor axes; they are
derived in `validate()` from these scalar kwargs (see Registry Model below).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q: (B, H, S_q, D); K, V: (B, H, S_kv, D); S_q may differ from S_kv (cross-attention). mask: (B, {1,H}, S_q, S_kv) when present |
| Dtype | bfloat16 (Phase 0). float32 / bfloat8_b are TARGET refinements |
| Layout | TILE_LAYOUT |
| Memory | DRAM or L1, interleaved |
| Alignment | Phase 0: S_q, S_kv, D divisible by 32 (tile_aligned) |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H, S_q, D) — same as Q |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved (follows input memory_config) |

## Dataflow Strategy

**DRAM → Tensix → DRAM, fully tiled, no inter-Tensix communication.** Each core owns a set of
independent `(b, h, q_block)` work units and computes them sequentially. There is no
multicast, no semaphore handshake, and no ring topology — every work unit is fully local to
one core, so the only synchronization is the per-core reader→compute→writer CB pipeline.

Per work unit `(b, h, i)`:

1. **Reader (NCRISC / RISCV_1)** reads the Q block `Q_i` (`q_chunk_t × D_t` tiles) once into
   `cb_q_in` via a `TensorAccessor` over the Q DRAM buffer. It also fills the two constant
   CBs once at kernel start: `cb_scale` (a single tile whose `[0][0]` element = `scale`, used
   as a scalar-broadcast multiplier) and `cb_reduce_scaler` (a single bf16 tile = 1.0 for both
   the MAX and SUM reductions). Then, for each KV block `j`, it streams `K_j` into `cb_k_in`
   (`k_chunk_t × D_t` tiles), `V_j` into `cb_v_in` (`k_chunk_t × D_t` tiles), and — when
   `mask_mode == custom` — the matching `M_ij` mask block into `cb_mask_in`
   (`q_chunk_t × k_chunk_t` tiles).
2. **Compute (TRISC unpack/math/pack)** runs the Flash Attention recurrence above, holding
   `Q_i` resident in `cb_q_in` across the whole KV loop and maintaining the running
   accumulators `cb_max`, `cb_l`, `cb_o_acc` across KV blocks. It emits the final normalized
   `O_i` (`q_chunk_t × D_t` tiles) into `cb_out`.
3. **Writer (BRISC / RISCV_0)** writes `O_i` from `cb_out` back to the output DRAM buffer via a
   `TensorAccessor`.

Data format is **tiled (32×32) at every stage** — Q, K, V, mask, and output all arrive and
leave in TILE_LAYOUT, so no tilize/untilize is needed.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one `(batch b, head h, Q-block i)` triple — produces one `q_chunk_t × D_t` output block |
| Total units | `B * H * num_q_blocks`, where `num_q_blocks = ceil(S_q / (q_chunk_t·32))` |
| Grid | `device.compute_with_storage_grid_size()` (all available Tensix cores) |
| Per-core work | `ttnn.split_work_to_cores(total_units, num_cores)` → each core gets `units_per_core` or `units_per_core+1` contiguous unit indices; the host passes each core its `[start_unit, num_units)` range as runtime args, and the kernel decodes `(b, h, i)` from a flat unit index |
| KV loop | each unit iterates over `num_kv_blocks = ceil(S_kv / (k_chunk_t·32))` KV blocks |
| Remainder | uneven `total_units / num_cores` handled by `split_work_to_cores` giving the first `remainder` cores one extra unit. Phase 0 requires tile-aligned S_q/S_kv so chunk counts divide cleanly; the last Q/KV chunk may be a partial number of tiles only when S is not a multiple of the chunk size (handled by a per-core `num_kv_blocks`/last-chunk tile count passed as runtime args). |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| cb_q_in | 0 | tile_size(bf16) | `q_chunk_t * D_t` (single block; +1 if double-buffering across Q-blocks → `2 * q_chunk_t * D_t`) | bf16 | reader | compute (QKᵀ in0) | held resident across the entire KV loop of one Q-block |
| cb_k_in | 1 | tile_size(bf16) | `2 * k_chunk_t * D_t` (double-buffered streaming) | bf16 | reader | compute (QKᵀ in1, transposed) | one KV block, consumed by QKᵀ matmul |
| cb_v_in | 2 | tile_size(bf16) | `2 * k_chunk_t * D_t` (double-buffered streaming) | bf16 | reader | compute (P·V in1) | one KV block, consumed by P·V matmul |
| cb_mask_in | 3 | tile_size(bf16) | `2 * q_chunk_t * k_chunk_t` (double-buffered streaming; allocated only when mask_mode=custom) | bf16 | reader | compute (score add) | one KV block, consumed by mask-add |
| cb_scale | 8 | tile_size(bf16) | 1 | bf16 | reader (fill once) | compute (scalar-bcast mul) | constant for whole kernel |
| cb_reduce_scaler | 9 | tile_size(bf16) | 1 | bf16 | reader (fill once) | compute (MAX & SUM reduce) | constant for whole kernel |
| cb_max | 10 | tile_size(bf16) | `q_chunk_t` (full column block) | bf16 | compute (accumulating MAX-reduce) | compute | running max m_i, persists across KV loop |
| cb_max_prev | 11 | tile_size(bf16) | `q_chunk_t` | bf16 | compute (copy of m_i before update) | compute | scratch within one KV iteration (holds m_prev) |
| cb_corr | 12 | tile_size(bf16) | `q_chunk_t` | bf16 | compute (exp(m_prev−m_new)) | compute | scratch within one KV iteration |
| cb_l | 13 | tile_size(bf16) | `q_chunk_t` | bf16 | compute (running sum update) | compute | running sum l_i, persists across KV loop |
| cb_l_block | 14 | tile_size(bf16) | `q_chunk_t` | bf16 | compute (SUM-reduce of P) | compute | scratch within one KV iteration |
| cb_qk | 24 | tile_size(bf16) | `q_chunk_t * k_chunk_t` (full score block) | bf16 | compute (QKᵀ matmul) | compute (scale/mask/reduce/sub) | scratch within one KV iteration; **B_q × B_kv only, never the full matrix** |
| cb_p | 25 | tile_size(bf16) | `q_chunk_t * k_chunk_t` (full prob block) | bf16 | compute (exp) | compute (SUM-reduce, P·V matmul in0) | scratch within one KV iteration |
| cb_o_acc | 26 | tile_size(bf16) | `q_chunk_t * D_t` (full output block) | bf16 | compute (rescale + add PV) | compute | running output O_i, persists across KV loop |
| cb_pv | 27 | tile_size(bf16) | `q_chunk_t * D_t` (full block) | bf16 | compute (P·V matmul) | compute (O_i rescale-add) | scratch within one KV iteration |
| cb_out | 16 | tile_size(bf16) | `2 * q_chunk_t * D_t` (double-buffered) | bf16 | compute (final normalize) | writer | one Q-block output, then drained to DRAM |

`D_t = D/32`, `q_chunk_t = B_q/32`, `k_chunk_t = B_kv/32`.

**CB sync (producer push = consumer wait), per KV iteration:**
- `cb_q_in`: pushed once (reader), waited each KV iteration by the QKᵀ matmul (retained, not
  popped); popped once by compute at end of Q-block.
- `cb_k_in`, `cb_v_in`: pushed once per KV block (reader), waited/popped once per KV block
  (compute matmuls).
- `cb_mask_in`: pushed once per KV block (reader, custom mask only), waited/popped once per KV
  block (compute add).
- Running accumulators `cb_max`, `cb_l`, `cb_o_acc`: written and re-read in place each KV
  iteration; one net push/pop per KV iteration balanced inside the helpers/eltwise ops.
- `cb_out`: pushed once per Q-block (compute), waited/popped once per Q-block (writer).

## API Mapping

Every mechanism has a verified file:line reference into `ttnn/cpp/ttnn/kernel_lib/`.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| Boot | raw_api | `mm_block_init()` (or `compute_kernel_hw_startup()`+`mm_init()`) | matmul_block_helpers.hpp:134 | — | — | — | Called exactly once at top of compute kernel_main; `matmul_block` Short init handles all later matmul re-inits |
| S = Q·Kᵀ·(implicit) | helper | `matmul_block` | matmul_block_helpers.hpp:810 | `<transpose=true, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, in0_policy=WaitAndRetainOnLastBlock, in1_policy=WaitAndPopPerKBlock>` | in0=`cb_q_in`, in1=`cb_k_in` | `cb_qk` (interm=`cb_qk`) | `MatmulBlockShape::of(q_chunk_t, k_chunk_t, out_sb_h, out_sb_w, in0_block_k=D_t, num_k_blocks=1)`; transpose=true gives Q·Kᵀ (K stored [B_kv,D], transposed to [D,B_kv]); Q retained across KV loop via WaitAndRetainOnLastBlock + num_k_blocks=1 |
| Scale | helper | `mul` (eltwise scalar broadcast) | eltwise_convenience.hpp:93 | `mul<cb_qk, cb_scale, cb_qk, BroadcastDim::Scalar>` (in-place: out CB = in CB) | `cb_qk`, `cb_scale` | `cb_qk` | applies `S *= scale`; `cb_scale[0][0] = scale`, broadcast across the whole tile (BroadcastDim::Scalar) |
| Mask add (custom) | helper | `add` (eltwise) | eltwise_convenience.hpp:55 | `add<cb_qk, cb_mask_in, cb_qk, BroadcastDim::None>` | `cb_qk`, `cb_mask_in` | `cb_qk` | only when mask_mode=custom; elementwise add of caller mask block; `BroadcastDim::None` (same q×kv tile shape) |
| m_blk + running max | helper | `reduce` (accumulating MAX) | reduce_helpers_compute.hpp:407 | `reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>` with `Accumulate(AccumulationConfig::with_cb(cb_max), kv_block_idx)` | `cb_qk` (scaled+masked), `cb_reduce_scaler` | `cb_max` | `ReduceInputBlockShape::of(q_chunk_t, k_chunk_t)`; accumulating MAX reduces this block's row-max into the running max `cb_max` → m_new = max(m_prev, rowmax(S)); WaitUpfrontNoPop keeps `cb_qk` resident for the later subtract |
| Backup m_prev | helper | `copy` (eltwise) | eltwise_convenience.hpp:179 | `copy<cb_max, cb_max_prev>` issued **before** the accumulating reduce above | `cb_max` | `cb_max_prev` | snapshots m_prev so corr can be computed after `cb_max` is updated to m_new; on the first KV block m_prev is the −inf init (correction = 0, handled by skip — see Compute Phases) |
| Correction = exp(m_prev−m_new) | helper | `sub` + `unary<Exp>` (eltwise_chain) | eltwise_convenience.hpp:74 / eltwise_math.hpp:21 | `sub<cb_max_prev, cb_max, cb_corr>` then `unary<Exp<>, cb_corr, cb_corr>` (or one fused `eltwise_chain(col(q_chunk_t), BinaryFpu<cb_max_prev,cb_max,Sub>{}, Exp<>{}, PackTile<cb_corr>{})`) | `cb_max_prev`, `cb_max` | `cb_corr` | column block `EltwiseShape::col(q_chunk_t)`; per-row scalar correction factor |
| P = exp(S − m_new) | helper | `sub` (bcast col) + `unary<Exp>` fused via `eltwise_chain` | eltwise_chain.hpp:576 ; eltwise_convenience.hpp:74,135 ; eltwise_math.hpp:21 | `eltwise_chain(EltwiseShape::grid(q_chunk_t, k_chunk_t), BinaryFpu<cb_qk, cb_max, BinaryFpuOp::Sub, BroadcastDim::Col, InputLifecycle::Streaming, InputLifecycle::HeldStream>{}, Exp<>{}, PackTile<cb_p>{})` | `cb_qk`, `cb_max` | `cb_p` | subtract m_new broadcast **down columns** (BroadcastDim::Col: m_new is `q_chunk_t × 1`, applied to every kv column), then exp; `cb_max` held (HeldStream) for reuse; `cb_qk` popped here |
| l_blk = rowsum(P) | helper | `reduce` (SUM) | reduce_helpers_compute.hpp:407 | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>` | `cb_p`, `cb_reduce_scaler` | `cb_l_block` | `ReduceInputBlockShape::of(q_chunk_t, k_chunk_t)`; WaitUpfrontNoPop keeps `cb_p` for the P·V matmul |
| l_i = corr·l_i + l_blk | helper | `mul` (in-place) + `add` (eltwise) | eltwise_convenience.hpp:93,55 | `mul<cb_l, cb_corr, cb_l>` then `add<cb_l, cb_l_block, cb_l>` (`BroadcastDim::None`) | `cb_l`, `cb_corr`, `cb_l_block` | `cb_l` | column block `EltwiseShape::col(q_chunk_t)`; running sum update; on first KV block l_i seeded to l_blk (skip the corr·l_i term) |
| PV = P · V_j | helper | `matmul_block` | matmul_block_helpers.hpp:810 | `<transpose=false, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, in0_policy=WaitAndPopPerKBlock, in1_policy=WaitAndPopPerKBlock>` | in0=`cb_p`, in1=`cb_v_in` | `cb_pv` (interm=`cb_pv`) | `MatmulBlockShape::of(q_chunk_t, D_t, out_sb_h, out_sb_w, in0_block_k=k_chunk_t, num_k_blocks=1)`; K-dim = B_kv; pops `cb_p` and `cb_v_in` |
| O_i = corr·O_i + PV | helper | `mul` (bcast col, in-place) + `add` (eltwise) | eltwise_convenience.hpp:93,55 | `mul<cb_o_acc, cb_corr, cb_o_acc, BroadcastDim::Col>` then `add<cb_o_acc, cb_pv, cb_o_acc>` | `cb_o_acc`, `cb_corr`, `cb_pv` | `cb_o_acc` | `EltwiseShape::grid(q_chunk_t, D_t)`; corr broadcast down columns of O across D; on first KV block O_i seeded to PV (skip the corr·O_i term) |
| Normalize O_i = O_i / l_i | helper | `reduce` post-op `Recip` then `mul` (bcast col) | reduce_helpers_compute.hpp:407 / eltwise_math.hpp:33 / eltwise_convenience.hpp:93 | compute `1/l_i` into `cb_l` via `unary<Recip<>, cb_l, cb_l>` (eltwise_convenience.hpp:135), then `mul<cb_o_acc, cb_l, cb_out, BroadcastDim::Col>` | `cb_o_acc`, `cb_l` | `cb_out` | final per-row normalization, broadcast 1/l_i down columns; emitted to `cb_out` for the writer |
| Scale tile fill | helper | `prepare_reduce_scaler`-style fill / generic tile fill | reduce_helpers_dataflow.hpp:65 | reader fills `cb_scale[0][0] = scale` (scalar-broadcast tile) | — | `cb_scale` | bf16; one tile; alternatively folded as a CT-known constant if scale is compile-time |
| Reduce scaler fill | helper | `prepare_reduce_scaler` | reduce_helpers_dataflow.hpp:65 | `prepare_reduce_scaler<cb_reduce_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>(1.0f)` for MAX path; **also** a SUM-layout scaler — see note | — | `cb_reduce_scaler` | bf16, value 1.0. **MAX+REDUCE_ROW uses row-0 fill; SUM+REDUCE_ROW uses col-0 (matmul) fill — these layouts differ.** Provide two scaler tiles (one per pool type) or two scaler CBs. Pool-type-aware overload is mandatory. |
| Reader/Writer addressing | raw_api | `TensorAccessor` | tech_reports/tensor_accessor/tensor_accessor.md | built from `TensorAccessorArgs(tensor)` CT args | DRAM Q/K/V/mask buffers / DRAM out buffer | CBs | tiled interleaved page access; tile id = `((b*H + h)*S + s_tile_row)*D_t + d_tile` style indexing per tensor |

### Helpers considered and rejected (raw-API fallbacks)

- **Running max via accumulating MAX-reduce, not a binary-max helper.** The online-softmax
  running max `m_new = max(m_prev, rowmax(S))` needs an element-wise max of two `q_chunk_t × 1`
  column tiles. The eltwise FPU op set has no Max — `BinaryFpuOp` is `{Add, Sub, Mul}` only
  (eltwise_chain.hpp:401) — and the SFPU binary set in eltwise_binary_sfpu.hpp (lines 39–310)
  has no `MaxBinary`. Rather than drop to a raw `binary_max_tile` LLK call, the design uses the
  `reduce` helper's built-in `Accumulate` path (reduce_helpers_compute.hpp:222) with
  `PoolType::MAX`, which reloads the running accumulator from `cb_max` and folds the new
  block's row-max into it via the reduce LLK's native max accumulation. This keeps the running
  max inside a helper (no raw LLK), at the cost of one extra `copy` to snapshot m_prev.
- **No raw matmul / reduce / exp.** Every compute phase maps onto `matmul_block`, `reduce`, or
  the eltwise helpers (`add`/`sub`/`mul`/`copy`/`unary<Exp>`/`unary<Recip>` and `eltwise_chain`).
  The only raw CB op is a single `cb_pop_front(cb_q_in, q_chunk_t*D_t)` after each Q-block's KV
  loop, because `matmul_block`'s `WaitAndRetainOnLastBlock` in0 policy deliberately never pops
  the retained Q operand (matmul_block_helpers.hpp:97, doc 80–96) — the caller must release it
  to admit the next Q-block. No helper covers "release a retained matmul input," so this single
  raw pop is unavoidable and correct.

## Compute Phases

Per Q-block: phases A–B run once; phases C–K run once per KV block; phases L–M run once at the
end. "First KV block" (j=0) seeds the accumulators directly (no correction term).

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| A | Boot init | raw | — | — | matmul state initialized (once per kernel) |
| B | Q resident | reader push | `cb_q_in` (q_chunk_t·D_t) pushed once | — | Q held resident; m_i, l_i, O_i conceptually −inf/0/0 |
| C | S = Q·Kᵀ | matmul_block (transpose) | `cb_q_in` (retained), `cb_k_in` (k_chunk_t·D_t) | `cb_qk` (q_chunk_t·k_chunk_t) | `cb_q_in` retained; `cb_k_in` popped |
| D | S *= scale | mul (scalar bcast) | `cb_qk`, `cb_scale` | `cb_qk` (in place) | — |
| E | S += M_ij (custom only) | add | `cb_qk`, `cb_mask_in` | `cb_qk` (in place) | `cb_mask_in` popped |
| F | snapshot m_prev | copy | `cb_max` (q_chunk_t) | `cb_max_prev` | only for j>0 (j=0: m_prev is init) |
| G | m_new = max(m_i, rowmax(S)) | reduce MAX + Accumulate | `cb_qk` (WaitUpfrontNoPop), `cb_reduce_scaler` | `cb_max` (q_chunk_t) | `cb_qk` retained (not popped) |
| H | corr = exp(m_prev − m_new) | eltwise_chain Sub+Exp | `cb_max_prev`, `cb_max` | `cb_corr` (q_chunk_t) | j=0: corr unused |
| I | P = exp(S − m_new) | eltwise_chain Sub(bcast col)+Exp | `cb_qk`, `cb_max` (held) | `cb_p` (q_chunk_t·k_chunk_t) | `cb_qk` popped |
| J | l_blk = rowsum(P); l_i = corr·l_i + l_blk | reduce SUM + mul + add | `cb_p` (WaitUpfrontNoPop), `cb_reduce_scaler`, `cb_l`, `cb_corr` | `cb_l_block`, `cb_l` | `cb_p` retained for P·V; j=0: l_i = l_blk |
| K | PV = P·V_j; O_i = corr·O_i + PV | matmul_block + mul(bcast col) + add | `cb_p`, `cb_v_in`, `cb_o_acc`, `cb_corr`, `cb_pv` | `cb_pv`, `cb_o_acc` | `cb_p`, `cb_v_in` popped; j=0: O_i = PV |
| L | 1/l_i | unary Recip | `cb_l` | `cb_l` (in place) | runs after KV loop |
| M | O_i /= l_i (normalize) | mul (bcast col) | `cb_o_acc`, `cb_l` | `cb_out` (q_chunk_t·D_t) | `cb_o_acc` freed; output pushed |
| N | release Q | raw cb_pop_front | `cb_q_in` | — | Q slot freed for next Q-block |

**First-KV-block handling (j=0):** instead of `m_prev`/correction, the design either (a) seeds
`cb_max` to −large, `cb_l` to 0, `cb_o_acc` to 0 and runs the generic path (corr = exp(−inf −
m_new) = 0, so corr·prev = 0), or (b) branches on `j==0` to write `cb_l = l_blk` and
`cb_o_acc = PV` directly. Approach (b) avoids materializing a −inf tile and is recommended; the
branch is a compile-time-free runtime `if (j == 0)`.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| D (scale) | mul | `cb_qk` [q_chunk_t × k_chunk_t] All | `cb_scale` scalar at [0][0] | Scalar |
| E (mask) | add | `cb_qk` [q_chunk_t × k_chunk_t] All | `cb_mask_in` [q_chunk_t × k_chunk_t] All | None |
| H (corr) | sub | `cb_max_prev` [q_chunk_t × 1] Col0 | `cb_max` [q_chunk_t × 1] Col0 | None (both column) |
| I (P) | sub | `cb_qk` [q_chunk_t × k_chunk_t] All | `cb_max` [q_chunk_t × 1] Col0 (per-row max) | Col (broadcast the single max column across all kv columns) |
| J (l update) | mul, add | `cb_l` [q_chunk_t × 1] Col0 | `cb_corr` / `cb_l_block` [q_chunk_t × 1] Col0 | None |
| K (O rescale) | mul | `cb_o_acc` [q_chunk_t × D_t] All | `cb_corr` [q_chunk_t × 1] Col0 | Col (corr broadcast across D columns) |
| M (normalize) | mul | `cb_o_acc` [q_chunk_t × D_t] All | `cb_l` (=1/l_i) [q_chunk_t × 1] Col0 | Col |

`REDUCE_ROW` reductions (MAX of S over kv columns, SUM of P over kv columns) produce a
`q_chunk_t × 1` per-row column result (valid region Col0). Subtracting/multiplying it back into
the `q_chunk_t × W` block therefore uses **`BroadcastDim::Col`** (the single valid column is
replicated across W). This matches the reduce-out → broadcast-col convention in the template.

## Registry Model (op file contract)

Declared inline in `scaled_dot_product_attention.py`:

- **INPUT_TAGGERS**
  - `tag_alignment(inputs, axes)` → examines Q's last two dims `(S_q, D)` = `inputs[0][-2], inputs[0][-1]`:
    `"w_non_aligned"` if `D % 32 != 0`; else `"h_non_aligned"` if `S_q % 32 != 0`; else `"tile_aligned"`.
  - `tag_attention_kind(inputs, axes)` → `"self"` if `inputs[0][-2] == inputs[1][-2]` (S_q == S_kv), else `"cross"`.
  - `tag_kv_heads(inputs, axes)` → compares Q's head count `inputs[0][1]` to K's `inputs[1][1]`:
    `"mha"` if equal; `"mqa"` if `inputs[1][1] == 1`; else `"gqa"` (assumes `H_q % H_kv == 0`).
    **This axis is required by the authoritative golden `feature_spec.py`** (`kv_heads_mode`
    TARGET = `[mha, gqa, mqa]`), so the op file MUST declare it even though the task's minimal
    tagger list omitted it. GQA/MQA require the kernel to broadcast each KV head across its group
    of Q heads (work-unit `(b, h_q, i)` reads K/V from head `h_q // (H_q/H_kv)`).
- **mask_mode / scale_mode are derived in `validate()`** from scalar kwargs (NOT taggers):
  - `mask_mode`: `"causal"` if `is_causal`; `"custom"` if `not is_causal and attn_mask is not None`; else `"none"`.
  - `scale_mode`: `"auto"` if `scale is None` else `"explicit"`.
  - Raise `ValueError` if `is_causal and attn_mask is not None` (mutually exclusive — Torch rule).
- **SUPPORTED (Phase 0):** `dtype=[bfloat16]`, `layout=[TILE_LAYOUT]`, `alignment=[tile_aligned]`,
  `attention_kind=[self, cross]`, `kv_heads_mode=[mha]`, `mask_mode=[none, custom]`,
  `scale_mode=[auto, explicit]`. (`kv_heads_mode=[gqa, mqa]` are TARGET refinements — they add KV
  head-broadcast to the reader but no new compute.)
- **EXCLUSIONS (Phase 0):** none.
- **validate()** is the first line of the public entry point. It builds the axes dict (dtype,
  layout from Q; alignment + attention_kind from taggers; mask_mode + scale_mode from kwargs),
  checks each axis against SUPPORTED (raise `UnsupportedAxisValue`), then EXCLUSIONS (raise
  `ExcludedCell`). It additionally raises `ValueError`/`RuntimeError` for tensor-shape contract
  violations: non-4D rank; Q.D ≠ K.D; K.S_kv ≠ V.S_kv; Q.B ≠ K.B or Q.H ≠ K.H (with K/V
  broadcastable per attn convention); incompatible mask dims.

### Causal refinement contract (arms when `causal` joins SUPPORTED[mask_mode])

Per the task `## Rules` — to be implemented when `mask_mode=causal` is added:

- Mask is generated **on-device** from `is_causal` (never a caller tensor, never a materialized
  full S_q × S_kv mask), and added to each score block before the running max (Phase E slot).
- Three regions per (Q-block i, KV-block j): blocks entirely in the past → unmasked; blocks
  entirely in the future → whole-tile −inf, **skipped outright** (no QKᵀ/softmax/PV — the
  ≈½-KV-work causal perf win); only the diagonal-straddling block gets a per-element triangular
  −inf mask generated on-device.
- Add EXCLUSION `{"mask_mode": "causal", "attention_kind": "cross"}` (raise
  `NotImplementedError`) — causal requires S_q == S_kv.
- Raise `ValueError` when `is_causal=True` and `attn_mask is not None` (already enforced in
  Phase 0's mask_mode derivation).

## Key Risks and Gotchas

- **O(S) memory is the load-bearing constraint.** `cb_qk` and `cb_p` MUST be sized to exactly
  `q_chunk_t × k_chunk_t` tiles (the per-block score/prob), never `S_q/32 × S_kv/32`. No CB and
  no DRAM buffer may hold the full attention matrix. This is what distinguishes Flash Attention
  from plain SDPA.
- **Running accumulators persist across the KV loop.** `cb_max` (m_i), `cb_l` (l_i), `cb_o_acc`
  (O_i) are read-modify-written in place each KV iteration and must NOT be popped/re-allocated
  between blocks. They are sized to a full block (`q_chunk_t` or `q_chunk_t·D_t`) — never
  double-buffered as streaming CBs.
- **Q is retained, not popped, by the QKᵀ matmul.** Use `in0_policy=WaitAndRetainOnLastBlock`
  with `num_k_blocks=1` so Q survives every KV iteration; the compute kernel must issue a single
  explicit `cb_pop_front(cb_q_in, ...)` after the KV loop (the one sanctioned raw CB op).
- **Two reduce scaler layouts.** MAX+REDUCE_ROW uses row-0 fill; SUM+REDUCE_ROW uses col-0
  (matmul) fill (reduce_helpers_dataflow.hpp:46–47). Provide a separate scaler tile/CB per pool
  type using the pool-type-aware `prepare_reduce_scaler<cb, PoolType, ReduceDim>` overload — the
  legacy single-arg overload is forbidden.
- **Numerical exactness requires fp32 DEST accumulation** for the online recurrence. The
  matmuls and reductions must accumulate in fp32 (DST_ACCUM_MODE) so the running-max rescale is
  exact; output is repacked to bf16. The recurrence is mathematically equivalent to two-pass
  softmax only when m_i/l_i/O_i carry full precision across blocks.
- **DEST capacity.** With fp32 DEST the limit is 4 tiles (dest_helpers.hpp:102). Score/prob/
  output blocks (`q_chunk_t·k_chunk_t`, `q_chunk_t·D_t`) exceed DEST, so `matmul_block` and
  `reduce` must sub-block internally (they do); eltwise ops stream per-tile. Choose
  `out_subblock_h/w` so each subblock ≤ DEST limit.
- **Broadcast direction.** Per-row stats (max, sum, corr) are `q_chunk_t × 1` columns; folding
  them back into `q_chunk_t × W` blocks is `BroadcastDim::Col`. Getting this wrong (using Row)
  silently corrupts softmax.
- **First KV block** seeds accumulators directly (l_i=l_blk, O_i=PV, m_i=m_new) to avoid a
  −inf correction tile; runtime `if (j==0)` branch.
- **Cross-attention (S_q ≠ S_kv)** only changes `num_kv_blocks` vs `num_q_blocks`; the per-block
  kernel is identical. K/V batch & head dims must match Q (or broadcast per attn_mask convention).

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB (per-KV-iteration balance verified above)
- [x] Reduce scaler CB is bfloat16, value 1.0
- [x] Reduce scaler uses pool-type-aware API (`prepare_reduce_scaler<cb, PoolType, ReduceDim>`), separate fill per pool type (MAX row-0 vs SUM col-0)
- [x] DEST: fp32 accumulation → 4-tile limit; matmul/reduce sub-block internally
- [x] Sequential helper intermediates (`cb_qk`, `cb_p`, `cb_o_acc`, `cb_pv`) sized to full block
- [x] Page sizes = tile_size(bf16) for all tile CBs
- [x] All cb_wait_front on a given CB use the same page count
- [x] Helpers not wrapped with extra CB ops (sole exception: documented `cb_pop_front(cb_q_in)`)
- [x] `mm_block_init()` / `compute_kernel_hw_startup()` called once before any helper use

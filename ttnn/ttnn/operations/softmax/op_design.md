# Operation Design: softmax

## Overview

| Field | Value |
|-------|-------|
| Classification | compute |
| Goal | Numerically-stable softmax along the last (W) or second-to-last (H) dimension of a 4D tile-aligned float32 tensor |
| Math | `output[n,c,h,w] = exp(x[n,c,h,w] - max(x[n,c,row_or_col])) / sum(exp(x[n,c,h,w] - max(x[n,c,row_or_col])))` where the max and sum are along `dim` (-1 = W, -2 = H) |
| Mode | Derivative |
| References | `reduce_helpers_compute.hpp`, `reduce_helpers_dataflow.hpp`, `eltwise_chain.hpp`, `eltwise_convenience.hpp`, `eltwise_math.hpp`, `eltwise_binary_sfpu.hpp`, `dest_helpers.hpp`, `compute_kernel_hw_startup.h` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | float32, TILE_LAYOUT, rank 4, H/W divisible by 32 | — | — |
| dim | int | no | -1, -2 (canonicalized: `dim if dim < 0 else dim - ndim`) | -1 | CT |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | no (keyword-only) | fp32_dest_acc_en=True required for float32 input; math_fidelity/math_approx_mode accepted at any value | `default_compute_kernel_config()` | CT |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | 4D (N, C, H, W); H and W divisible by 32 |
| Dtype | float32 |
| Layout | TILE_LAYOUT |
| Memory | DRAM or L1 (interleaved) |

### Output

| Property | Value |
|----------|-------|
| Shape | Same as input (N, C, H, W) |
| Dtype | float32 |
| Layout | TILE_LAYOUT |
| Memory | Same as input |

## Dataflow Strategy

Data enters as tiles (TILE_LAYOUT, float32). No tilize/untilize is needed — input is already tiled and output preserves the layout.

**Intra-Tensix data path:**

1. **Reader (NCRISC/RISCV_1):** Reads input tiles from DRAM/L1 into `cb_input_tiles` in row-major tile order (n, c, ht, wt). Also prepares two constant scaler tiles (one for MAX reduce, one for SUM reduce) via `prepare_reduce_scaler` once at kernel start.
2. **Compute (TRISC 0/1/2):** Runs the 4-phase softmax pipeline using reduce and eltwise-chain helpers. Intermediate results flow through L1 CBs between phases.
3. **Writer (BRISC/RISCV_0):** Reads output tiles from `cb_output_tiles` and writes them to DRAM/L1 in the same row-major tile order.

**No inter-Tensix communication.** Each core independently processes its assigned (N, C) slabs. No multicast, semaphores, or ring topology required.

**Per-slab processing (dim=-1 example, W reduction):**

- Reader pushes Ht×Wt tiles for one (n, c) slab into `cb_input_tiles`.
- Compute Phase 1 (max reduce) waits for all Ht×Wt tiles, produces Ht max tiles in `cb_max`. Input tiles are NOT popped (WaitUpfrontNoPop) — retained for Phase 2.
- Compute Phase 2 (fused sub+exp) reads input tiles (Bulk lifecycle, pops at end) and max tiles (HeldBulk, broadcast across Wt), produces Ht×Wt exp tiles in `cb_exp`.
- Compute Phase 3 (sum reduce with fused recip) waits for all Ht×Wt exp tiles, produces Ht recip-sum tiles in `cb_recip_sum`. Exp tiles NOT popped — retained for Phase 4.
- Compute Phase 4 (mul) reads exp tiles (Bulk, pops at end) and recip-sum tiles (HeldBulk, broadcast across Wt), produces Ht×Wt output tiles in `cb_output_tiles`.
- Writer drains `cb_output_tiles` as tiles arrive.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One (N, C) slab: Ht × Wt tiles |
| Grid | Device grid, capped at NC slabs |
| Per-core work | `ceil(NC / num_cores)` slabs; each slab = Ht×Wt input tiles → Ht×Wt output tiles |
| Remainder | Last core(s) get fewer slabs; `split_work_to_cores` handles balanced distribution |

Where `NC = N * C`, `Ht = H / 32`, `Wt = W / 32`.

Each core processes its slabs sequentially. Within each slab, the 4-phase pipeline runs to completion before the next slab begins. The reader and compute alternate on `cb_input_tiles`: reader fills a slab, compute consumes it, reader fills the next.

## Circular Buffers

CB indices follow the convention: 0–7 input, 8–15 special, 16–23 output, 24–31 intermediate.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | tile_size(fp32) = 4096 | Ht×Wt | fp32 | Reader | Compute (max reduce, sub_exp) | Single-buffered per slab; filled by reader, consumed+popped by sub_exp |
| `cb_scaler_max` | 1 | tile_size(bf16) = 2048 | 1 | bf16 | Reader | Compute (max reduce) | Constant; prepared once at kernel start, reused across all slabs |
| `cb_scaler_sum` | 2 | tile_size(bf16) = 2048 | 1 | bf16 | Reader | Compute (sum reduce) | Constant; prepared once at kernel start, reused across all slabs |
| `cb_max` | 24 | tile_size(fp32) = 4096 | Ht (dim=-1) / Wt (dim=-2) | fp32 | Compute (max reduce) | Compute (sub_exp) | Intermediate; full block (sequential helper boundary) |
| `cb_exp` | 25 | tile_size(fp32) = 4096 | Ht×Wt | fp32 | Compute (sub_exp) | Compute (sum reduce, mul) | Intermediate; full block (sequential helper boundary) |
| `cb_recip_sum` | 26 | tile_size(fp32) = 4096 | Ht (dim=-1) / Wt (dim=-2) | fp32 | Compute (sum reduce) | Compute (mul) | Intermediate; full block (sequential helper boundary) |
| `cb_output_tiles` | 16 | tile_size(fp32) = 4096 | 2 | fp32 | Compute (mul) | Writer | Double-buffered; streaming between compute and writer |

**Sizing rationale:**

- `cb_input_tiles` (Ht×Wt pages): The max reduce uses `WaitUpfrontNoPop`, which calls `wait_front(Ht×Wt)` — the CB must hold the entire slab. Single-buffered to minimize L1 pressure; reader and compute alternate per slab.
- `cb_scaler_max`, `cb_scaler_sum` (1 page each): The reduce helper calls `scaler_dfb.wait_front(1)` and never pops — the tile persists for the kernel lifetime. Bfloat16 format per the reduce scaler convention.
- `cb_max`, `cb_recip_sum` (Ht or Wt pages): Sequential helper intermediates — both phases own all TRISCs, so the full block must be present before the consumer phase begins. HeldBulk lifecycle on the consumer side waits for all tiles upfront without popping.
- `cb_exp` (Ht×Wt pages): The sum reduce uses `WaitUpfrontNoPop`, requiring all exp tiles to be present. The mul phase subsequently reads them via Bulk lifecycle (pops at end). Full block required.
- `cb_output_tiles` (2 pages): Streaming output — the mul's PackTile uses `OutputLifecycle::Streaming` (per-tile reserve+push), and the writer drains one tile at a time. Double-buffered for reader-writer pipelining.

**L1 budget per core (single slab):** `2 × Ht × Wt × 4096 + (Ht or Wt) × 2 × 4096 + 2 × 2048 + 2 × 4096` bytes. For (1,1,128,512): ~560 KB. For (1,1,32,4096): ~1.07 MB. Exceeds L1 for very wide tensors (Wt > 128) — see Key Risks.

## API Mapping

Every mechanism has an exact file:line reference. All compute phases use helpers — no raw-API fallbacks.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|----------------------|--------------------------|---------------------------|--------------|
| Boot init | helper | `compute_kernel_hw_startup()` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:71` | `<SrcOrder::Regular>`, args: `(cb_input_tiles, cb_scaler_max, cb_max)` | — | — | Called once at kernel start before any helper. Configures SrcA=input, SrcB=scaler, Pack=intermediate. |
| Scaler prep (max) | helper | `prepare_reduce_scaler<>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:66` | `<cb_scaler_max, PoolType::MAX, ReduceDim::REDUCE_ROW>` (dim=-1) or `<cb_scaler_max, PoolType::MAX, ReduceDim::REDUCE_COL>` (dim=-2) | — | `cb_scaler_max` | Called once by reader at kernel start. Manages own reserve_back(1)+push_back(1). Scaler value = 1.0f. Format deduced from DFB (bf16). |
| Scaler prep (sum) | helper | `prepare_reduce_scaler<>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:66` | `<cb_scaler_sum, PoolType::SUM, ReduceDim::REDUCE_ROW>` (dim=-1) or `<cb_scaler_sum, PoolType::SUM, ReduceDim::REDUCE_COL>` (dim=-2) | — | `cb_scaler_sum` | Called once by reader at kernel start. Manages own reserve_back(1)+push_back(1). Scaler value = 1.0f. For SUM+REDUCE_ROW: col-0 fill (matmul path). For SUM+REDUCE_COL: row-0 fill. |
| 1. Max reduce (dim=-1) | helper | `reduce<>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:421` | `<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_input_tiles, cb_scaler_max, cb_max, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, NoAccumulation, NoOp>` | `cb_input_tiles` (Ht×Wt tiles, waited, NOT popped), `cb_scaler_max` (1 tile) | `cb_max` (Ht tiles) | `ReduceInputBlockShape::of(Ht, Wt, 1)`, `ReduceInputMemoryLayout::contiguous()`. Helper manages all CB ops: bulk wait_front(Ht×Wt), bulk reserve_back(Ht), per-row pack, bulk push_back(Ht). Input NOT popped — tiles persist for Phase 2. |
| 1. Max reduce (dim=-2) | helper | `reduce<>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:421` | `<PoolType::MAX, ReduceDim::REDUCE_COL, cb_input_tiles, cb_scaler_max, cb_max, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, NoAccumulation, NoOp>` | `cb_input_tiles` (Ht×Wt tiles, waited, NOT popped), `cb_scaler_max` (1 tile) | `cb_max` (Wt tiles) | `ReduceInputBlockShape::of(Ht, Wt, 1)`, `ReduceInputMemoryLayout::contiguous()`. REDUCE_COL uses chunk_size=DEST_AUTO_LIMIT (4 for fp32). Helper manages all CB ops. Input NOT popped. |
| 2. Sub+Exp (dim=-1) | helper | `eltwise_chain()` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:577` | `EltwiseShape::grid(Ht, Wt)` + chain elements (see below) | `cb_input_tiles` (Ht×Wt, Block, Bulk — pops at end), `cb_max` (Ht, Col, HeldBulk — no pop) | `cb_exp` (Ht×Wt, Streaming) | Chain: `BinaryFpu<cb_input_tiles, cb_max, BinaryFpuOp::Sub, BroadcastDim::Col, InputLifecycle::Bulk, InputLifecycle::HeldBulk, BinaryDataFormatReconfig::Input, Dst::D0, OperandKind::Block, OperandKind::Col>{}` → `Exp<>{}` → `PackTile<cb_exp, OutputLifecycle::Streaming>{}`. Chain owns dst-sync, CB lifecycle, dtype reconfig. |
| 2. Sub+Exp (dim=-2) | helper | `eltwise_chain()` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:577` | `EltwiseShape::grid(Ht, Wt)` + chain elements (see below) | `cb_input_tiles` (Ht×Wt, Block, Bulk — pops at end), `cb_max` (Wt, Row, HeldBulk — no pop) | `cb_exp` (Ht×Wt, Streaming) | Chain: `BinaryFpu<cb_input_tiles, cb_max, BinaryFpuOp::Sub, BroadcastDim::Row, InputLifecycle::Bulk, InputLifecycle::HeldBulk, BinaryDataFormatReconfig::Input, Dst::D0, OperandKind::Block, OperandKind::Row>{}` → `Exp<>{}` → `PackTile<cb_exp, OutputLifecycle::Streaming>{}`. BroadcastDim::Row broadcasts max across H rows. |
| Pop cb_max | raw_api | `DataflowBuffer::pop_front()` | — | `pop_front(Ht)` (dim=-1) or `pop_front(Wt)` (dim=-2) | `cb_max` | — | Explicit pop after sub_exp to free intermediate for next slab. HeldBulk left these unpopped. |
| 3. Sum+Recip (dim=-1) | helper | `reduce<>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:421` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_exp, cb_scaler_sum, cb_recip_sum, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, NoAccumulation, recip_lambda>` | `cb_exp` (Ht×Wt, waited, NOT popped), `cb_scaler_sum` (1 tile) | `cb_recip_sum` (Ht tiles, each = 1/sum) | `ReduceInputBlockShape::of(Ht, Wt, 1)`, `ReduceInputMemoryLayout::contiguous()`. PostReduceOp = `[](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); }` (file:line for recip_tile: `tt_metal/hw/inc/api/compute/eltwise_unary.h`). SUM+REDUCE_ROW uses matmul path (col-0 scaler). Exp tiles NOT popped — persist for Phase 4. |
| 3. Sum+Recip (dim=-2) | helper | `reduce<>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:421` | `<PoolType::SUM, ReduceDim::REDUCE_COL, cb_exp, cb_scaler_sum, cb_recip_sum, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, NoAccumulation, recip_lambda>` | `cb_exp` (Ht×Wt, waited, NOT popped), `cb_scaler_sum` (1 tile) | `cb_recip_sum` (Wt tiles, each = 1/sum) | Same PostReduceOp lambda. SUM+REDUCE_COL uses reduce_tile LLK (row-0 scaler). Exp tiles NOT popped. |
| 4. Mul (dim=-1) | helper | `mul<>()` | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp:82` | `<cb_exp, cb_recip_sum, cb_output_tiles, BroadcastDim::Col, InputLifecycle::Bulk, InputLifecycle::HeldBulk, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block, OperandKind::Col>` | `cb_exp` (Ht×Wt, Block, Bulk — pops at end), `cb_recip_sum` (Ht, Col, HeldBulk — no pop) | `cb_output_tiles` (Ht×Wt, Streaming) | `EltwiseShape::grid(Ht, Wt)`. Convenience wrapper forwards to `eltwise_chain` with `BinaryFpu(Mul, Col) + PackTile`. |
| 4. Mul (dim=-2) | helper | `mul<>()` | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp:82` | `<cb_exp, cb_recip_sum, cb_output_tiles, BroadcastDim::Row, InputLifecycle::Bulk, InputLifecycle::HeldBulk, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block, OperandKind::Row>` | `cb_exp` (Ht×Wt, Block, Bulk — pops at end), `cb_recip_sum` (Wt, Row, HeldBulk — no pop) | `cb_output_tiles` (Ht×Wt, Streaming) | `EltwiseShape::grid(Ht, Wt)`. BroadcastDim::Row broadcasts recip-sum across H rows. |
| Pop cb_recip_sum | raw_api | `DataflowBuffer::pop_front()` | — | `pop_front(Ht)` (dim=-1) or `pop_front(Wt)` (dim=-2) | `cb_recip_sum` | — | Explicit pop after mul to free intermediate for next slab. HeldBulk left these unpopped. |

**No raw-API fallbacks for compute phases.** The two `pop_front` calls are CB maintenance operations between helper phases (freeing intermediates that HeldBulk intentionally left unpopped), not compute phases themselves.

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | Boot init | `compute_kernel_hw_startup` | — | — | Hardware configured for reduce ops |
| 1 | Max reduce | `reduce<MAX, REDUCE_ROW/COL, WaitUpfrontNoPop>` | `cb_input_tiles`: Ht×Wt tiles (waited, NOT popped); `cb_scaler_max`: 1 tile (waited, NOT popped) | `cb_max`: Ht (dim=-1) or Wt (dim=-2) tiles (bulk reserved + pushed) | `cb_input_tiles` retains Ht×Wt tiles; `cb_max` has Ht/Wt tiles; `cb_scaler_max` still has 1 tile |
| 2 | Sub+Exp (fused) | `eltwise_chain` with `BinaryFpu(Sub) + Exp + PackTile` | `cb_input_tiles`: Ht×Wt (Bulk → waited upfront, popped at end); `cb_max`: Ht/Wt (HeldBulk → waited upfront, NOT popped) | `cb_exp`: Ht×Wt tiles (Streaming → per-tile push) | `cb_input_tiles` EMPTY (popped by Bulk); `cb_max` still has Ht/Wt tiles (needs explicit pop); `cb_exp` has Ht×Wt tiles |
| 2a | Pop max | raw `pop_front` | `cb_max`: pop Ht/Wt tiles | — | `cb_max` EMPTY |
| 3 | Sum+Recip (fused) | `reduce<SUM, REDUCE_ROW/COL, WaitUpfrontNoPop, PostReduceOp=recip>` | `cb_exp`: Ht×Wt (waited, NOT popped); `cb_scaler_sum`: 1 tile (waited, NOT popped) | `cb_recip_sum`: Ht/Wt tiles (each = 1/sum, bulk reserved + pushed) | `cb_exp` retains Ht×Wt tiles; `cb_recip_sum` has Ht/Wt tiles; `cb_scaler_sum` still has 1 tile |
| 4 | Mul (broadcast) | `mul<>()` convenience wrapper | `cb_exp`: Ht×Wt (Bulk → waited upfront, popped at end); `cb_recip_sum`: Ht/Wt (HeldBulk → waited upfront, NOT popped) | `cb_output_tiles`: Ht×Wt tiles (Streaming → per-tile push) | `cb_exp` EMPTY (popped by Bulk); `cb_recip_sum` still has Ht/Wt tiles (needs explicit pop); `cb_output_tiles` has Ht×Wt tiles streaming to writer |
| 4a | Pop recip_sum | raw `pop_front` | `cb_recip_sum`: pop Ht/Wt tiles | — | `cb_recip_sum` EMPTY |

**Phase 1→2 contract:** Max reduce leaves input tiles in `cb_input_tiles` (WaitUpfrontNoPop). Sub_exp reads them via Bulk lifecycle (which pops at end). The max result in `cb_max` is read via HeldBulk (no pop) — tiles must be explicitly freed after sub_exp completes.

**Phase 3→4 contract:** Sum reduce leaves exp tiles in `cb_exp` (WaitUpfrontNoPop). Mul reads them via Bulk (pops at end). The recip-sum in `cb_recip_sum` is read via HeldBulk (no pop) — tiles must be explicitly freed after mul completes.

**DEST budget per phase (fp32, fp32_dest_acc_en=True → DEST_AUTO_LIMIT = 4):**

| Phase | DEST usage | Limit check |
|-------|-----------|-------------|
| Max reduce (REDUCE_ROW) | 1 tile per row | ✓ 1 ≤ 4 |
| Max reduce (REDUCE_COL) | chunk_size = 4 tiles per chunk | ✓ 4 ≤ 4 |
| Sub+Exp chain | 1 DEST slot (D0) | ✓ 1 ≤ 4 |
| Sum reduce (REDUCE_ROW) | 1 tile per row | ✓ 1 ≤ 4 |
| Sum reduce (REDUCE_COL) | chunk_size = 4 tiles per chunk | ✓ 4 ≤ 4 |
| Mul chain | 1 DEST slot (D0) | ✓ 1 ≤ 4 |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| Sub+Exp (dim=-1) | Sub | `cb_input_tiles`: full Ht×Wt block (OperandKind::Block) | `cb_max`: Ht tiles, one per row (OperandKind::Col) — REDUCE_ROW output is column-shaped | BroadcastDim::Col (broadcast max across W columns) |
| Sub+Exp (dim=-2) | Sub | `cb_input_tiles`: full Ht×Wt block (OperandKind::Block) | `cb_max`: Wt tiles, one per column (OperandKind::Row) — REDUCE_COL output is row-shaped | BroadcastDim::Row (broadcast max across H rows) |
| Mul (dim=-1) | Mul | `cb_exp`: full Ht×Wt block (OperandKind::Block) | `cb_recip_sum`: Ht tiles, one per row (OperandKind::Col) | BroadcastDim::Col (broadcast recip-sum across W columns) |
| Mul (dim=-2) | Mul | `cb_exp`: full Ht×Wt block (OperandKind::Block) | `cb_recip_sum`: Wt tiles, one per column (OperandKind::Row) | BroadcastDim::Row (broadcast recip-sum across H rows) |

**BroadcastDim naming convention** (from `eltwise_chain.hpp:418-424`): The dim names *which axis is broadcast*, not which was reduced. A REDUCE_ROW result is column-shaped (Ht×1) and broadcasts back across columns via `BroadcastDim::Col`. A REDUCE_COL result is row-shaped (1×Wt) and broadcasts down rows via `BroadcastDim::Row`.

**OperandKind consistency:** For `BroadcastDim::Col`, operand B uses `OperandKind::Col` — `idx<Col>(i_flat, ht, wt) = ht` (reads one tile per row, re-read across all wt). For `BroadcastDim::Row`, operand B uses `OperandKind::Row` — `idx<Row>(i_flat, ht, wt) = wt` (reads one tile per column, re-read across all ht). Verified at `eltwise_chain.inl:292-297`.

**Lifecycle compatibility** (verified at `eltwise_chain.hpp:270-289`):
- `OperandKind::Block` + `InputLifecycle::Bulk` — legal (Bulk in Block's allowed set). Bulk waits Ht×Wt upfront, pops at end.
- `OperandKind::Col` + `InputLifecycle::HeldBulk` — legal (HeldBulk in Col's allowed set). HeldBulk waits Ht upfront, no pop.
- `OperandKind::Row` + `InputLifecycle::HeldBulk` — legal (HeldBulk in Row's allowed set). HeldBulk waits Wt upfront, no pop.
- `valid_policy_mode_v` check passes for all combinations (bcast modes with non-streaming policies).

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | EltwiseShape |
|-------------|----------------|--------------------|--------------|-----------------------|--------------|
| -1 (W) | REDUCE_ROW | Col0: Ht tiles, one per tile-row | BroadcastDim::Col | `ReduceInputBlockShape::of(Ht, Wt, 1)` | `EltwiseShape::grid(Ht, Wt)` |
| -2 (H) | REDUCE_COL | Row0: Wt tiles, one per tile-column | BroadcastDim::Row | `ReduceInputBlockShape::of(Ht, Wt, 1)` | `EltwiseShape::grid(Ht, Wt)` |

**ReduceDim selection:** `dim=-1` reduces along W → `REDUCE_ROW` (each row of Wt tiles collapses to 1 tile). `dim=-2` reduces along H → `REDUCE_COL` (each column of Ht tiles collapses to 1 tile). The `dim` value is passed as a compile-time arg and canonicalized (`dim if dim < 0 else dim - ndim`) before the support check, so positive aliases work (rank-4 `dim=3` ≡ `-1`, `dim=2` ≡ `-2`).

**Scaler fill layout:**
- MAX + REDUCE_ROW: `reduce_uses_matmul<MAX, REDUCE_ROW>()` = false → row-0 fill (reduce_tile LLK). Verified at `reduce_helpers_common.hpp:15-19`.
- MAX + REDUCE_COL: `reduce_uses_matmul<MAX, REDUCE_COL>()` = false → row-0 fill.
- SUM + REDUCE_ROW: `reduce_uses_matmul<SUM, REDUCE_ROW>()` = true → col-0 fill (matmul path).
- SUM + REDUCE_COL: `reduce_uses_matmul<SUM, REDUCE_COL>()` = false → row-0 fill.

The `prepare_reduce_scaler` dataflow helper (`reduce_helpers_dataflow.hpp:66`) auto-selects the fill layout based on `<PoolType, ReduceDim>` template args, matching the compute-side dispatch.

## Key Risks and Gotchas

1. **L1 capacity for wide tensors:** `cb_input_tiles` and `cb_exp` each require Ht×Wt pages (full slab). For Wt > 128 (e.g. (1,1,32,4096): Wt=128), total L1 exceeds 1 MB per core. Work distribution across more cores mitigates this (each core handles fewer slabs), but per-slab L1 is the bottleneck. Future K-blocking along the reduce dimension would allow sub-slab processing.

2. **Intermediate CBs must hold full blocks:** `cb_max` (Ht/Wt pages), `cb_exp` (Ht×Wt pages), and `cb_recip_sum` (Ht/Wt pages) are sequential-helper intermediates. Both producer and consumer run on TRISC (same processor), so no pipelining is possible — the full block must be present before the consumer begins. These CBs are sized to the full block, not double-buffered.

3. **HeldBulk intermediates need explicit pop:** The `HeldBulk` lifecycle on operand B (cb_max in Phase 2, cb_recip_sum in Phase 4) intentionally does NOT pop tiles — they persist for potential reuse. Since softmax doesn't reuse them across slabs, the compute kernel must explicitly `pop_front(Ht)` or `pop_front(Wt)` after the consuming phase to free the CB for the next slab. Missing this pop causes `cb_reserve_back` to deadlock on the next slab.

4. **fp32 DEST limit (4 tiles):** With `fp32_dest_acc_en=True`, `DEST_AUTO_LIMIT = 4` (half-sync). The REDUCE_COL path chunks by `DEST_AUTO_LIMIT` columns at a time — for Wt > 4, multiple chunks are processed. All other phases use ≤ 1 DEST slot. No phase exceeds the limit.

5. **Scaler CB format is bfloat16:** Even though input/output are float32, the scaler CBs (`cb_scaler_max`, `cb_scaler_sum`) are bfloat16. The reduce LLK handles mixed-format (fp32 input, bf16 scaler) internally. The `prepare_reduce_scaler` function deduces format from the DFB and accepts Float16_b or Float32.

6. **Pool-type-aware scaler API:** Both scaler preparations use the 3-template-arg overload `prepare_reduce_scaler<cb, PoolType, ReduceDim>(1.0f)`, NOT the legacy 1-template-arg `prepare_reduce_scaler<cb>(1.0f)`. Different PoolType/ReduceDim combinations require different tile fill patterns (row-0 vs col-0). Verified at `reduce_helpers_dataflow.hpp:66`.

7. **CB sync invariant (push = wait):** Every CB's producer push count equals consumer wait count:
   - `cb_input_tiles`: reader pushes Ht×Wt per slab; max reduce waits Ht×Wt (WaitUpfrontNoPop). ✓
   - `cb_scaler_max`: reader pushes 1; max reduce waits 1. ✓ (never popped, reused)
   - `cb_scaler_sum`: reader pushes 1; sum reduce waits 1. ✓
   - `cb_max`: max reduce pushes Ht/Wt; sub_exp waits Ht/Wt (HeldBulk upfront). ✓
   - `cb_exp`: sub_exp pushes Ht×Wt (Streaming per-tile); sum reduce waits Ht×Wt (WaitUpfrontNoPop). ✓
   - `cb_recip_sum`: sum reduce pushes Ht/Wt; mul waits Ht/Wt (HeldBulk upfront). ✓
   - `cb_output_tiles`: mul pushes Ht×Wt (Streaming per-tile); writer waits per-tile. ✓

8. **Compile-time dim dispatch:** `dim` must be a compile-time arg for both reader and compute kernels because `PoolType`, `ReduceDim`, `BroadcastDim`, and `OperandKind` are template parameters. The kernels use `if constexpr (DIM == -1)` to select REDUCE_ROW/Col and BroadcastDim::Col/Row.

9. **Exp<> template defaults:** The fused sub_exp chain uses `Exp<>{}` with default template params (`Approx::Exact, Approx::Fast, Dst::D0`). If PCC doesn't meet 0.999 for float32, the implementer may switch to `Exp<Approx::Exact, Approx::Exact>` for a more precise exp.

## Structural Impossibilities

The existing `eval/golden_tests/softmax/feature_spec.py` already defines `INVALID` as `[{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}]`. No additional structural impossibilities were identified for softmax beyond this canonical block-format vs row-major case. The `fp32 + fp32_dest_acc_en=False` combination is legal but lossy — it belongs in `EXCLUSIONS` (op-side xfail), not `INVALID`.

# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (normalization) |
| Goal | RMSNorm: normalize each row by the root-mean-square along the last dimension, with an optional learnable scale `gamma` and additive numerical stabilizer `epsilon`. |
| Math | `output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]` (gamma defaults to 1.0 when not supplied) |
| Mode | Derivative |
| References | `ttnn/ttnn/operations/toy_variance/` (two-pass streaming reduce), `ttnn/ttnn/operations/toy_tilize_untilize/` (in-kernel tilize/untilize), `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; last-dim ≥ 1; layout ∈ {ROW_MAJOR, TILE}; dtype ∈ {bfloat16, float32}; for TILE_LAYOUT, H % 32 == 0 and W % 32 == 0 | — | runtime (buffer + accessor) |
| `gamma` | `Optional[ttnn.Tensor]` | no | `None` or shape `(1, 1, 1, W)` matching last dim of input; ROW_MAJOR_LAYOUT only; same dtype as input or float32 | `None` | runtime (buffer + accessor) + CT `HAS_GAMMA` flag |
| `epsilon` | `float` | no (kw-only) | finite, ≥ 0 | `1e-6` | runtime (bit-cast to `uint32_t`, passed as compute kernel RT arg) |

Python-side validation in `rms_norm()`:

| Check | Error |
|-------|-------|
| `len(input_tensor.shape) < 2` | `ValueError("rms_norm: input must have at least 2 dimensions")` |
| `gamma is not None and gamma.shape[-1] != input_tensor.shape[-1]` | `ValueError("rms_norm: gamma last dim must match input last dim")` |
| input layout is `TILE_LAYOUT` and `H % 32 != 0 or W % 32 != 0` | `RuntimeError("rms_norm: TILE_LAYOUT input requires H and W divisible by 32")` |
| input dtype not in `{bfloat16, float32}` | `RuntimeError("rms_norm: only bfloat16 and float32 inputs are supported")` |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)` with rank ≥ 2; flatten leading dims into NC for kernel purposes (kernel sees `(NC, H, W)`) |
| Dtype | `bfloat16` or `float32` |
| Layout | `ROW_MAJOR_LAYOUT` or `TILE_LAYOUT` (CT-gated) |
| Memory | DRAM interleaved |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)` with W matching input last dim |
| Dtype | `bfloat16` or `float32` (independent of input dtype) |
| Layout | `ROW_MAJOR_LAYOUT` |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | identical to input |
| Layout | identical to input |
| Memory | DRAM interleaved (`ttnn.DRAM_MEMORY_CONFIG`) |

## Dataflow Strategy

End-to-end data path per Tensix core (DRAM → reader → compute → writer → DRAM):

| Phase | Producer (RISC) | Format | Carrier CB | Consumer (RISC) |
|-------|-----------------|--------|------------|-----------------|
| read input | reader (NCRISC) | RM sticks if RM input, tiles if TILE input | `cb_input_raw_rm` (RM) or `cb_input_tiles` (TILE) | compute (TRISC) |
| in-kernel tilize input (RM input only) | compute | tiles | `cb_input_tiles` | compute |
| read gamma (per row chunk) | reader | RM sticks (single stick) | `cb_gamma_rm` | compute |
| in-kernel tilize gamma (per row chunk) | compute | tiles (Wt tiles; only row 0 carries valid data per tile, broadcast-row semantics use only row 0) | `cb_gamma_tiled` | compute |
| reduce mean(x²) → rsqrt(mean+eps) | compute | tile | `cb_mean_sq` → in-place transform | compute |
| multiply x · rsqrt (· gamma) | compute | tiles | `cb_output_tiles` (TILE output) or `cb_x_norm` then `cb_output_tiles` (gamma path) | compute / untilize |
| untilize output (RM output only) | compute | RM sticks | `cb_output_rm` | writer |
| write output | writer (BRISC) | RM sticks or tiles | — | DRAM |

No Tensix-to-Tensix communication is required: each core processes an independent set of row chunks. The reduce is per-row (along W), wholly local to a core. All synchronization is intra-Tensix via CBs.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one row chunk = 32 rows (1 tile-height block, Wt tiles wide). For RM input with partial-H, the last chunk has `H % 32` valid rows. |
| Total work | `Ht = ceil(H / 32) * NC` (NC = product of leading dims) |
| Grid | `device.compute_with_storage_grid_size()` |
| Per-core split | `ttnn.split_work_to_cores(grid_size, Ht)` → `(num_cores, all_cores, core_group_1, core_group_2, chunks_per_core_g1, chunks_per_core_g2)` |
| Per-core work | each core processes a contiguous range of row chunks `[start_chunk, start_chunk + chunks_for_this_core)` |
| Remainder | handled by the two core groups returned by `split_work_to_cores` (group 1 gets `chunks_per_core_g1`, group 2 gets `chunks_per_core_g2 = chunks_per_core_g1 - 1`) |

The reader and writer each receive a `start_chunk` and a `num_chunks` runtime arg. The compute kernel receives only `num_chunks` (it has no DRAM addressing — all data arrives via CBs).

## Circular Buffers

Compile-time conditional CBs are gated by `INPUT_IS_RM`, `OUTPUT_IS_RM`, and `HAS_GAMMA` (boolean compile-time args). Sizes use:
- `tile = ttnn.tile_size(dtype)` (per-CB dtype)
- `Wt = ceil(W / 32)`
- `padded_row_bytes = ceil(W * element_size / (32 * element_size)) * (32 * element_size)` (row aligned to a tile-row width)
- `padded_gamma_row_bytes = ceil(W * gamma_element_size / (32 * gamma_element_size)) * (32 * gamma_element_size)`

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_raw_rm` | 0 | `padded_row_bytes` (one RM stick) | `2 * 32` (double-buffer one chunk of 32 sticks) | input dtype | reader | compute (tilize) | one row chunk; only present when `INPUT_IS_RM` |
| `cb_input_tiles` | 1 | `tile_size(input_dtype)` | `2 * Wt` (double-buffer for cross-row-chunk pipelining) | input dtype | reader (TILE input) **or** compute tilize (RM input) | compute pass 1 stage A (NoWaitNoPop) + pass 2 stage D (WaitAndPop) | held Wt tiles across stages A → D within a row chunk, then popped Wt by stage D |
| `cb_gamma_rm` | 2 | `padded_gamma_row_bytes` (1 stick) | `2` (double-buffer one row chunk's gamma read) | gamma dtype | reader | compute (tilize gamma) | one row chunk; only present when `HAS_GAMMA` |
| `cb_gamma_tiled` | 3 | `tile_size(gamma_dtype)` | `2 * Wt` (double-buffer Wt tiles per row chunk) | gamma dtype | compute (tilize gamma) | compute pass 2 stage E (WaitAndPop, streams Wt) | one row chunk (re-tilized each chunk); only present when `HAS_GAMMA` |
| `cb_scaler` | 4 | `tile_size(bfloat16)` (reduce scaler is always bf16) | `1` (full scaler only) or `2` (full + partial when `HAS_PARTIAL_W`) | `bfloat16` | reader (one-time, at kernel start) | compute reduce (held persistently for all reductions) | whole kernel; popped once at very end |
| `cb_output_tiles` | 16 | `tile_size(output_dtype)` | `2 * Wt` (must hold Wt for untilize consumption when RM output; double-buffered for TILE-output writer pipelining) | output dtype (= input dtype) | compute pass 2 stage E (gamma) or stage D (no gamma) | writer (TILE output) **or** compute untilize (RM output) | per row chunk |
| `cb_output_rm` | 17 | `padded_row_bytes` (one RM stick, output dtype) | `2 * 32` (double-buffer one chunk of 32 sticks) | output dtype | compute untilize | writer | per row chunk; only present when `OUTPUT_IS_RM` |
| `cb_x_sq` | 24 | `tile_size(input_dtype)` | `Wt` (must hold full Wt for sequential helper consumption by reduce — stage A and stage B run sequentially on the same row chunk, no overlap) | input dtype | compute pass 1 stage A | compute pass 1 stage B reduce (BulkWaitBulkPop) | per row chunk; pushed Wt and popped Wt in same row chunk |
| `cb_mean_sq` | 25 | `tile_size(input_dtype)` | `2` (1 active tile + 1 spare; `transform_in_place` pops-before-reserve so 1 is sufficient, 2 is for safety) | input dtype | compute reduce → in-place `transform_in_place` overwrites with `rsqrt(mean_sq + eps)` | compute pass 2 stage D (WaitNoPop, waited once per pass-2 iter) | per row chunk; popped 1 at end of row chunk |
| `cb_x_norm` | 26 | `tile_size(input_dtype)` | `Wt` (must hold full Wt; sequential consumption by stage E in same row chunk) | input dtype | compute pass 2 stage D | compute pass 2 stage E (WaitAndPop) | per row chunk; only present when `HAS_GAMMA` |

CB sync invariant (producer push count == consumer wait count) verification:

| CB | Per-row-chunk push count | Per-row-chunk consumer wait count |
|----|--------------------------|-----------------------------------|
| `cb_input_raw_rm` (RM in) | reader pushes 32 sticks | tilize waits 32 sticks per `tilize<Wt, cb_input_raw_rm, cb_input_tiles>(1, 32)` |
| `cb_input_tiles` | Wt (either from reader TILE-input path or from tilize RM-input path) | stage A external `cb_wait_front(Wt)` + chain NoWaitNoPop (no pop); stage D chain WaitAndPop (pops Wt) → net pops match net pushes |
| `cb_gamma_rm` (HAS_GAMMA) | 1 stick | tilize waits 1 stick per row chunk |
| `cb_gamma_tiled` (HAS_GAMMA) | Wt (compute tilize) | stage E chain WaitAndPop streams Wt |
| `cb_scaler` | 1 or 2 (reader, ONCE at kernel start) | reduce internally waits and does NOT pop; final pop matches |
| `cb_x_sq` | Wt (stage A) | stage B reduce BulkWaitBulkPop pops Wt |
| `cb_mean_sq` | 1 (reduce) + 1 (transform_in_place re-push) − 1 (transform_in_place pop) = net +1 | stage D WaitNoPop waits 1 (never pops); end-of-chunk explicit pop 1 |
| `cb_x_norm` (HAS_GAMMA) | Wt (stage D) | stage E WaitAndPop pops Wt |
| `cb_output_tiles` | Wt (stage E or stage D) | writer pops Wt (TILE out) **or** compute untilize pops Wt (RM out) |
| `cb_output_rm` (RM out) | 32 sticks (untilize) | writer pops 32 sticks |

## API Mapping

Every mechanism used in the compute, reader, and writer kernels has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| **Boot — compute kernel hardware init** | raw API | `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:32` (D5 contract) | `(cb_input_tiles, cb_input_tiles, cb_x_sq)` — picks the first chain's (CbA, CbB, CbOut) triple | — | — | Must be the FIRST statement of `MAIN()` per `compute_kernel_hw_startup.h:26-30`; chain helpers do NOT call it internally |
| **Reader — scaler tile setup** (HAS_PARTIAL_W false) | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / W)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:65-67` | `cb_id=cb_scaler`, `pool_type=SUM`, `reduce_dim=REDUCE_ROW`, scaler value `1/W` for SUM-as-mean | — | `cb_scaler` | Pool-type-aware overload (system rule); REDUCE_ROW + SUM uses matmul-path col-0 fill |
| **Reader — partial scaler setup** (HAS_PARTIAL_W true) | helper | `dataflow_kernel_lib::prepare_partial_reduce_scalers<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, partial_w>(1.0f / W)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:136-142` | `partial_positions=partial_w=W % 32` | — | `cb_scaler` (2 tiles: full + partial) | Pair with `ReducePartialScaler::last_tile_at(1)` on the compute side |
| **Reader — input tile stream** (INPUT_IS_RM false) | raw API | `noc_async_read_tile(...)` over `TensorAccessor`, with per-tile `cb_reserve_back(cb_input_tiles, 1)` / `cb_push_back(cb_input_tiles, 1)` | `ttnn/cpp/ttnn/operations/toy_reduce_partial/kernels/reader.cpp:43-51` (canonical pattern) | per-tile loop streaming Wt tiles per row chunk (both passes; reader streams the SAME Wt tiles twice — once for pass 1, once for pass 2) | DRAM input | `cb_input_tiles` | TILE input only |
| **Reader — input stick stream** (INPUT_IS_RM true) | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_raw_rm, TilizeGranularity::TILE>(accessor, 32, row_bytes, start_page=chunk_first_row)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:77-79` | `cb_id=cb_input_raw_rm`, `granularity=TILE` (page=tile_size, reader fills 32 sticks per tile-page batch) | DRAM input | `cb_input_raw_rm` | Reader is invoked TWICE per row chunk (once per compute-pass), each invocation reads the same 32 sticks. **Helpers considered and rejected**: ROW granularity would be valid but TILE granularity keeps the input CB sized symmetric with the tilize compute call (`tilize<Wt, cb_input_raw_rm, cb_input_tiles>(1)` per `tilize_helpers.hpp:112`), simplifying CB sizing. |
| **Reader — gamma stick** (HAS_GAMMA true) | raw API | `noc_async_read` from `gamma_accessor.get_noc_addr(0)` (gamma has only 1 stick) into `cb_gamma_rm` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:77-79` (similar pattern; `read_sticks_for_tilize` would also work but iterates from `start_page`) | 1 stick of gamma data per row chunk push | DRAM gamma | `cb_gamma_rm` | **Helpers considered and rejected**: `read_sticks_for_tilize` with ROW granularity, file:line `tilize_helpers_dataflow.hpp:77`, but its iteration assumes `total_num_rows` distinct page ids starting from `start_page`. Gamma has only 1 logical page; iterating 32 times would read out-of-bounds pages. Raw `noc_async_read` from page 0 is the correct primitive (single-page, single-stick). |
| **Compute — tilize input** (INPUT_IS_RM true) | helper | `compute_kernel_lib::tilize<Wt, cb_input_raw_rm, cb_input_tiles, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::NoReconfigure>(1)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:144-153` | `block_width_tiles=Wt`, `num_blocks=1` per row chunk × 2 passes; symmetric mode (cb_input_raw_rm pages are tile-sized) | `cb_input_raw_rm` | `cb_input_tiles` | RM input only; called once per pass per row chunk |
| **Compute — tilize gamma** (HAS_GAMMA true) | helper | `compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma_tiled, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1, /*total_input_pages=*/1)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:144-153` (asymmetric form, `total_input_pages` arg) | asymmetric mode: 1 row of gamma → Wt tiles. Rows 1-31 of each output tile are uninitialized — broadcast-row consumer reads only row 0 so this is sound. | `cb_gamma_rm` | `cb_gamma_tiled` | Called once per row chunk |
| **Compute — pass 1 stage A: x → x²** | helper | `compute_kernel_lib::eltwise_chain(Wt, CopyTile<cb_input_tiles, Dst::D0, CopyTilePolicy::NoWaitNoPop, CbIndexMode::BlockIter>{}, Square<Dst::D0>{}, PackTile<cb_x_sq, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{})` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:738-739` (chain entry); `ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp:45-48` (Square) | NoWaitNoPop + BlockIter — caller pre-waits cb_input_tiles before the chain, no pop inside chain (input is reused in stage D) | `cb_input_tiles` (Wt indexed) | `cb_x_sq` (Wt streamed) | Caller calls `cb_wait_front(cb_input_tiles, Wt)` before this chain |
| **Compute — pass 1 stage B: mean(x²)** | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(cb_x_sq, cb_scaler, cb_mean_sq, ReduceInputBlockShape::row(Wt), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, NoOp{}, partial_scaler)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:400-415` | `reduce_type=SUM`, `reduce_dim=REDUCE_ROW`, `input_policy=BulkWaitBulkPop`, `partial_scaler = HAS_PARTIAL_W ? ReducePartialScaler::last_tile_at(1) : ReducePartialScaler::none()` (`reduce_helpers_compute.hpp:184-185`); shape `(1, Wt, 1)` | `cb_x_sq` (Wt tiles consumed and popped) + `cb_scaler` (held persistently) | `cb_mean_sq` (1 tile) | SUM + scaler=1/W ≡ mean. Partial scaler zeros padded-W contributions per `reduce_helpers_compute.hpp:158-186`. |
| **Compute — pass 1 stage C: add eps + rsqrt** | helper | `compute_kernel_lib::transform_in_place(cb_mean_sq, [&](uint32_t dst) { add_unary_tile_init(); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile<false>(dst); })` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:110-111` | Single-tile in-place transform; pops cb_mean_sq, applies lambda on DST[0], re-packs into cb_mean_sq | `cb_mean_sq` (1 tile in, 1 tile out — same CB) | `cb_mean_sq` (overwritten with `rsqrt(mean_sq + eps)`) | Helper bundles SRCA + packer reconfig per `streaming_reduce_helpers.hpp:106-109`. `eps_bits = std::bit_cast<uint32_t>(epsilon)` host-side, passed as compute kernel RT arg 0. |
| **Compute — pass 2 stage D (no gamma): x · rsqrt → out** | helper | `compute_kernel_lib::eltwise_chain(Wt, BinaryFpu<cb_input_tiles, cb_mean_sq, cb_output_tiles, BinaryFpuOp::Mul, BroadcastDim::Col, BinaryDataFormatReconfig::InputAndOutput, CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitNoPop, CbIndexMode::FirstTile, Dst::D0, CbIndexMode::FirstTile>{}, PackTile<cb_output_tiles, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{})` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:603-615` (BinaryFpu signature); `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl:519-546` (bcast path) | BroadcastDim::Col: rsqrt is the column-broadcast operand (`rsqrt[0]` replicated across columns of `x[t]`). A-side WaitAndPop streams x; B-side WaitNoPop holds rsqrt across all Wt iterations. | `cb_input_tiles` (Wt popped) + `cb_mean_sq` (1 tile, retained) | `cb_output_tiles` (Wt) | Only when `HAS_GAMMA` is false |
| **Compute — pass 2 stage D (gamma): x · rsqrt → norm** | helper | Same as above but `BinaryFpu<cb_input_tiles, cb_mean_sq, cb_x_norm, ...>{}` and `PackTile<cb_x_norm, ...>{}` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:603-615` | identical to no-gamma form except output CB is `cb_x_norm` (intermediate before gamma multiply) | `cb_input_tiles` (Wt popped) + `cb_mean_sq` (1 tile, retained) | `cb_x_norm` (Wt) | Only when `HAS_GAMMA` is true |
| **Compute — pass 2 stage E (gamma): norm · gamma → out** | helper | `compute_kernel_lib::eltwise_chain(Wt, BinaryFpu<cb_x_norm, cb_gamma_tiled, cb_output_tiles, BinaryFpuOp::Mul, BroadcastDim::Row, BinaryDataFormatReconfig::InputAndOutput, CopyTilePolicy::WaitAndPop, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, Dst::D0, CbIndexMode::FirstTile>{}, PackTile<cb_output_tiles, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{})` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:603-615`; `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl:519-546` | BroadcastDim::Row: gamma row 0 broadcast across all rows of x_norm. Both sides WaitAndPop + FirstTile — each iter pops one tile from x_norm and one from gamma, so over Wt iters, all Wt gamma tiles are streamed and consumed. | `cb_x_norm` (Wt popped) + `cb_gamma_tiled` (Wt popped) | `cb_output_tiles` (Wt) | Only when `HAS_GAMMA` is true |
| **Compute — untilize output** (OUTPUT_IS_RM true) | helper | `compute_kernel_lib::untilize<Wt, cb_output_tiles, cb_output_rm, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:132-140` | `block_width_tiles=Wt`, 1 block (one tile-height row chunk) | `cb_output_tiles` (Wt popped) | `cb_output_rm` (32 sticks pushed) | RM output only |
| **Writer — write tiles** (OUTPUT_IS_RM false) | raw API | `noc_async_write_tile(...)` over `TensorAccessor` with per-tile `cb_wait_front(cb_output_tiles, 1)` / `cb_pop_front(cb_output_tiles, 1)` | `ttnn/cpp/ttnn/operations/toy_reduce_partial` (canonical writer pattern); standard `noc_async_write_tile` in `api/dataflow/dataflow_api.h` | per-tile loop streaming Wt tiles per row chunk | `cb_output_tiles` | DRAM output | TILE output only |
| **Writer — write sticks** (OUTPUT_IS_RM true) | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(accessor, total_num_rows_for_core, row_bytes, start_page=chunk_first_row)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:107-109` | Handles partial-H (writes only `total_num_rows` valid sticks) and partial-W (writes only `row_bytes` per stick) | `cb_output_rm` | DRAM output | RM output only |

### Helpers considered and rejected — explicit raw-API justifications

There are TWO raw-API call sites in this design, both in dataflow kernels (no compute kernel raw-API use; every compute phase is a helper):

1. **Tile reader / writer** (TILE input/output paths): `noc_async_read_tile` + per-tile CB reserve/push and `noc_async_write_tile` + per-tile CB wait/pop. There is no `read_tiles_*` aggregator helper in `ttnn/cpp/ttnn/kernel_lib/` that covers per-tile streaming of tiles directly into a CB (the existing `tilize_helpers_dataflow.hpp:77-109` family covers RM-stick reads for tilize and untilize output writes; reading already-tiled tensors goes through the standard `noc_async_read_tile` + `TensorAccessor` pattern, which is the canonical primitive). Verified by reading `tilize_helpers_dataflow.hpp` end-to-end — the API surface is sticks-for-tilize and sticks-after-untilize only.

2. **Gamma single-stick read**: `noc_async_read` of page 0 into `cb_gamma_rm`. `dataflow_kernel_lib::read_sticks_for_tilize` (`tilize_helpers_dataflow.hpp:77-79`) iterates `total_num_rows` pages starting from `start_page`; gamma has a single page so iterating ≥ 2 reads would address out-of-bounds pages, and iterating 1 row works but requires `total_input_pages=1` on the tilize side. Either approach is acceptable — for clarity the implementer may use `read_sticks_for_tilize<cb_gamma_rm, TilizeGranularity::ROW>(gamma_accessor, 1, padded_gamma_row_bytes)` here too; that is the path the design selects when feasible. Marked "raw API" in the table only as a defensive note.

For every compute-kernel phase, a helper exists and is used. There is **no raw-API compute phase** in this design.

## Compute Phases

Sequential phase execution per row chunk. CB state column shows which CBs still hold data after each phase.

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 (HAS_GAMMA only, per row chunk) | tilize gamma stick → Wt tiles | yes — `tilize` | `cb_gamma_rm` (1 stick) | `cb_gamma_tiled` (Wt tiles) | `cb_gamma_tiled`: Wt (holds gamma, awaits stage E) |
| 1a (INPUT_IS_RM only) | tilize input sticks → Wt tiles (pass 1 push) | yes — `tilize` | `cb_input_raw_rm` (32 sticks) | `cb_input_tiles` (Wt tiles) | `cb_input_tiles`: Wt |
| 1b (always) | external `cb_wait_front(cb_input_tiles, Wt)` so stage A can index Wt tiles | raw cb api | — | — | `cb_input_tiles`: Wt (still queued; nothing popped) |
| 2 | pass 1 stage A: square each tile | yes — `eltwise_chain` with `Square` SFPU | `cb_input_tiles` (Wt, NoWaitNoPop) | `cb_x_sq` (Wt) | `cb_input_tiles`: Wt held; `cb_x_sq`: Wt available |
| 3 | pass 1 stage B: reduce SUM/REDUCE_ROW with scaler=1/W | yes — `reduce<>` | `cb_x_sq` (Wt, BulkWaitBulkPop) + `cb_scaler` (held) | `cb_mean_sq` (1 tile) | `cb_input_tiles`: Wt held; `cb_x_sq`: 0; `cb_mean_sq`: 1 |
| 4 | stage C: in-place `cb_mean_sq` → `add_unary_tile(eps_bits)` → `rsqrt_tile<false>` | yes — `transform_in_place` | `cb_mean_sq` (1 tile in, popped + re-packed) | `cb_mean_sq` (1 tile, now `rsqrt(mean_sq + eps)`) | `cb_input_tiles`: Wt held; `cb_mean_sq`: 1 (rsqrt scaler) |
| 5a (INPUT_IS_RM only) | tilize input sticks → Wt tiles (pass 2 push — reader re-supplied the same 32 sticks) | yes — `tilize` | `cb_input_raw_rm` (32 sticks, pass-2 batch) | `cb_input_tiles` (Wt more tiles appended after the pass-1 Wt tiles are popped by stage D below) | discussed in CB sync table; reader/tilize timing is back-to-back across passes |
| 6a (HAS_GAMMA false) | pass 2 stage D: `x · rsqrt` with BroadcastDim::Col | yes — `eltwise_chain` with `BinaryFpu<Mul, Col>` | `cb_input_tiles` (Wt, WaitAndPop) + `cb_mean_sq` (WaitNoPop) | `cb_output_tiles` (Wt) | `cb_input_tiles`: 0; `cb_mean_sq`: 1 (still held); `cb_output_tiles`: Wt |
| 6b (HAS_GAMMA true) | pass 2 stage D: `x · rsqrt` with BroadcastDim::Col → `cb_x_norm` | yes — `eltwise_chain` with `BinaryFpu<Mul, Col>` | `cb_input_tiles` (Wt, WaitAndPop) + `cb_mean_sq` (WaitNoPop) | `cb_x_norm` (Wt) | `cb_input_tiles`: 0; `cb_mean_sq`: 1 (still held); `cb_x_norm`: Wt |
| 7 (HAS_GAMMA true) | pass 2 stage E: `x_norm · gamma` with BroadcastDim::Row | yes — `eltwise_chain` with `BinaryFpu<Mul, Row>` | `cb_x_norm` (Wt, WaitAndPop) + `cb_gamma_tiled` (Wt, WaitAndPop) | `cb_output_tiles` (Wt) | `cb_x_norm`: 0; `cb_gamma_tiled`: 0 (gamma re-tilized next chunk); `cb_output_tiles`: Wt |
| 8 (OUTPUT_IS_RM only) | untilize `cb_output_tiles` → 32 sticks | yes — `untilize` | `cb_output_tiles` (Wt) | `cb_output_rm` (32 sticks) | `cb_output_tiles`: 0; `cb_output_rm`: 32 sticks |
| 9 | explicit `cb_pop_front(cb_mean_sq, 1)` — release rsqrt scaler for next row chunk | raw cb api | `cb_mean_sq` | — | `cb_mean_sq`: 0; all per-chunk CBs released |

End-of-kernel:

| # | Operation | Notes |
|---|-----------|-------|
| 10 | `cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1)` | Once, after final row chunk |

## Broadcast Verification

The kernel uses two FPU binary broadcast operations (stage D and stage E). The reduce produces a `(rows=1, cols=1)` "vector" in each row's tile (REDUCE_ROW output has shape `(Ht, 1)`, valid in col 0 of each output tile per `eltwise_chain.hpp:380-389` mapping).

| Phase | Op | CB_A (semantic name) valid region | CB_B (semantic name) valid region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| Stage D (`x · rsqrt`) | `mul_tiles_bcast<BroadcastType::COL>` (FPU) | `cb_input_tiles[t]`: full tile `[H=32, W=32]` valid | `cb_mean_sq[0]`: column 0 valid (REDUCE_ROW output, mapped to BroadcastDim::Col by `eltwise_chain.hpp:380-389`) | Col — `cb_mean_sq[0]`'s col 0 is replicated across columns of `cb_input_tiles[t]` |
| Stage E (`x_norm · gamma`) | `mul_tiles_bcast<BroadcastType::ROW>` (FPU) | `cb_x_norm[t]`: full tile valid | `cb_gamma_tiled[t]`: row 0 valid (tilize produced from a single gamma stick; rows 1-31 are uninitialized but unread under BroadcastDim::Row) | Row — `cb_gamma_tiled[t]`'s row 0 is replicated across rows of `cb_x_norm[t]` |

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|------------|
| 1 | `cb_input_tiles` must persist across stage A and stage D — same data, two reads — so the chain in stage A uses `CopyTilePolicy::NoWaitNoPop` and the caller pre-waits `Wt` tiles before the chain. Stage D then streams (WaitAndPop) and pops `Wt`. If the policies are flipped (e.g. stage A WaitAndPop), pass 2 sees an empty CB and hangs. | Stage A chain element MUST use `CopyTilePolicy::NoWaitNoPop` + `CbIndexMode::BlockIter`. Stage D must use `WaitAndPop` + `FirstTile`. The caller's external `cb_wait_front(cb_input_tiles, Wt)` is mandatory before stage A. |
| 2 | `cb_mean_sq` is reused as the rsqrt scaler — `transform_in_place` pops-then-re-pushes the same tile. Forgetting the final `cb_pop_front(cb_mean_sq, 1)` at end of row chunk leaks 1 page; next chunk's accumulate_reduce will block on `cb_reserve_back` once the (size-2) CB is full. | The compute kernel MUST `cb_pop_front(cb_mean_sq, 1)` at the end of each row chunk (phase 9). |
| 3 | Scaler CB must be `bfloat16` regardless of input dtype. Reduce LLK expects the scaler tile in `bfloat16` packed format. | `cb_scaler` data_format is hardcoded `ttnn.bfloat16` and `page_size = ttnn.tile_size(ttnn.bfloat16)`. |
| 4 | Reduce scaler API must be pool-type-aware: `prepare_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>(...)` (or the partial variant), NOT the legacy single-template overload — REDUCE_ROW + SUM/AVG uses matmul-path col-0 fill, whereas the legacy default is row-0 fill which produces wrong results on this path. | Reader uses `prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f/W)` or `prepare_partial_reduce_scalers<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, partial_w>(1.0f/W)`. |
| 5 | Non-tile-aligned W (RM input only): the last W-tile has garbage in its padded columns. Without the partial scaler, the reduce would sum (x²-of-garbage) into mean. | Reader prepares a 2-tile scaler (full + partial). Compute passes `ReducePartialScaler::last_tile_at(1)` to `reduce<>`. Verified per `reduce_helpers_dataflow.hpp:104-117` and `reduce_helpers_compute.hpp:158-186`. |
| 6 | Non-tile-aligned H (RM input/output): the last row chunk has fewer than 32 valid sticks. Reader pads in L1 (garbage in unfilled rows); compute produces garbage in those rows; writer skips them via `write_sticks_after_untilize`. | Writer runtime arg is `num_valid_sticks_for_this_core` (not always 32 × chunks_for_this_core). The writer helper handles the partial last chunk. |
| 7 | Gamma tile has only row 0 valid (rows 1-31 uninitialized after `tilize<Wt, ..>(1, /*total_input_pages=*/1)`). BroadcastDim::Row LLK reads only row 0, so the garbage rows are unread. If the implementer ever switches stage E to a non-broadcast multiply, the garbage rows would corrupt the output. | Stage E MUST use `BroadcastDim::Row`. Compile-time-assertable via the template arg. |
| 8 | FP32 path requires `ComputeConfigDescriptor(fp32_dest_acc_en=True)` so DEST holds 32-bit accumulators and the chain elements operate at full precision. Without this, fp32 inputs are silently downcast to bf16 in DEST, violating the spec. | `create_program_descriptor` reads `input_tensor.dtype == ttnn.float32` and sets `fp32_dest_acc_en` accordingly. `DEST_AUTO_LIMIT` halves automatically per `dest_helpers.hpp:88-99`. |
| 9 | DEST capacity: this kernel only ever holds 1 DEST slot at a time (every chain pushes to `Dst::D0`, every pack reads `Dst::D0`). No DEST-pressure risk even with fp32. | No special handling needed. |
| 10 | Sequential helper invariant for `cb_x_sq` and `cb_x_norm`: stage A pushes Wt before stage B consumes; stage D pushes Wt before stage E consumes. CB MUST hold all Wt tiles or the producer chain stalls on `cb_reserve_back`. | `cb_x_sq` and `cb_x_norm` sized to `Wt` pages (not 2). |
| 11 | `compute_kernel_hw_startup` must be the FIRST statement of `MAIN()` — calling it mid-kernel is undefined per `compute_kernel_hw_startup.h:26-30`. The chain helpers do NOT call it internally per `eltwise_chain.hpp:32`. | First statement of compute `MAIN()` is `compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_x_sq)`. After that, all per-element inits are owned by the chain / helper bodies. |
| 12 | The current design tilizes gamma every row chunk (re-reads the same 1 gamma stick from DRAM Ht times). For small Ht this is cheap; for very large Ht and very wide W this is wasted memory traffic. Acceptable for v1. | Future refinement: tilize gamma once at kernel start and keep `cb_gamma_tiled` persistent across all row chunks, then access via runtime-offset bcast (would require dropping the `eltwise_chain` for stage E in favor of raw `mul_tiles_bcast<BroadcastType::ROW>` with runtime tile index — chain does NOT expose `BlockIter + runtime base` index mode per `eltwise_chain.inl:528-532`). |
| 13 | Wide-W L1 budget: `cb_input_tiles` and `cb_x_sq` are each `Wt * tile_size` bytes; the design holds both simultaneously. For Wt > ~ floor(L1_budget / (2 * tile_size)) the kernel won't fit. The SUPPORTED universe in the op file caps W accordingly for v1. | v1 SUPPORTED constrains W. Future refinement: introduce W-blocking via `accumulate_reduce_block` (`streaming_reduce_helpers.hpp:53-61`) and a sub-stick reader; estimated L1 reduction proportional to `BLOCK_SIZE / Wt`. |
| 14 | Re-tilizing input twice per row chunk (RM input path): pass 1 and pass 2 each invoke the tilize compute helper, doubling tilize work for RM inputs. Acceptable for v1; TILE input path avoids this. | v1 ships with this overhead. Future refinement: keep `cb_input_tiles` full across both passes (already the design for TILE input); for RM input, the reader would need to push sticks only once and the compute would tilize only once, but that requires sizing `cb_input_tiles` to `2*Wt` so the pass-2 multiply can stream-pop without colliding with pass-1's held tiles (already specified). The implementer can elect to tilize only once by re-using the already-tilized pass-1 tiles in pass 2 — this requires reader to NOT push sticks the second time, which is a CT flag. Phase 0 implementation can choose either path; both are correct. |

## Structural impossibilities (for `INVALID` in `feature_spec.py`)

The skill-generated `feature_spec.py` already contains a complete `INVALID` list. The planner has no additional candidates beyond what the skill captured: `bfloat8_b` × `ROW_MAJOR_LAYOUT` (on both input and gamma) is the only universe-impossibility; the no-gamma canonicalization (5 entries) is the only canonical-redundancy bucket. The op signature accepts no other axes whose cross-product produces structurally impossible cells.

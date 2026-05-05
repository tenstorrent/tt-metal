# Operation Design: layer_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute |
| Goal | Normalize every logical row of an interleaved tensor across the last dimension `W`, optionally apply row-wise affine parameters, preserve the input layout, and handle non-tile-aligned shapes without host-side transforms. |
| Math | `mean[r] = sum_j(x[r,j]) / W`, `var[r] = sum_j((x[r,j] - mean[r])^2) / W`, `invstd[r] = rsqrt(var[r] + epsilon)`, `y[r,j] = ((x[r,j] - mean[r]) * invstd[r]) * gamma[j] + beta[j]` |
| Mode | Hybrid |
| Public API | `from ttnn.operations.layer_norm import layer_norm` |
| References | `METALIUM_GUIDE.md`; `tech_reports/tensor_layouts/tensor_layouts.md`; `tech_reports/tensor_accessor/tensor_accessor.md`; `.codex/references/ttnn-cb-memory-fundamentals.md`; `.codex/references/ttnn-python-utility-bindings.md`; `ttnn/ttnn/operations/toy_variance/`; `ttnn/ttnn/operations/toy_tilize_untilize/`; `ttnn/cpp/ttnn/kernel_lib/` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | Yes | rank >= 2, interleaved, layout in `{ROW_MAJOR_LAYOUT, TILE_LAYOUT}`, dtype in `{bfloat16, float32, bfloat8_b}` | None | RT |
| `gamma` | `ttnn.Tensor \| None` | No | shape `(1, 1, 1, W)`, `ROW_MAJOR_LAYOUT`, dtype matches `input_tensor.dtype` | `None` | RT |
| `beta` | `ttnn.Tensor \| None` | No | shape `(1, 1, 1, W)`, `ROW_MAJOR_LAYOUT`, dtype matches `input_tensor.dtype` | `None` | RT |
| `epsilon` | `float` | No | `epsilon > 0` | `1e-5` | RT |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor \| None` | No | any valid descriptor | `None` -> `ttnn.ComputeConfigDescriptor()` | Host |

## Validation

| Check | Condition | Action |
|-------|-----------|--------|
| Rank | `len(input_tensor.shape) < 2` | Raise `ValueError` |
| Input dtype | dtype not in `{ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b}` | Raise `ValueError` |
| Gamma width | `gamma is not None and gamma.shape[-1] != input_tensor.shape[-1]` | Raise `ValueError` |
| Beta width | `beta is not None and beta.shape[-1] != input_tensor.shape[-1]` | Raise `ValueError` |
| Gamma layout | `gamma is not None and gamma.layout != ttnn.ROW_MAJOR_LAYOUT` | Raise `ValueError` |
| Beta layout | `beta is not None and beta.layout != ttnn.ROW_MAJOR_LAYOUT` | Raise `ValueError` |
| Gamma dtype | `gamma is not None and gamma.dtype != input_tensor.dtype` | Raise `ValueError` |
| Beta dtype | `beta is not None and beta.dtype != input_tensor.dtype` | Raise `ValueError` |
| Block-format RM input | `input_tensor.dtype == ttnn.bfloat8_b and input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT` | Raise `ValueError` |
| Block-format affine | `input_tensor.dtype == ttnn.bfloat8_b and (gamma is not None or beta is not None)` | Raise `ValueError` |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Logical shape | Let `R = product(input_tensor.shape[:-1])` and `W = input_tensor.shape[-1]`; the operation normalizes the flattened 2D view `[R, W]`. |
| Layouts | `ROW_MAJOR_LAYOUT` and `TILE_LAYOUT` are both accepted natively. |
| Memory | Interleaved DRAM only. |
| Dtypes | `bfloat16`, `float32`, `bfloat8_b` |
| Gamma / beta broadcast | Always one logical row of width `W`; broadcast across all `R` rows. |

### Output

| Property | Value |
|----------|-------|
| Shape | Same as `input_tensor.shape` |
| Dtype | Same as `input_tensor.dtype` |
| Layout | Same as `input_tensor.layout` |
| Memory | Same memory config as the allocated output tensor; the entry point defaults to `ttnn.DRAM_MEMORY_CONFIG` |

## Derived Dimensions

| Symbol | Definition |
|--------|------------|
| `R` | `product(input_tensor.shape[:-1])` |
| `W` | `input_tensor.shape[-1]` |
| `Rt` | `ttnn.div_up(R, 32)` |
| `Wt` | `ttnn.div_up(W, 32)` |
| `BLOCK_W_TILES` | `ttnn.find_max_divisor(Wt, 8)` |
| `NUM_W_BLOCKS` | `Wt / BLOCK_W_TILES` |
| `rows_per_core_max` | `max(rows_per_core_group_1, rows_per_core_group_2)` after `ttnn.split_work_to_cores(...)` |
| `row_tiles_per_core_max` | `ttnn.div_up(rows_per_core_max, 32)` for RM input, or `rows_per_core_max` directly for TILE input because the work unit is already a row-tile |
| `rm_block_row_bytes` | `round_up(BLOCK_W_TILES * 32 * input_tensor.element_size(), ttnn.get_dram_alignment())` |

## Dataflow Strategy

| Path | Reader Payload | Compute Entry | Statistics Passes | Output Pass | Writer Payload |
|------|----------------|---------------|-------------------|-------------|----------------|
| `ROW_MAJOR_LAYOUT -> ROW_MAJOR_LAYOUT` | Reader emits one width-block row segment at a time into `cb_input_rm_rows`; segment tail bytes are zero-filled up to `rm_block_row_bytes`. | `tilize<..., Fp32Mode::Lossless>` for float32, standard tilize for bf16. | Pass 1 computes row means block-by-block; Pass 2 computes row variances block-by-block; both use partial reduce scalers to ignore padded columns in the last tile. | Pass 3 re-reads each width block, subtracts mean, multiplies by `rsqrt(var + epsilon)`, optionally applies gamma and beta, then `untilize`s the block. | Writer writes only the valid bytes of each row segment back to the correct column offset in the output rows. |
| `TILE_LAYOUT -> TILE_LAYOUT` | Reader emits contiguous tile blocks (`row_tiles_this_core x BLOCK_W_TILES`) directly into `cb_input_tiles`. | No layout conversion on the input path. | Same Pass 1 and Pass 2 helper sequence as the RM path. | Same Pass 3 helper sequence as the RM path, but final tiles are copied directly to `cb_output_tiles`. | Writer writes output tiles with `noc_async_write_tile`-style tiled page ordering. |
| `Gamma / beta` | Reader emits one RM width-block segment for gamma and/or beta only during Pass 3; both use the same block width as the input path. | Each segment is tilized once into `cb_gamma_tiles` / `cb_beta_tiles`, reused across every row-tile of the current width block, then popped. | Not used in Pass 1 or Pass 2. | Broadcast with `BroadcastDim::ROW` over the current width block only. | N/A |

## Width Blocking

| Decision | Value |
|----------|-------|
| Width-block heuristic | `BLOCK_W_TILES = ttnn.find_max_divisor(Wt, 8)` |
| Why width blocking is mandatory | RM output cannot buffer the full row width for arbitrary `W`, and compute-helper-to-compute-helper CBs must hold every tile a prior helper produces. Width blocking bounds both `cb_input_tiles` and `cb_work_tiles` to one block. |
| Why the divisor requirement is intentional | `NUM_W_BLOCKS` stays constant across all blocks, which keeps helper block shapes and CB sizing uniform. |
| Non-tile-aligned `W` handling | The final tile inside the final width block is handled by a partial reduce scaler in Pass 1 and Pass 2, and by valid-byte tracking in the RM reader/writer in Pass 3. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `ROW_MAJOR_LAYOUT`: contiguous actual rows of the flattened `[R, W]` view. `TILE_LAYOUT`: contiguous logical row-tiles of the flattened tiled `[Rt, Wt]` view. |
| Grid | `all_cores = CoreRangeSet(CoreRange((0,0), (grid.x-1, grid.y-1)))`, where `grid = device.compute_with_storage_grid_size()` |
| Split function | `ttnn.split_work_to_cores(all_cores, R)` for RM input; `ttnn.split_work_to_cores(all_cores, Rt)` for TILE input |
| Per-core work | Every active core owns all `NUM_W_BLOCKS` width blocks for its row range. RM cores derive `row_tiles_this_core = div_up(rows_this_core, 32)`. TILE cores use `row_tiles_this_core = work_units_this_core`. |
| Remainder handling | Use `core_group_1` / `core_group_2` from `ttnn.split_work_to_cores(...)`; CBs are sized to the max of the two groups, and per-core runtime args carry the true row count. |
| Single-core fallback | If `ttnn.split_work_to_cores(...)` returns `num_cores == 1`, the same kernels run on `(0,0)` only. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_rm_rows` | 0 | `rm_block_row_bytes` | `2 * min(32, rows_per_core_max)` | `input_tensor.dtype` | RM reader | `tilize()` | One width block of RM input, double-buffered at 32-row granularity |
| `cb_input_tiles` | 1 | `ttnn.tile_size(input_tensor.dtype)` | `row_tiles_per_core_max * BLOCK_W_TILES` | `input_tensor.dtype` | TILE reader or RM `tilize()` | `sub()` / `accumulate_reduce_block()` | One compute block; must hold the full block because a compute helper produces it and a later compute helper consumes it |
| `cb_reduce_scaler` | 2 | `ttnn.tile_size(ttnn.bfloat16)` | `2` if `W % 32 != 0`, else `1` | `ttnn.bfloat16` | Reader | `accumulate_reduce_block()` | All three passes |
| `cb_epsilon_scalar` | 3 | `ttnn.tile_size(ttnn.bfloat16)` | `1` | `ttnn.bfloat16` | Reader | `add_in_place<BroadcastDim::SCALAR>()` | Between Pass 2 and Pass 3 |
| `cb_gamma_rm_rows` | 4 | `rm_block_row_bytes` | `1` | `gamma.dtype` | Reader | `tilize()` | Current Pass 3 width block only |
| `cb_beta_rm_rows` | 5 | `rm_block_row_bytes` | `1` | `beta.dtype` | Reader | `tilize()` | Current Pass 3 width block only |
| `cb_gamma_tiles` | 6 | `ttnn.tile_size(input_tensor.dtype)` | `BLOCK_W_TILES` | `input_tensor.dtype` | `tilize()` | `mul_in_place<BroadcastDim::ROW>()` | Current Pass 3 width block only |
| `cb_beta_tiles` | 7 | `ttnn.tile_size(input_tensor.dtype)` | `BLOCK_W_TILES` | `input_tensor.dtype` | `tilize()` | `add_in_place<BroadcastDim::ROW>()` | Current Pass 3 width block only |
| `cb_output_tiles` | 16 | `ttnn.tile_size(output_tensor.dtype)` | `2` | `output_tensor.dtype` | `copy_tiles()` | TILE writer | TILE output path only |
| `cb_output_rm_segments` | 17 | `ttnn.tile_size(output_tensor.dtype)` | `2 * BLOCK_W_TILES` | `output_tensor.dtype` | `untilize()` | RM writer | RM output path only; double-buffered one tile-row at a time |
| `cb_mean_tiles` | 24 | `ttnn.tile_size(input_tensor.dtype)` | `row_tiles_per_core_max` | `input_tensor.dtype` | Pass 1 `accumulate_reduce_block()` | Pass 2 / Pass 3 `sub<BroadcastDim::COL>()` | Persistent from end of Pass 1 through end of Pass 3 |
| `cb_variance_tiles` | 25 | `ttnn.tile_size(input_tensor.dtype)` | `row_tiles_per_core_max` | `input_tensor.dtype` | Pass 2 `accumulate_reduce_block()` | `add_in_place<BroadcastDim::SCALAR>()`, then `sfpu_op<Rsqrt>` | Persistent from end of Pass 2 until consumed by `sfpu_op()` |
| `cb_invstd_tiles` | 26 | `ttnn.tile_size(input_tensor.dtype)` | `row_tiles_per_core_max` | `input_tensor.dtype` | `sfpu_op<Rsqrt>()` | Pass 3 `mul_in_place<BroadcastDim::COL>()` | Persistent throughout Pass 3 |
| `cb_work_tiles` | 27 | `ttnn.tile_size(input_tensor.dtype)` | `row_tiles_per_core_max * BLOCK_W_TILES` | `input_tensor.dtype` | `sub()` in Pass 2 / Pass 3 | `square_in_place()`, `accumulate_reduce_block()`, affine helpers, `copy_tiles()` / `untilize()` | One compute block; compute-owned only |

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|-------------------------|--------------------------|---------------------------|--------------|
| RM block reader | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:100-113` | `page_id = start_row + row`, `offset = block_col_start_bytes` | N/A | `cb_input_rm_rows`, `cb_gamma_rm_rows`, `cb_beta_rm_rows` | Uses per-row byte offsets to read one width block from a larger RM row. Helpers considered and rejected: `read_sticks_for_tilize` only accepts contiguous full-stick reads via `(accessor, total_num_rows, row_bytes, start_page)` and has no column-offset argument, so it cannot read width blocks without buffering whole rows (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:47-79`). |
| RM block reader | raw_api | `cb_reserve_back`, `get_write_ptr`, `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:404-423`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:322-327`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:208-223` | Reserve one row-segment page per RM row or one gamma/beta row page | N/A | `cb_input_rm_rows`, `cb_gamma_rm_rows`, `cb_beta_rm_rows` | Producer-side CB control in the dataflow kernels. |
| RM block reader | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:533-574` | `size = valid_block_bytes` with explicit zero-fill of the remainder | N/A | `cb_input_rm_rows`, `cb_gamma_rm_rows`, `cb_beta_rm_rows` | Reads only the valid byte range of the current width block; padded tail bytes are zeroed in L1 before `cb_push_back`. |
| Constant preparation | raw_api | `noc_async_write` / local L1 stores | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:827-847` | Fill one epsilon tile in `cb_epsilon_scalar` | N/A | `cb_epsilon_scalar` | Helpers considered and rejected: `prepare_reduce_scaler` is explicitly only for reduce-scaler tiles, not arbitrary constants (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:21-38`). |
| Reduce scaler setup | helper | `prepare_reduce_scaler` / `prepare_partial_reduce_scalers` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:65-67`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:136-142` | `<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` with caller-supplied `1.0f / W` | N/A | `cb_reduce_scaler` | Emits one full scaler tile, or a full + partial pair when `W % 32 != 0`. |
| RM input tilize | helper | `tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:145-154` | `<BLOCK_W_TILES, cb_input_rm_rows, cb_input_tiles, InitAndUninit, WaitBlock, UnpackAndPackReconfigure, Fp32Mode::Lossless>` for float32; `Fp32Mode::Fast` for bf16 | `cb_input_rm_rows` | `cb_input_tiles` | Uses row-granularity input pages (`total_input_pages = rows_this_core` for the current block). Float32 must use `Fp32Mode::Lossless` because the default fast path truncates to TF32 (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:48-54`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:138-143`). |
| Gamma/beta tilize | helper | `tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:145-154` | `<BLOCK_W_TILES, cb_gamma_rm_rows, cb_gamma_tiles, ...>` and `<BLOCK_W_TILES, cb_beta_rm_rows, cb_beta_tiles, ...>` | `cb_gamma_rm_rows`, `cb_beta_rm_rows` | `cb_gamma_tiles`, `cb_beta_tiles` | One-row RM segments are tilized once per width block. |
| Pass 1 mean accumulation | helper | `accumulate_reduce_block` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:47-61`, `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.inl:21-52` | `<PoolType::SUM, ReduceDim::REDUCE_ROW>` | `cb_input_tiles`, `cb_reduce_scaler` | `cb_mean_tiles` | Uses `Accumulate::at(cb_mean_tiles, block_idx)` internally and routes the partial scaler only to the last width block. |
| Pass 2 centering | helper | `sub` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:282-293` | `<BroadcastDim::COL, WaitAndPopPerTile, WaitUpfrontNoPop>` | `cb_input_tiles`, `cb_mean_tiles` | `cb_work_tiles` | `cb_mean_tiles` persists across all width blocks of Pass 2. |
| Pass 2 squaring | helper | `square_in_place` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:464-469`, `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.inl:769-782` | `<reconfig=INPUT_AND_OUTPUT>` | `cb_work_tiles` | `cb_work_tiles` | Compute-owned full-block CB only; no reader/writer may push/pop this CB while the in-place helper runs. |
| Pass 2 variance accumulation | helper | `accumulate_reduce_block` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:47-61`, `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.inl:21-52` | `<PoolType::SUM, ReduceDim::REDUCE_ROW>` | `cb_work_tiles`, `cb_reduce_scaler` | `cb_variance_tiles` | Same block routing as Pass 1; the last width block receives the partial scaler. |
| Variance finalization | helper | `add_in_place` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:437-444` | `<BroadcastDim::SCALAR, WaitUpfrontNoPop>` | `cb_variance_tiles`, `cb_epsilon_scalar` | `cb_variance_tiles` | Adds epsilon to every row-stat tile before rsqrt. Only column 0 is semantically consumed later. |
| Inverse std | helper | `sfpu_op` with `Rsqrt` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:447-450`, `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1438-1445` | `sfpu_op<cb_variance_tiles, SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile>(cb_invstd_tiles, row_tiles_this_core, Rsqrt<>{})` | `cb_variance_tiles` | `cb_invstd_tiles` | Uses the helper-managed SFPU path instead of raw `rsqrt_tile`. Drains variance tiles as it produces inverse-std tiles. |
| Pass 3 centering | helper | `sub` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:282-293` | `<BroadcastDim::COL, WaitAndPopPerTile, WaitUpfrontNoPop>` | `cb_input_tiles`, `cb_mean_tiles` | `cb_work_tiles` | Same broadcast contract as Pass 2. |
| Pass 3 normalize | helper | `mul_in_place` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:455-462` | `<BroadcastDim::COL, WaitUpfrontNoPop>` | `cb_work_tiles`, `cb_invstd_tiles` | `cb_work_tiles` | Reuses one inverse-std column tile per row-tile across the full width block. |
| Optional gamma | helper | `mul_in_place` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:455-462` | `<BroadcastDim::ROW, WaitUpfrontNoPop>` | `cb_work_tiles`, `cb_gamma_tiles` | `cb_work_tiles` | Reuses one gamma tile row across every row-tile of the current width block. |
| Optional beta | helper | `add_in_place` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:437-444` | `<BroadcastDim::ROW, WaitUpfrontNoPop>` | `cb_work_tiles`, `cb_beta_tiles` | `cb_work_tiles` | Same row broadcast contract as gamma. |
| TILE output drain | helper | `copy_tiles` | `ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp:146-150`, `ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.inl:20-72` | `<CopyInputPolicy::WaitAndPop>` | `cb_work_tiles` | `cb_output_tiles` | Streaming compute-to-writer handoff for the TILE path. |
| RM output untilize | helper | `untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:133-141` | `<BLOCK_W_TILES, cb_work_tiles, cb_output_rm_segments>` | `cb_work_tiles` | `cb_output_rm_segments` | Emits one tile-row (`BLOCK_W_TILES` pages) at a time to the RM writer. |
| RM block writer | raw_api | `cb_wait_front`, `get_read_ptr`, `cb_pop_front` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:473-485`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:344-347`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:258-274` | Wait/pop one untilized segment page at a time | `cb_output_rm_segments` | N/A | Consumer-side CB control in the RM writer. |
| RM block writer | raw_api | `noc_async_write` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:827-847` | `size = valid_block_bytes`, destination offset = `block_col_start_bytes` | `cb_output_rm_segments` | Output tensor | Helpers considered and rejected: `write_sticks_after_untilize` only writes full sticks from a `start_page` and has no per-block column offset (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:82-109`), so the RM writer must issue raw partial-row writes. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|----|-----------------------------------|-----------------------------------|---------------|
| Pass 2 | `sub` | `cb_input_tiles`: All | `cb_mean_tiles`: Col0 | `COL` |
| Pass 3 | `sub` | `cb_input_tiles`: All | `cb_mean_tiles`: Col0 | `COL` |
| Pass 3 | `mul_in_place` (invstd) | `cb_work_tiles`: All | `cb_invstd_tiles`: Col0 | `COL` |
| Pass 3 | `mul_in_place` (gamma) | `cb_work_tiles`: All | `cb_gamma_tiles`: Row0 | `ROW` |
| Pass 3 | `add_in_place` (beta) | `cb_work_tiles`: All | `cb_beta_tiles`: Row0 | `ROW` |
| Finalize variance | `add_in_place` (epsilon) | `cb_variance_tiles`: Col0 | `cb_epsilon_scalar`: single tile | `SCALAR` |

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | Prepare `1/W` reduce scaler tile(s) and epsilon scalar tile | Reader raw + reduce-scaler helper | N/A | `cb_reduce_scaler`, `cb_epsilon_scalar` | Both constant CBs persist; nothing is popped yet |
| 2 | For each width block in Pass 1: read current input block, tilize if needed, accumulate row means | Yes | `cb_input_tiles` current block, `cb_reduce_scaler` persistent | `cb_mean_tiles` persistent | `cb_input_tiles` drained each block; `cb_mean_tiles` holds `row_tiles_this_core` row-stat tiles after the last block |
| 3 | For each width block in Pass 2: read current input block, center by mean, square in place, accumulate row variances | Yes | `cb_input_tiles` current block, `cb_mean_tiles` persistent, `cb_reduce_scaler` persistent | `cb_work_tiles` transient, `cb_variance_tiles` persistent | `cb_work_tiles` drained each block; `cb_variance_tiles` holds `row_tiles_this_core` variance tiles after the last block |
| 4 | Add epsilon in place to every variance tile | Yes | `cb_variance_tiles`, `cb_epsilon_scalar` | `cb_variance_tiles` | `cb_epsilon_scalar` remains available until explicitly popped after rsqrt setup |
| 5 | Convert variance to inverse std with SFPU rsqrt | Yes | `cb_variance_tiles` | `cb_invstd_tiles` | `cb_variance_tiles` is drained by `sfpu_op`; `cb_invstd_tiles` persists for Pass 3 |
| 6 | For each width block in Pass 3: read current input block, center, multiply by inverse std, optionally multiply by gamma and add beta | Yes | `cb_input_tiles` current block, `cb_mean_tiles` persistent, `cb_invstd_tiles` persistent, optional `cb_gamma_tiles` / `cb_beta_tiles` current block | `cb_work_tiles` | `cb_work_tiles` holds the final output block until drained; gamma/beta block tiles are popped once per block |
| 7 | Drain the final block to the output path | Yes | `cb_work_tiles` | `cb_output_tiles` (TILE path) or `cb_output_rm_segments` (RM path) | Current width block fully drained to the writer |
| 8 | End-of-pass cleanup | Raw CB pops | `cb_mean_tiles`, `cb_invstd_tiles`, `cb_reduce_scaler`, `cb_epsilon_scalar` | N/A | All persistent CBs are popped exactly once after the final width block |

## Build Order

| Step | Bring-up target | Verification hint |
|------|-----------------|-------------------|
| 1 | TILE input, TILE output, no affine, single core, aligned bf16 | Use all-ones rows and distinct row offsets; verify mean tiles become the row constants and output rows become zeros. |
| 2 | Enable width blocking on TILE input (`Wt > BLOCK_W_TILES`) | DPRINT the per-block row means and ensure the final block alone receives the partial scaler when `W % 32 != 0`. |
| 3 | Add Pass 2 variance + rsqrt finalization | Use rows with known variance (e.g. arithmetic progressions) and verify `invstd` against the PyTorch reference for one row-tile. |
| 4 | Add Pass 3 normalize and optional gamma/beta on TILE input | Use `gamma = ones`, `beta = zeros` first, then `gamma = 2`, `beta = 3` to make affine errors obvious. |
| 5 | Add RM reader/tilize and RM writer/untilize width-block paths | Start with aligned `W = BLOCK_W_TILES * 32`; then enable non-aligned `W` and check that only valid bytes are written back. |
| 6 | Add multi-core work splitting | Compare 1-core and N-core outputs on identical inputs; DPRINT per-core start row / row count to verify contiguous row partitioning. |

## Key Risks and Gotchas

| Topic | Requirement |
|-------|-------------|
| Block-format layout constraint | `bfloat8_b` has no valid ROW_MAJOR tensor representation in this repo, so RM input and affine tensors are rejected for that dtype. |
| Full-block compute intermediates | `cb_input_tiles` and `cb_work_tiles` must hold the entire current width block because `tilize`, `sub`, `square_in_place`, `copy_tiles`, and `untilize` are sequential compute helpers with no overlap between them. |
| Float32 tilize | All float32 RM tilize calls must use `Fp32Mode::Lossless`; the fast mode truncates to TF32. |
| Partial reduce scaler | If `W % 32 != 0`, the reader must emit two scaler tiles and compute must only pop both after the final pass completes. |
| Epsilon tile | Do not misuse `prepare_reduce_scaler` for epsilon; it is restricted to reduce-scaler layouts only. |
| RM segment IO | RM reader and writer operate on width-block byte ranges, not full rows. Failing to zero-fill input tails or respect output valid-byte counts will corrupt non-aligned shapes. |
| Gamma / beta reuse | `cb_gamma_tiles` and `cb_beta_tiles` must use `WaitUpfrontNoPop` during the current width block and be popped once after the block finishes. |
| CB single-producer rule | `cb_work_tiles`, `cb_mean_tiles`, `cb_variance_tiles`, and `cb_invstd_tiles` are compute-owned only; reader and writer never touch them. |
| Default compute config | The Python entry point must resolve `compute_kernel_config = ttnn.ComputeConfigDescriptor()` when the caller passes `None`, then pass that object directly into the compute `KernelDescriptor.config` field. |

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB
- [x] Reduce scaler CB is bfloat16
- [x] Reduce scaler uses pool-type-aware API (`prepare_reduce_scaler<cb, PoolType, ReduceDim>` / `prepare_partial_reduce_scalers<cb, PoolType, ReduceDim, ...>`)
- [x] DEST capacity is delegated to helper-managed batching (`DEST_AUTO_LIMIT`) for reduce and SFPU phases
- [x] Sequential helper intermediates (`cb_input_tiles`, `cb_work_tiles`) are sized to the full current width block
- [x] Page sizes are aligned to tile size or aligned RM segment size
- [x] RM CBs count pages in row segments; tile CBs count pages in tiles
- [x] Every `cb_wait_front` call on a given CB uses a consistent page count contract

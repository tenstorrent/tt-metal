# Reduce H (Height-Dimension Reduction) Implementation Analysis

## Overview

The **Reduce H** operation performs reduction along the height dimension of a tensor. Given an input tensor of shape `[N, C, H, W]` in tile layout, it reduces the H dimension to produce an output of shape `[N, C, 1, W]` (one tile row of height). Each column of tiles is independently reduced to a single output tile, so the output has `NC * Wt` total tiles.

The reduction supports SUM, AVG, and MAX operations (controlled by `ReduceOpMath`), with an optional scaler and optional negation. Internally, the hardware reduction dimension name is `REDUCE_COL` (collapsing rows within each column).

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp`

**Critical naming convention**: The host-level dimension `ReduceOpDim::H` maps to the hardware/LLK dimension `ReduceDim::REDUCE_COL`. This is because "reduce H" means "collapse height", which in tile terms means reducing along column directions (accumulating rows for each column position). The `get_defines` function in `reduce_op.cpp` (line 27) performs this mapping: `ReduceOpDim::H -> "ReduceDim::REDUCE_COL"`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile column |
| **Unit size** | `Ht` tiles (one full column of height tiles) reduced to 1 output tile |
| **Total units** | `NC * Wt` (number of batches times width in tiles) |
| **Loop structure** | Outer: iterate over assigned columns. Inner: for each column, iterate over `Ht` rows, feeding tiles to compute which accumulates all `Ht` tiles into one DST register and packs a single output tile. |

Each core is assigned a contiguous range of "columns" where a column represents one tile-width position across all `Ht` tile rows. The columns span across batches: column index `i` maps to batch `i / Wt` and width position `i % Wt`.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[N*C, H, W]` (flattened to 3 effective dimensions) |
| **Dimension convention** | `NC, H, W` where `NC = shape[0]*shape[1]`, `H = shape[2]`, `W = shape[3]` |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED (primary path) or WIDTH_SHARDED (alternate path) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Configurable (bfloat16, float32, etc.) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | `[N*C, 1_tile, W]` (H dimension reduced to single tile row) |
| **Dimension convention** | `NC, 1, W` |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED (primary path) or WIDTH_SHARDED (alternate path) |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Configurable (may differ from input via `output_dtype`) |

### Layout Transformations

No explicit tilize/untilize occurs within the program factory. The input tensor must already be in TILE_LAYOUT. The `reduce_tile` LLK internally handles the row-to-column accumulation within and across tiles. The scaler tile is constructed in row-0-only format (only the first row of each face is populated with the scaler value).

## Data Flow Pattern

### Interleaved Path (Primary Focus)

**Stage 1 - Reader prepares scaler tile (once)**:
The reader kernel calls `dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)` to construct the scaler tile in CB `c_2`. This function:
1. Reserves 1 tile in `c_2`
2. Zeros all faces via NOC read from hardware zeros region
3. Fills row 0 of each face with the scaler value (bit-cast to appropriate format)
4. Pushes the tile

**Stage 2 - Reader feeds input tiles in chunked column-major order**:
The reader iterates tiles in `N -> W_skip(chunk) -> H -> W_chunk` order. For each chunk of `row_chunk` columns (where `row_chunk = DEST_AUTO_LIMIT`):
1. For each of the `Ht` rows in the height dimension:
   - For each column within the current chunk:
     - Reserve 1 tile in CB `c_0`
     - Read tile from DRAM via tensor accessor
     - Push 1 tile to CB `c_0`

The reader sends tiles one at a time, and the compute kernel processes them one at a time (WaitAndPopPerTile policy).

**Stage 3 - Compute reduces height within each column chunk**:
The compute kernel calls:
```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3,
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
```

Inside the `reduce()` library function for `REDUCE_COL` with `WaitAndPopPerTile`:
1. `reduce_init` configures unpacker, math, and packer for reduction
2. `cb_wait_front(scaler_cb, 1)` waits for scaler tile
3. For each batch `nc` in `[0, NC)`:
   - For each chunk of `chunk_size` columns starting at `wt`:
     - `tile_regs_acquire()` -- zeroes DST registers, grants math access
     - For each row `ht` in `[0, Ht)`:
       - For each column `i` in `[wt, chunk_end)`:
         - `cb_wait_front(input_cb, 1)` -- wait for 1 input tile
         - `reduce_tile<REDUCE_OP, REDUCE_COL>(input_cb, scaler_cb, 0, 0, dst_idx)` -- accumulate into DST[dst_idx], dst_idx increments per column in chunk
         - `cb_pop_front(input_cb, 1)` -- free input tile
     - `tile_regs_commit()` -- hand DST to packer
     - `tile_regs_wait()` -- packer waits for math
     - For each column in chunk: `pack_tile(dst_idx, output_cb)` with per-tile reserve/push
     - `tile_regs_release()` -- free DST for next acquire
4. `reduce_uninit` resets packer edge mask

**Stage 4 - Writer outputs reduced tiles**:
For each output tile:
1. `cb_wait_front(output_cb, 1)` -- wait for packed tile
2. Write tile to DRAM via tensor accessor (`noc_async_write_page`)
3. `cb_pop_front(output_cb, 1)` -- free output CB slot

### Key Insight: Chunked Column Processing and DEST Register Mapping

The `REDUCE_COL` path processes columns in chunks of `DEST_AUTO_LIMIT` (typically 4-16 depending on FP32 accumulation and sync mode). Within a chunk, each column gets its own DST register slot. All `Ht` rows for the chunk are processed, and `reduce_tile` accumulates tile `(ht, col_i)` into `DST[col_i - chunk_start]`. This means the reader must deliver tiles in `H -> W_chunk` inner order (for each height row, deliver all columns in the chunk), which is exactly what the reader's triple-nested loop does.

After all rows are processed, the chunk's DST registers contain the fully reduced results for `chunk_size` columns, which are packed to the output CB.

## Circular Buffer Configuration

### Interleaved Path (non-negate)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler tile | 1 tile | 1 tile | Single | Reader | Compute | Program (written once, read persistently) |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |

### Interleaved Path (negate mode)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | `chunk_size` tiles | `chunk_size` tiles | Single | Reader | Compute | Block |
| c_2 | cb_scaler | Scaler tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_output | Output tile staging | `chunk_size` tiles | `chunk_size` tiles | Single | Compute | Writer | Block |
| c_4 | cb_acc | Accumulator | `chunk_size` tiles | `chunk_size` tiles | Single | Compute | Compute | Block |
| c_5 | cb_ineg | Negation intermediate | `chunk_size` tiles | `chunk_size` tiles | Single | Compute | Compute | Block |

### Width-Sharded Path

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_src1 | Sharded input (globally allocated) | `shard_tiles` | N/A | N/A | Hardware (shard) | Reader | Program |
| c_2 | cb_scaler | Scaler tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_output | Output tiles (globally allocated) | `output_shard_tiles` | N/A | N/A | Compute | Hardware (shard) | Program |

**Note on scaler CB data format**: The scaler uses `Float16_b` by default. It only uses `Float32` when the input data format is `Float32` AND the architecture is not Blackhole (line 39-41 of program factory).

## Pipeline Pattern Summary

For the standard interleaved, non-negate path:
- **CB c_0 (input)**: 2-tile capacity with 1-tile block = **Double-buffered**. Reader can fill one slot while compute processes another.
- **CB c_2 (scaler)**: 1-tile capacity, written once and consumed persistently = **Single-buffered, persistent**. The `cb_wait_front` in compute waits once, and the tile is never popped.
- **CB c_3 (output)**: 2-tile capacity with 1-tile block = **Double-buffered**. Compute can pack one tile while writer sends another.

## Index Calculations

### Reader Tile Index Mapping (Interleaved Path)

The reader uses `TensorAccessor` for DRAM address resolution. The key tile index is `curr_id`, which maps a logical tile position in the row-major tile grid to a page ID for the tensor accessor.

**Tile numbering convention**: Tiles are numbered in row-major order within the `[NC, Ht, Wt]` grid:
- Tile at `(nc, ht, wt)` has index: `nc * HtWt + ht * Wt + wt`

**Starting tile for a core**: Given `num_cols_read` total columns already assigned to previous cores:
- `col_start_tile_id = (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)` -- this computes the tile ID of the first tile in the first assigned column, accounting for batch boundaries.
- `curr_col_in_batch = num_cols_read % Wt` -- the column position within the current batch.

**Column traversal**: The reader traverses columns in chunks. Within each chunk, for each height row, it visits all chunk columns. The batch-crossing logic (lines 78-85 of reader) handles the case where consecutive assigned columns span a batch boundary: when `w == Wt`, the column index wraps to the next batch and `col_start_tile_id` is advanced accordingly.

**Height stride**: After processing all columns in the chunk for one height row, the reader advances to the next height row via `curr_id = reset_curr_id + (j + 1) * Wt` (line 87), which moves down one tile row (stride = `Wt` tiles).

### Compute Index Mapping

The compute kernel does not perform explicit index calculations. It relies on the reader to deliver tiles in the correct order and uses the `WaitAndPopPerTile` policy, processing tiles at CB index 0 (always the front of the CB). The `reduce_tile` call with `tile_idx=0` always operates on the next available tile.

The DST register index `dst_idx` starts at 0 for each chunk and increments for each column within the chunk (see `reduce_helpers_compute.inl` lines 396-416). This means DST[0] accumulates column `wt`, DST[1] accumulates column `wt+1`, etc.

## Memory Access Patterns

### Read Pattern
- **Ordering**: Column-major within chunks, row-major across chunks. Specifically: for each chunk of `DEST_AUTO_LIMIT` consecutive tile columns, the reader reads all `Ht` rows for those columns before moving to the next chunk.
- **Within a chunk row**: Sequential tile reads across the chunk width (contiguous in DRAM if interleaved)
- **Between chunk rows**: Strided by `Wt` tiles (jumping down one tile row in the height dimension)
- **Access type**: Single-tile reads via tensor accessor, with `noc_async_read_barrier` after each tile (synchronous read pattern)

### Write Pattern
- **Ordering**: Sequential across output tiles (one tile per reduced column)
- **Access type**: Single-tile writes via tensor accessor with `noc_async_writes_flushed` per tile, final `noc_async_write_barrier`

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` |
| **Total cores** | `min(num_cols, max_cores)` where `num_cols = NC * Wt` |
| **Work per core** | `num_cols_per_core_group_1` or `num_cols_per_core_group_2` columns |
| **Load balancing** | Two-group split via `split_work_to_cores`: group 1 gets `ceil(num_cols/num_cores)` columns, group 2 gets `floor(num_cols/num_cores)` |

The work unit is one "column" (tile column across full height). `split_work_to_cores` distributes `NC * Wt` columns across cores. Cores in `core_group_1` get one more column than cores in `core_group_2` if the division has a remainder.

For width-sharded mode, the core grid comes from the shard spec, and each core handles its local shard's columns.

## Arguments

### Reader Compile-Time Arguments (Interleaved Path)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile rows in height dimension |
| 1 | Wt | uint32_t | Number of tile columns in width dimension |
| 2 | HtWt | uint32_t | Total tiles per batch (`Ht * Wt`) |
| 3 | scaler_bits | uint32_t | Scaler float value bit-cast to uint32_t |
| 4+ | TensorAccessorArgs | varies | Tensor accessor parameters for DRAM address mapping |

### Reader Compile-Time Defines (Interleaved Path)

| Define | Value | Description |
|--------|-------|-------------|
| ENABLE_FP32_DEST_ACC | "0" or "1" | Controls DEST_AUTO_LIMIT calculation in reader |
| DST_SYNC_FULL | "0" or "1" | Controls DEST_AUTO_LIMIT calculation in reader |

### Reader Runtime Arguments (Interleaved Path)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | col_start_tile_id | uint32_t | Starting tile ID for this core's first column |
| 2 | curr_col_in_batch | uint32_t | Column position within current batch |
| 3 | num_cols | uint32_t | Total number of columns assigned to this core |

### Compute Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile rows in height dimension |
| 1 | num_cols_per_core | uint32_t | Number of columns to reduce (labeled "Wt" in code comments but actually per-core count) |
| 2 | NC | uint32_t | Always 1 (batches are handled by flattening into num_cols) |

### Compute Compile-Time Defines

| Define | Value | Description |
|--------|-------|-------------|
| REDUCE_OP | "PoolType::SUM" / "PoolType::AVG" / "PoolType::MAX" | Reduction operation type |
| REDUCE_DIM | "ReduceDim::REDUCE_COL" | Always REDUCE_COL for H reduction |

**Important note on compute args**: The compute kernel receives `NC=1` because batches are already flattened into the column count. The reader delivers `num_cols * Ht` tiles total, which the compute sees as `NC=1` batch of `Ht` rows by `num_cols` columns. The `reduce()` library function then chunks the `num_cols` columns by `DEST_AUTO_LIMIT`.

### Writer Compile-Time Arguments (Interleaved Path)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_3) |
| 1+ | TensorAccessorArgs | varies | Tensor accessor parameters for DRAM address mapping |

### Writer Runtime Arguments (Interleaved Path)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Number of output tiles to write (`num_cols_per_core`) |
| 2 | start_id | uint32_t | Starting output tile index (`num_cols_read`) |

## Kernel Implementations

### Reader Kernel (Interleaved Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM (tensor accessor) | CB c_0 (input), CB c_2 (scaler) | Read tiles in chunked column-major order; prepare scaler tile once |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`
- **Key Logic**:
  - `DEST_AUTO_LIMIT` is computed at compile time from the same defines as the compute kernel, ensuring reader and compute agree on chunk size.
  - `prepare_reduce_scaler<cb_id_in2>(scaler_f)` constructs the scaler tile by zeroing all faces and filling row 0 of each face with the scaler value.
  - The triple-nested loop (lines 60-89) delivers tiles in the order compute expects: outer over column chunks, middle over height rows, inner over columns within chunk. This matches the `REDUCE_COL` + `WaitAndPopPerTile` consumption pattern.
  - Batch boundary crossing (lines 78-85): when `w == Wt`, the column pointer wraps to the start of the next batch. This allows contiguous column assignment across batch boundaries.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (RISCV_2) | N/A | CB c_0 (input), CB c_2 (scaler) | CB c_3 (output) | Height reduction via reduce_tile LLK |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp`
- **Key Logic**:
  - Calls `compute_kernel_hw_startup(c_0, c_2, c_3)` to initialize all three compute RISC-V cores (unpack, math, pack) with the correct CB data formats.
  - Delegates to `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(c_0, c_2, c_3, ReduceInputBlockShape::of(Ht, Wt, NC))`.
  - Inside the library: for each chunk of columns, `tile_regs_acquire()` zeroes DST, then for each height row, `reduce_tile` accumulates into DST[dst_idx] per column. After all rows, results are packed to output CB.
  - The `REDUCE_DIM` define is `ReduceDim::REDUCE_COL`, meaning `reduce_tile` accumulates along the height (row) dimension of each tile.

### Writer Kernel (Interleaved Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | NCRISC (RISCV_1) | NOC1 | CB c_3 (output) | DRAM (tensor accessor) | Write reduced tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic single-tile-at-a-time writer. Waits for one tile in output CB, writes it to DRAM at `start_id + i`, pops, and repeats for `num_tiles` tiles.

## Implementation Notes

### Scaler Setup Pattern
The scaler is prepared by the reader kernel, not the compute kernel. The float scaler value is bit-cast to `uint32_t` on the host (line 131: `std::bit_cast<uint32_t>(operation_attributes.scaler)`) and passed as a compile-time argument. The reader kernel bit-casts it back to float and calls `prepare_reduce_scaler`, which handles format conversion (bfloat16 packing or float32 pass-through) and tile construction.

For a softmax operation performing `sum(exp(x))` reduction along H, the scaler would be `1.0f` (since we want a plain sum, not an average).

### DEST_AUTO_LIMIT Synchronization
Both reader and compute kernels compute `DEST_AUTO_LIMIT` from the same compile-time configuration (FP32 accumulation mode and sync mode). The program factory passes `ENABLE_FP32_DEST_ACC` and `DST_SYNC_FULL` as reader defines (lines 164-165), which must match the compute kernel's JIT-generated values. This ensures both kernels agree on the chunk size without explicit host communication.

Capacity table for DEST_AUTO_LIMIT:
- SyncHalf + fp16: 8 tiles
- SyncHalf + fp32: 4 tiles
- SyncFull + fp16: 16 tiles
- SyncFull + fp32: 8 tiles

### Negate Mode
When `operation_attributes.negate` is true, the program uses `reduce_h_neg.cpp` as the compute kernel (line 207) and allocates additional CBs (c_4 for accumulator, c_5 for negation intermediate). This is used for implementing `reduce_min` as `-reduce_max(-x)`.

### Width-Sharded Alternate Path
The program factory has a complete alternate path for width-sharded tensors (lines 69-76, 81-95, 113-120, 147-156, 178-186, 251-260). The sharded reader copies tiles from the locally-allocated shard CB (c_1) to the input CB (c_0) one at a time, reading tiles column-major within each batch. The sharded writer simply waits for all output tiles to appear in the globally-allocated output CB. The chunk size is forced to 1 for width-sharded mode (line 51).

### How This Pattern Applies to Softmax
For a softmax operation needing `sum(exp(x))` along H:
1. **Scaler**: Pass `1.0f` as the scaler for a plain SUM reduction.
2. **Input policy**: `WaitAndPopPerTile` is the simplest and works with 2-tile CB capacity.
3. **Reader order**: The chunked column-major order naturally matches what `REDUCE_COL` expects.
4. **Compute call**: `reduce<PoolType::SUM, ReduceDim::REDUCE_COL, WaitAndPopPerTile>(...)`.
5. **Post-reduce op**: For softmax, a `recip_tile` callback can be added to compute `1/sum(exp(x))` inline.
6. **Accumulation**: For block-wise softmax where the full height doesn't fit in one pass, the `Accumulate` type can be used with an accumulator CB.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does reduce_tile work in the compute kernel API? What are its parameters? How does REDUCE_COL differ from REDUCE_ROW?"
   **Reason**: Needed to understand the fundamental LLK primitive used for reduction.
   **Key Findings**: `reduce_tile(icb, icb_scaler, itile, itile_scaler, idst)` accumulates the input tile into DST[idst]. REDUCE_COL accumulates along the height (rows), so for a given column position, values from all rows are combined. REDUCE_ROW accumulates along width. The function requires prior `reduce_init` and subsequent `reduce_uninit`.

2. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: Needed to understand the core distribution strategy used in the program factory.
   **Key Findings**: Returns `(num_cores, all_cores, core_group_1, core_group_2, units_per_group_1, units_per_group_2)`. Group 1 gets `ceil(total/cores)` units, group 2 gets `floor(total/cores)`. Group 2 is empty if evenly divisible.

3. **Query**: "What is the tile_regs_acquire/commit/wait/release cycle? How do DST registers work for accumulation?"
   **Reason**: Needed to understand how the compute pipeline manages DST registers during reduction.
   **Key Findings**: `acquire` zeroes and locks DST for math. `commit` hands to packer. `wait` ensures packer waits for math completion. `release` frees DST. DST capacity depends on FP32 mode (8 vs 16 in full-sync) and sync mode (half vs full). The acquire-commit-wait-release cycle is the fundamental synchronization mechanism between math and pack stages.

4. **Query**: "What is the role of the scaler tile in reduction operations? How is it prepared for REDUCE_COL?"
   **Reason**: Needed to understand scaler tile format and usage.
   **Key Findings**: The scaler tile is a full tile with only row 0 of each face populated. For SUM, scaler is 1.0. For AVG, scaler is 1/N. The scaler is multiplied during each `reduce_tile` call. Format is typically Float16_b (or Float32 matching input format on non-Blackhole).

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Contains the unified `reduce()` function that the compute kernel calls.
   **Key Information**: The REDUCE_COL path uses chunking by DEST_AUTO_LIMIT. Within a chunk, for each height row, reduce_tile is called for each column in the chunk, with dst_idx incrementing per column. This explains the reader's tile delivery order. The WaitAndPopPerTile policy waits/pops each tile individually.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Contains DEST_AUTO_LIMIT auto-detection logic.
   **Key Information**: DEST_AUTO_LIMIT is computed from DST_SYNC_MODE and DST_ACCUM_MODE at compile time, shared between reader and compute via matching defines.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.inl`
   **Reason**: Contains `prepare_reduce_scaler` which constructs the scaler tile.
   **Key Information**: Scaler preparation involves zeroing all faces, then filling row 0 of each face. For bfloat16, two bf16 values are packed into each uint32. The function is templated on CB ID and auto-detects data format.

4. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
   **Reason**: Contains `get_defines` which maps host-level enums to LLK defines.
   **Key Information**: `ReduceOpDim::H` maps to `"ReduceDim::REDUCE_COL"`. `ReduceOpMath::SUM` maps to `"PoolType::SUM"`. The defines `REDUCE_OP` and `REDUCE_DIM` are injected into the compute kernel.

5. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Contains the hardware initialization function called at compute kernel start.
   **Key Information**: `compute_kernel_hw_startup(icb0, icb1, ocb)` configures all three compute RISC-V cores: UNPACK (llk_unpack_hw_configure), MATH (llk_math_pack_sync_init, llk_math_hw_configure), PACK (llk_pack_hw_configure, llk_pack_init). The 3-arg version takes input CB A, input CB B (scaler), and output CB.

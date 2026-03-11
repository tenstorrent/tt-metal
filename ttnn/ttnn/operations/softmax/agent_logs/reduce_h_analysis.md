# Reduce H (Height/Column Reduction) Implementation Analysis

## Overview

The **reduce_op_multi_core_h** operation performs height-wise reduction (REDUCE_COL) on a 4D tiled tensor. For each tile-column position across the width (W) and batch (NC) dimensions, it reduces all Ht tile-rows down to a single output tile. For example, reducing a `[1, 1, 64, 128]` tensor along H yields `[1, 1, 32, 128]` (one tile-row of output).

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp`

**Key design insight**: Height reduction is fundamentally different from width reduction. In width reduction, one row of tiles is consumed to produce one output tile (sequential accumulation). In height reduction, each column position must accumulate across all rows. Because the hardware has a limited number of DST registers, column positions are processed in **chunks** -- groups of tile-columns that fit in DST simultaneously. Within each chunk, tiles arrive in an interleaved order (all columns in the chunk for row 0, then all for row 1, etc.) so that multiple column-wise accumulators run in parallel inside DST.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-column |
| **Unit size** | 1 tile-column = Ht input tiles producing 1 output tile |
| **Total units** | `NC * Wt` (batch-count times tile-width) |
| **Loop structure** | Outer: iterate over columns in chunks of `row_chunk`. Inner: for each chunk, iterate `Ht` rows, reading `chunk_size` tiles per row. |

One "column" is a vertical strip of `Ht` tiles at a fixed (batch, w) position. All Ht tiles in the column are reduced (summed, maxed, or averaged) to produce one output tile.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] (4D) |
| **Dimension convention** | NCHW (shape[0]=N, shape[1]=C, shape[2]=H, shape[3]=W) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or WIDTH_SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | bfloat16, float32, or other tile-compatible types |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, 32, W] (height reduced to one tile-row) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or WIDTH_SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (sharded) |
| **Data type** | Configurable via `output_dtype` |

### Layout Transformations

No explicit tilize/untilize within this operation. Input must already be in TILE_LAYOUT. The host-side `reduce()` wrapper in `reduce_op.cpp` calls `tilize_with_val_padding` if needed before dispatching to this program factory.

## Data Flow Pattern

### Standard (Non-Negate) Path

The standard path uses the `compute_kernel_lib::reduce<>` helper library, which encapsulates the entire compute loop. The data flows as follows:

1. **Reader** (dataflow kernel) prepares the scaler tile in CB2, then reads input tiles from DRAM in a **chunked column-interleaved order**: for each chunk of `row_chunk` consecutive tile-columns, it reads all tiles row-by-row with columns interleaved.

2. **Compute** kernel calls `compute_kernel_hw_startup(c_0, c_2, c_3)` then delegates entirely to `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(c_0, c_2, c_3, ReduceInputBlockShape::of(Ht, Wt, NC))`.

3. **Writer** reads output tiles from CB3 one at a time and writes them to DRAM sequentially.

### Negate Path

The negate path (`reduce_h_neg.cpp`) is used for `reduce_min` (computing min as `-max(-x)`). It does NOT use the library helper but implements the loop manually with explicit CB management for intermediate negation buffers (CB4, CB5). Data flow:

1. Reader fills CB0 (same as standard).
2. Compute: for each chunk, for each row:
   - Read `ntiles` from CB0, negate them, pack to CB5 (ineg).
   - If not the first row, reload accumulator from CB4.
   - Reduce from CB5 into DST, accumulating across rows.
   - Pack partial results to CB4 (accumulator).
3. After all rows in a chunk, negate the final accumulated result and pack to CB3 (output).

### Tile Reading Order (Critical for Height Reduction)

The reader sends tiles in **N C W_skip H W_chunk** order. This means:

```
For Ht=3, Wt=4, row_chunk=2:
  Chunk 1: (0,0) (0,1) (1,0) (1,1) (2,0) (2,1)
  Chunk 2: (0,2) (0,3) (1,2) (1,3) (2,2) (2,3)
```

Within each chunk, tiles from the same row but different columns arrive consecutively, then we move to the next row. This allows the compute kernel to use separate DST register indices for each column in the chunk, accumulating the reduction for all columns in parallel.

**Contrast with width reduction**: In width reduction (REDUCE_ROW), tiles arrive in natural row-major order -- all Wt tiles of a row arrive, get reduced to one output, then the next row. There is no chunking concern because only one DST slot is needed per row.

## Circular Buffer Configuration

### Standard (Non-Negate) Path

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler constant tile | 1 tile | 1 tile | Single | Reader | Compute | Program (entire kernel) |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per output tile) |

**Note on CB0/CB3 capacity**: When `negate=false`, both are sized to 2 tiles (double-buffered). The compute library uses `WaitAndPopPerTile` policy, consuming one tile at a time, so 2 tiles allows overlap between reader filling one slot and compute consuming the other.

### Negate Path (Additional CBs)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tile staging | chunk_size tiles | chunk_size tiles | Single | Reader | Compute | Block (per row in chunk) |
| c_2 | cb_scaler | Scaler constant tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_output | Output tile staging | chunk_size tiles | chunk_size tiles | Single | Compute | Writer | Block (per chunk) |
| c_4 | cb_acc | Accumulator (persists across rows) | chunk_size tiles | chunk_size tiles | Single | Compute | Compute | Row (reused across Ht iterations) |
| c_5 | cb_ineg | Negated input intermediate | chunk_size tiles | chunk_size tiles | Single | Compute | Compute | Block (per row iteration) |

### Scaler CB (c_2) Setup

The scaler tile is prepared by the reader kernel using `dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)`. This:
1. Calls `cb_reserve_back(cb_id, 1)`
2. Zeros the entire tile using NoC reads from the hardware zeros region
3. Fills row 0 of each face (4 faces for full tile) with the scaler value encoded in the CB's data format (bfloat16 or float32)
4. Calls `cb_push_back(cb_id, 1)`

The scaler value is passed as a `uint32_t` bit-pattern in compile-time args (host `std::bit_cast<uint32_t>(operation_attributes.scaler)`) and reconstructed to float on device via `__builtin_bit_cast(float, scaler_bits)`.

The scaler CB format decision (line 38-42 of program factory):
- If input is Float32 AND device is NOT Blackhole: scaler format = Float32
- Otherwise: scaler format = Float16_b

### Multi-Pass Data Reuse Patterns

**CB4 (accumulator) persists across rows within a chunk** (negate path only): After processing each row, partial reduction results are packed into CB4. On the next row iteration (`ht > 0`), those tiles are read back from CB4 into DST before the next `reduce_tile` call accumulates the new row's contribution. This is the classic "reload accumulator" pattern. CB4 is both produced and consumed by the compute kernel itself.

**CB2 (scaler) persists for the entire program**: The scaler tile is written once by the reader and remains in the CB for the duration of the kernel. The `reduce_tile` call references the scaler via `itile_scaler=0` (always index 0 in the CB). The library code calls `cb_wait_front(scaler_cb, 1)` once at the start and never pops.

**Standard path (library-based)**: The library's `WaitAndPopPerTile` policy means CB0 tiles do not persist; each tile is waited for, consumed by `reduce_tile`, and popped immediately. The DST registers serve as the implicit accumulator across rows (no explicit accumulator CB needed because the library keeps DST acquired across the entire chunk's row iterations).

## Pipeline Pattern Summary

### Standard Path
- **CB0 (input)**: Double-buffered (capacity=2, block=1). Reader and compute can overlap.
- **CB2 (scaler)**: Single-buffered, persistent. Written once, read many times.
- **CB3 (output)**: Double-buffered (capacity=2, block=1). Compute and writer can overlap.

### Negate Path
- **CB0 (input)**: Single-buffered at chunk granularity (capacity=chunk_size, block=chunk_size).
- **CB4 (acc)**: Single-buffered, acts as explicit accumulator between row passes.
- **CB5 (ineg)**: Single-buffered, transient per-row intermediate.
- **CB3 (output)**: Single-buffered at chunk granularity.

## Compute Kernel Structure: Standard Path (`reduce_h.cpp`)

### Function Signature and Compile-Time Args

```cpp
void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);  // tile-rows in H
    uint32_t Wt = get_compile_time_arg_val(1);  // tile-cols assigned to this core
    uint32_t NC = get_compile_time_arg_val(2);   // always 1 (batches factored into Wt)
```

### Hardware Initialization

```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
```

This three-argument overload configures:
- **Unpack**: `llk_unpack_hw_configure<DST_ACCUM_MODE>(c_0, c_2)` -- sets SRCA=c_0, SRCB=c_2
- **Math**: `llk_math_pack_sync_init`, `llk_math_hw_configure`
- **Pack**: `llk_pack_hw_configure(c_3)`, `llk_pack_init`, `llk_pack_dest_init`

### The reduce<> Library Call

```cpp
compute_kernel_lib::reduce<
    REDUCE_OP,                                          // e.g., PoolType::SUM
    REDUCE_DIM,                                         // ReduceDim::REDUCE_COL (for H reduction)
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    tt::CBIndex::c_0,                                   // input CB
    tt::CBIndex::c_2,                                   // scaler CB
    tt::CBIndex::c_3,                                   // output CB
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
```

### Inside the Library: REDUCE_COL Dispatch (from `reduce_helpers_compute.inl`)

The REDUCE_COL branch (lines 349-453 of the .inl file) implements:

```
chunk_size = DEST_AUTO_LIMIT    // auto-detected from sync/accum mode
for each batch (nc in 0..num_batches):
    for each chunk (wt in 0..Wt, step chunk_size):
        current_chunk = min(chunk_size, Wt - wt)
        tiles_in_chunk = Ht * current_chunk

        tile_regs_acquire()     // acquire DST registers

        for each row (ht in 0..Ht):
            dst_idx = 0         // base DST index for this chunk
            for each col in chunk (i in wt..chunk_end):
                // WaitAndPopPerTile: wait for 1 tile, reduce, pop 1 tile
                cb_wait_front(input_cb, 1)
                reduce_tile<REDUCE_OP, REDUCE_COL>(input_cb, scaler_cb, 0, 0, dst_idx)
                cb_pop_front(input_cb, 1)
                dst_idx++       // next column uses next DST register

        // All Ht rows accumulated into DST[0..current_chunk-1]
        post_reduce_op(...)     // NoOp for standard reduce

        tile_regs_commit()
        tile_regs_wait()
        for each col in chunk:
            cb_reserve_back(output_cb, 1)
            pack_tile(i, output_cb)
            cb_push_back(output_cb, 1)
        tile_regs_release()
```

**Critical observation**: DST registers are acquired ONCE for the entire chunk and held across ALL Ht row iterations. This is what makes height reduction work without an explicit accumulator CB in the standard path -- the DST registers themselves serve as the accumulators. Each `reduce_tile` call accumulates into its designated DST[dst_idx], and because the tile_regs are not released between rows, the accumulated values persist.

### The `reduce_tile` Function

```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation>
void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst);
```

Parameters:
- `icb`: Input CB (c_0)
- `icb_scaler`: Scaler CB (c_2)
- `itile`: Index of tile within input CB (0 for WaitAndPopPerTile, since only 1 tile is ever present)
- `itile_scaler`: Index of scaler tile (always 0)
- `idst`: DST register index to accumulate into (0..chunk_size-1 for different columns)

For **REDUCE_COL**: The hardware reduces each column of the input tile (summing/maxing the 32 rows within each face pair), producing a result that goes into row 0 of the DST tile at index `idst`. Successive calls with the same `idst` accumulate (add/max) into that DST register.

### The `reduce_init` / `reduce_uninit` Pair

```cpp
reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(input_cb, scaler_cb, output_cb);
// ... reduce_tile calls ...
reduce_uninit<enforce_fp32_accumulation>();
```

`reduce_init` configures:
- Unpack: `llk_unpack_AB_reduce_init` (sets up SRCA/SRCB for reduce)
- Math: `llk_math_reduce_init` (configures math engine for reduce op)
- Pack: `llk_pack_reduce_mask_config` (sets edge masks so only reduced rows/cols are packed)

`reduce_uninit` clears the packer edge mask so subsequent operations pack correctly.

## Compute Kernel Structure: Negate Path (`reduce_h_neg.cpp`)

This kernel performs `-max(-x)` to compute `min(x)`. It manages all CB interactions manually (no library helper).

### Key Differences from Standard Path

1. **Explicit negation stages**: Uses `copy_tile` + `negative_tile` (SFPU operations) to negate input tiles before and after reduction.

2. **Explicit accumulator CB (c_4)**: Unlike the standard path where DST persists across rows, the negate path must release DST between negation and reduction phases. So it uses CB4 as an explicit accumulator:
   - After each row's reduction, results are packed from DST to CB4
   - Before the next row, they are reloaded from CB4 to DST via `copy_tile`

3. **Intermediate negation CB (c_5)**: Negated input tiles are stored in CB5 so they can be consumed by the reduce phase (which requires them in SRCA format, not in DST).

4. **`row_chunk` from DEST_AUTO_LIMIT**: Same auto-detection as the standard path.

### Detailed Per-Row Flow (Negate Path)

```
for each row ht in 0..Ht:
    // Phase 1: Negate input tiles
    tile_regs_acquire()
    cb_wait_front(cb_input, ntiles)
    copy_tile(cb_input, i, i)       // input[i] -> DST[i]
    negative_tile(i)                 // DST[i] = -DST[i]
    tile_regs_commit()
    cb_pop_front(cb_input, ntiles)
    cb_reserve_back(cb_ineg, ntiles)
    tile_regs_wait()
    pack_tile(i, cb_ineg)           // DST[i] -> CB5
    tile_regs_release()
    cb_push_back(cb_ineg, ntiles)

    // Phase 2: Reduce (accumulate) negated tiles
    tile_regs_acquire()
    if (ht > 0):
        cb_wait_front(cb_acc, ntiles)
        copy_tile(cb_acc, i, i)     // reload accumulator from CB4 -> DST[i]
    cb_wait_front(cb_ineg, ntiles)
    reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc)
    reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, i, 0, i)
    reduce_uninit(cb_ineg)
    tile_regs_commit()
    cb_pop_front(cb_ineg, ntiles)
    if (ht > 0): cb_pop_front(cb_acc, ntiles)
    cb_reserve_back(cb_acc, ntiles)
    tile_regs_wait()
    pack_tile(i, cb_acc)            // DST[i] -> CB4 (accumulator)
    tile_regs_release()
    cb_push_back(cb_acc, ntiles)

// After all rows: Final negation of accumulated result
tile_regs_acquire()
cb_wait_front(cb_acc, ntiles)
copy_tile(cb_acc, i, i)            // CB4 -> DST
negative_tile(i)                    // DST = -DST (undo the initial negation)
tile_regs_commit()
cb_pop_front(cb_acc, ntiles)
cb_reserve_back(cb_output, ntiles)
tile_regs_wait()
pack_tile(i, cb_output)            // DST -> CB3 (output)
tile_regs_release()
cb_push_back(cb_output, ntiles)
```

## DEST Register Capacity and Chunk Size

The chunk size (number of tile-columns processed simultaneously) is determined by DEST register capacity, auto-detected at compile time via `compute_kernel_lib::DEST_AUTO_LIMIT` (from `dest_helpers.hpp`):

| Sync Mode | Accum Mode | DEST Tiles | Chunk Size |
|-----------|------------|------------|------------|
| SyncHalf  | 16-bit (fp16) | 8 | 8 |
| SyncHalf  | 32-bit (fp32) | 4 | 4 |
| SyncFull  | 16-bit (fp16) | 16 | 16 |
| SyncFull  | 32-bit (fp32) | 8 | 8 |

The host-side program factory computes the same value via `ttnn::get_dest_reg_count(compute_kernel_config)` for CB sizing (negate path). For the standard path, the host simply uses capacity=2 (double-buffered) since the library consumes tiles one at a time.

Both the reader and compute kernels auto-detect chunk size independently using the same `DEST_AUTO_LIMIT` constant, ensuring they agree on the tile ordering without an explicit host parameter.

## Index Calculations

### Reader Kernel Index Mapping

The reader (`reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`) maps logical (row, col) positions to physical tile IDs using:

**Initial position** (from runtime args):
- `col_start_tile_id`: Starting tile in the flattened tile array. Computed by host as `(num_cols_read / Wt * HtWt) + (num_cols_read % Wt)`.
- `curr_col_in_batch`: Column position within the current batch, `num_cols_read % Wt`.

**Traversal** (column-interleaved chunked order):
```
for each chunk (i in 0..num_cols, step row_chunk):
    curr_id = col_start_tile_id  // start of this chunk's first column
    for each row (j in 0..Ht):
        for each col in chunk (k in i..chunk_end):
            read tile at curr_id
            w++; curr_id++
            if (w == Wt):  // crossed batch boundary
                col_start_tile_id = curr_id + (Ht - j - 1) * Wt + 1
                curr_id = col_start_tile_id + j * Wt
                w = 0
        curr_id = reset_curr_id + (j+1) * Wt  // stride to next row
```

The key formula for moving between rows: `curr_id = reset_curr_id + (j+1) * Wt`. Since tiles are stored in row-major order within each batch (tile at position (h,w) is at base + h*Wt + w), advancing by Wt moves to the same column in the next row.

### Batch Boundary Handling

When a chunk spans a batch boundary (w reaches Wt), the reader adjusts both `col_start_tile_id` and `curr_id` to jump to the first column of the next batch at the correct row.

## Memory Access Patterns

### Read Pattern (Interleaved Mode)
- **Order**: Column-interleaved chunked. Not purely sequential or purely strided.
- **Within a chunk row**: Sequential tile reads (adjacent columns).
- **Between rows**: Stride of Wt tiles (skip to same column in next row).
- **Between chunks**: Skip to next set of columns.
- **One tile at a time**: Each `noc.async_read` fetches a single tile, followed by `noc.async_read_barrier` (blocking).

### Write Pattern (Interleaved Mode)
- **Order**: Sequential. Output tiles are written in the order they are produced (column-major within each batch, left to right).
- **One tile at a time**: `noc_async_write_page` per tile with `noc_async_writes_flushed` barrier.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-wise by default) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` |
| **Total cores** | `min(num_cols, max_cores)` where `num_cols = NC * Wt` |
| **Work per core** | `num_cols / num_cores` tile-columns (core_group_1 gets +1 if remainder) |
| **Load balancing** | Two-group: core_group_1 gets `ceil(num_cols/num_cores)` cols, core_group_2 gets `floor(num_cols/num_cores)` cols |

### Work Splitting Details

Work is split across tile-columns (not tile-rows). Each core processes a contiguous range of tile-columns across all batches. For each assigned column, the core reads all Ht rows, reduces them, and writes one output tile.

The compute kernel receives the same `Ht` for all cores but different `Wt` values:
- core_group_1: `Wt = num_cols_per_core_group_1`
- core_group_2: `Wt = num_cols_per_core_group_2`
- `NC = 1` always (the batch dimension is folded into the column count)

### Width-Sharded Special Case

When both input and output are WIDTH_SHARDED, the core grid is taken from the shard spec directly. Each core processes the columns in its shard. `num_cols_per_core = NC * (shard_width / TILE_WIDTH)`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows in height dimension |
| 1 | Wt | uint32_t | Number of tile-columns in width dimension (global) |
| 2 | HtWt | uint32_t | Total tiles per batch (Ht * Wt) |
| 3 | scaler_bits | uint32_t | Scaler value as bit-cast uint32_t |
| 4+ | TensorAccessorArgs | varies | Tensor accessor configuration for DRAM reads |

Reader also receives defines:
- `ENABLE_FP32_DEST_ACC`: "0" or "1" (controls DEST_AUTO_LIMIT in reader)
- `DST_SYNC_FULL`: "0" or "1" (controls DEST_AUTO_LIMIT in reader)

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows to reduce over |
| 1 | Wt | uint32_t | Number of tile-columns assigned to this core |
| 2 | NC | uint32_t | Always 1 (batches folded into Wt) |

Compute also receives defines:
- `REDUCE_OP`: e.g., `PoolType::SUM`, `PoolType::MAX`, `PoolType::AVG`
- `REDUCE_DIM`: `ReduceDim::REDUCE_COL` (height reduction maps to REDUCE_COL in hardware terminology)

**Naming convention note**: The program factory is called "reduce_h" (reduce height), but the hardware enum is `REDUCE_COL` because reducing along H means collapsing columns of tiles.

#### Writer Kernel (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_3) |
| 1+ | TensorAccessorArgs | varies | Tensor accessor configuration for DRAM writes |

### Runtime Arguments

#### Reader Kernel (Interleaved, per-core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM address |
| 1 | col_start_tile_id | uint32_t | Starting tile ID: `(num_cols_read / Wt * HtWt) + (num_cols_read % Wt)` |
| 2 | curr_col_in_batch | uint32_t | Starting column within batch: `num_cols_read % Wt` |
| 3 | num_cols | uint32_t | Number of tile-columns this core processes |

#### Writer Kernel (Interleaved, per-core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | num_pages | uint32_t | Number of output tiles to write (= num_cols for this core) |
| 2 | start_id | uint32_t | Output tile start index (= num_cols_read) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (src_addr) | CB0 (input), CB2 (scaler) | Prepare scaler tile, read input tiles in chunked column order |
| compute | RISCV_2 (unpack+math+pack) | N/A | CB0, CB2 | CB3 | reduce_tile accumulation across Ht rows per column-chunk |
| writer | RISCV_1 | NOC1 | CB3 | DRAM (dst_addr) | Sequential write of output tiles |

### Reader Kernel Details

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`

**Key Logic**:
- Uses `dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)` to fill the scaler CB once
- Uses `TensorAccessor` for address calculation with `noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0})`
- Handles batch boundary crossings when `w == Wt`
- Row stride is `Wt` tiles (advancing `curr_id` by `Wt` moves to next row, same column)

### Compute Kernel Details (Standard)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp`

**Key Logic**: Single call to `compute_kernel_lib::reduce<>` with `WaitAndPopPerTile` policy and `ReduceDataFormatReconfigMode::NONE` (since reduce is the first and only operation after hw_startup).

### Compute Kernel Details (Negate)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h_neg.cpp`

**Key Logic**:
- Manual triple-phase loop per row: negate -> reduce-accumulate -> store-accumulator
- Uses `copy_tile_init`, `copy_tile`, `negative_tile_init`, `negative_tile` for negation
- Uses `reduce_init`/`reduce_tile`/`reduce_uninit` for reduction (must init/uninit per row because of intervening copy operations that corrupt SRCA config)
- Uses `reconfig_data_format_srca` and `pack_reconfig_data_format` when switching between CB sources

### Writer Kernel Details

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**: Generic tile writer. Loops `num_pages` times, calling `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` for each tile. Uses `TensorAccessor` for address mapping.

## How Height Reduction Differs from Width Reduction

| Aspect | Width Reduction (REDUCE_ROW) | Height Reduction (REDUCE_COL) |
|--------|------------------------------|-------------------------------|
| **Tile arrival order** | Row-major: all Wt tiles of row arrive, then next row | Chunked column-interleaved: chunk_size cols x Ht rows |
| **DST usage** | 1 DST register per row (accumulate Wt tiles) | chunk_size DST registers (one per column, accumulate Ht rows) |
| **Chunking needed** | No (only 1 DST register needed) | Yes (limited DST registers) |
| **Reader complexity** | Simple sequential reads | Complex strided reads with batch boundary handling |
| **Output order** | 1 tile per Wt input tiles | chunk_size tiles per (Ht * chunk_size) input tiles |
| **Hardware reduce_dim** | REDUCE_ROW (sum across columns within tile) | REDUCE_COL (sum across rows within tile) |

## Implementation Notes

1. **The `NC=1` trick**: The program factory always passes `NC=1` to the compute kernel, with the actual batch count folded into the `Wt` parameter (`num_cols_per_core = NC * Wt / num_cores`). This works because from the compute kernel's perspective, processing columns across batch boundaries is identical to processing more columns within a single batch.

2. **Data format for scaler**: The scaler CB uses Float16_b by default, switching to Float32 only when the input is Float32 and the device is not Blackhole. This ensures compatibility with the reduce hardware's expectations.

3. **FP32 dest accumulation**: Controlled by `ComputeConfig.fp32_dest_acc_en`. When enabled, DST registers use 32-bit precision but capacity is halved (affecting chunk size).

4. **Width-sharded path**: Uses a different reader kernel (`reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`) and writer (`writer_unary_sharded.cpp`). The sharded reader maps CB1 directly to the input buffer's L1 address via `set_globally_allocated_address`. Similarly, the output CB is mapped to the output buffer. This avoids DRAM reads/writes entirely.

5. **Relevance to softmax**: For a softmax operation computing `exp(x_i) / sum(exp(x_j))` along dim=-2 (height), the height reduction pattern here (REDUCE_COL with chunked column processing) is directly applicable for the `sum(exp(x_j))` step. The `post_reduce_op` lambda parameter of the library's `reduce<>` function can be used to apply `recip_tile` immediately after the sum, producing `1/sum` in-place. The `WaitUpfrontNoPop` or `NoWaitNoPop` input policies would allow the exp tiles to remain in the CB for subsequent broadcast-multiply with the reciprocal.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does split_work_to_cores work in tt-metal? How are cores distributed for multi-core operations and what are core_group_1 vs core_group_2?"
   **Reason**: Understanding how work is distributed across cores for the multi-core program factory.
   **Key Findings**: `split_work_to_cores` divides `num_cols` across available cores. When not evenly divisible, core_group_1 gets `ceil(N/cores)` units and core_group_2 gets `floor(N/cores)`. Returns both groups plus their per-core work counts.

2. **Query**: "How does reduce_tile work in the TT-Metal compute API for REDUCE_COL?" (query failed)
   **Reason**: Understanding the hardware-level reduce operation.
   **Key Findings**: Retrieved from source code instead. `reduce_tile` unpacks input tile and scaler into SRCA/SRCB, then the math engine performs the reduction, accumulating into DST[idst].

3. **Query**: "What is compute_kernel_hw_startup?" (query failed)
   **Reason**: Understanding the initialization sequence.
   **Key Findings**: Retrieved from `compute_kernel_hw_startup.h`. Configures unpack, math sync, math hw, pack hw, pack init, and pack dest init for the three specified CBs.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Understanding reduce_tile and reduce_init API signatures and semantics.
   **Key Information**: `reduce_tile(icb, icb_scaler, itile, itile_scaler, idst)` performs reduction of tile from icb using scaler from icb_scaler, accumulating into DST[idst]. Scaler must have row 0 of each face filled. `reduce_uninit` must be called to clear packer edge masks before other operations.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Understanding the library-level reduce function that the standard compute kernel delegates to.
   **Key Information**: The `reduce<>` template handles all three reduce dimensions. For REDUCE_COL, it auto-chunks by DEST_AUTO_LIMIT, iterates rows within each chunk, and uses DST registers as implicit accumulators. Supports configurable input policies and post-reduce operations.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST_AUTO_LIMIT and chunk size determination.
   **Key Information**: DEST capacity depends on sync mode (half/full) and accumulation mode (16/32 bit). `DEST_AUTO_LIMIT = get_dest_limit()` is a compile-time constant available to both dataflow and compute kernels.

4. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp` (lines 20-42)
   **Reason**: Understanding the define mapping from high-level enums to kernel-level constants.
   **Key Information**: `ReduceOpDim::H` maps to `ReduceDim::REDUCE_COL`. `ReduceOpMath::SUM` maps to `PoolType::SUM`. The `get_defines` function produces the `REDUCE_OP` and `REDUCE_DIM` defines passed to the compute kernel.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.inl`
   **Reason**: Understanding scaler tile preparation.
   **Key Information**: `prepare_reduce_scaler<cb_id>(scaler_f)` zeros the entire tile via NoC reads from hardware zeros, then fills row 0 of each face with the scaler value. Format is auto-detected from the CB's data format.

6. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Understanding the hardware initialization sequence.
   **Key Information**: The 3-arg overload configures unpack (SRCA=icb0, SRCB=icb1), math engine, and packer for the specified CBs. Must be called before any compute operations.

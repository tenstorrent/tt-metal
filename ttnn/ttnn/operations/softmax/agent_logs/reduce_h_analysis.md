# Reduce H (Height-Dimension Reduction) Implementation Analysis

## Overview

This analysis documents the **reduce_op_multi_core_h** program factory, which performs
height-dimension (dim=-2) reduction on tiled tensors. It reduces all Ht tile-rows in
each tile-column to a single output tile per column. Supported math operations are SUM,
AVG, and MAX.

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp`

**Focus**: Compute kernel structure, CB layout, scalar/constant setup, how tiles along
the H dimension are accumulated, and how the `reduce` helper library orchestrates the
reduction.

There are two code paths in the compute kernel:

1. **Standard path** (`reduce_h.cpp`): Uses the `compute_kernel_lib::reduce<>` helper
   library, which encapsulates all DST register management, CB synchronization, and
   tile-by-tile accumulation.
2. **Negate path** (`reduce_h_neg.cpp`): Manually implements the reduction loop with
   explicit `negative_tile`, `copy_tile`, `reduce_tile`, `pack_tile` calls and an
   intermediate accumulator CB. Used for `reduce_min = -reduce_max(-x)`.

Both paths ultimately call `reduce_tile<REDUCE_OP, REDUCE_DIM>(...)` in a loop over
Ht, accumulating into DST registers.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile column (one Wt-position across all Ht rows) |
| **Unit size** | Ht input tiles -> 1 output tile |
| **Total units** | NC * Wt (batches x width-tiles) |
| **Loop structure** | Outer: batches (NC). Middle: column chunks (Wt / chunk_size). Inner: Ht rows per column |

One "work unit" is the complete reduction of one tile-column: reading Ht tiles that
share the same (batch, column) position, reducing them into a single output tile.
Work is distributed across cores as groups of these columns.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N*C, H, W] (treated as [NC, Ht*32, Wt*32]) |
| **Dimension convention** | NC x H x W (batch dims collapsed into NC) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or WIDTH_SHARDED |
| **Buffer type** | DRAM (interleaved) or L1 (width-sharded) |
| **Data type** | Configurable (bfloat16, float32, etc.) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N*C, 1, W] (padded to [NC, 32, Wt*32]) |
| **Dimension convention** | NC x 1 x W (H dimension reduced to 1 tile-row) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | Same as input (INTERLEAVED or WIDTH_SHARDED) |
| **Buffer type** | Same as input |
| **Data type** | Configurable (may differ from input) |

### Layout Transformations

No explicit tilize/untilize. Both input and output are in TILE_LAYOUT. The reduction
collapses H from Ht tile-rows to 1 tile-row; width and batch dimensions are preserved.

---

## Data Flow Pattern

### Standard Path (reduce_h.cpp)

1. **Reader kernel** prepares the **scaler tile** in CB c_2 by calling
   `dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f)`. This fills
   row 0 of each face with the scaler value (bit-cast from host-provided float).

2. **Reader kernel** fetches input tiles from DRAM/L1 in a specific **column-chunked**
   order and pushes them one-at-a-time into CB c_0. The order is:
   ```
   For each chunk of row_chunk columns:
     For each height row ht in [0, Ht):
       For each column wt in chunk:
         read tile at (ht, wt) -> push to CB c_0
   ```
   This matches the `REDUCE_COL` + `WaitAndPopPerTile` consumption pattern expected
   by the compute kernel.

3. **Compute kernel** calls `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(...)`.
   Inside the library:
   - For each batch (NC):
     - For each chunk of `chunk_size` columns (where chunk_size = DEST_AUTO_LIMIT):
       - `tile_regs_acquire()` -- acquire DST registers
       - For each height row ht in [0, Ht):
         - For each column in chunk:
           - `cb_wait_front(input_cb, 1)` -- wait for 1 tile in CB c_0
           - `reduce_tile<REDUCE_OP, REDUCE_COL>(input_cb, scaler_cb, 0, 0, dst_idx)` -- reduce into DST[dst_idx]
           - `cb_pop_front(input_cb, 1)` -- free the tile from CB c_0
           - `dst_idx++` (each column in the chunk has its own DST register)
       - `tile_regs_commit()` -- signal math done
       - `tile_regs_wait()` -- wait for pack
       - For each column in chunk:
         - `cb_reserve_back(output_cb, 1)` + `pack_tile(dst_idx, output_cb)` + `cb_push_back(output_cb, 1)`
       - `tile_regs_release()` -- release DST registers

4. **Writer kernel** reads from CB c_3 and writes output tiles to DRAM/L1.

### Key Insight: reduce_tile Accumulation

`reduce_tile<PoolType, REDUCE_COL>()` is an **accumulating** operation. When called
multiple times with the same `dst_idx`, the hardware adds (or maxes) the new tile's
contribution into the existing value in `DST[dst_idx]`. This is what enables the
height reduction: calling `reduce_tile` once per height-row for the same column
accumulates all Ht contributions into a single DST register.

For `REDUCE_COL` specifically, the hardware reduces each input tile's columns (rows
within the tile are collapsed) and accumulates the result column-wise into the DST
register. After processing all Ht tiles for a column, `DST[dst_idx]` holds the
fully reduced result.

---

## Circular Buffer Configuration

### Standard Path (non-negate, interleaved)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_2 | cb_scaler | Reduce scaler constant | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block |

### Negate Path (interleaved)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | chunk_size tiles | chunk_size tiles | Single | Reader | Compute | Block |
| c_2 | cb_scaler | Reduce scaler constant | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_output | Output tile staging | chunk_size tiles | chunk_size tiles | Single | Compute | Writer | Block |
| c_4 | cb_acc | Reduction accumulator | chunk_size tiles | chunk_size tiles | Single | Compute | Compute | Row (across Ht iterations) |
| c_5 | cb_ineg | Negated input intermediate | chunk_size tiles | chunk_size tiles | Single | Compute | Compute | Block |

### Width-Sharded Path

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_shard | Shard buffer (globally allocated) | num_shard_tiles | 1 tile | N/A (mapped to tensor) | N/A | Reader | Program |
| c_2 | cb_scaler | Reduce scaler constant | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_output | Output (globally allocated) | num_output_tiles | 1 tile | N/A (mapped to tensor) | Compute | Writer | Program |

### Notes on CB Sizing

- **chunk_size** = `DEST_AUTO_LIMIT` = number of DST registers available, which
  depends on sync mode and accumulation mode:
  - Half-sync + bfloat16: 8 tiles
  - Half-sync + fp32: 4 tiles
  - Full-sync + bfloat16: 16 tiles
  - Full-sync + fp32: 8 tiles
- In the standard path, CB c_0 and c_3 are sized to 2 tiles (double-buffered) for
  the `WaitAndPopPerTile` policy.
- In the negate path, CB c_0 and c_3 are sized to `chunk_size` tiles because the
  negate kernel processes an entire chunk of tiles at once.

---

## Pipeline Pattern Summary

### Standard Path
- **CB c_0 (input)**: Double-buffered (2 tiles, block=1). Reader can push next tile
  while compute processes current tile.
- **CB c_2 (scaler)**: Single-buffered, written once at startup, read throughout.
- **CB c_3 (output)**: Double-buffered (2 tiles, block=1). Compute can push next
  result while writer sends current result.

### Negate Path
- All CBs are single-buffered at chunk_size. No overlap between reader and compute
  within a chunk, but the chunk-level granularity allows processing multiple columns
  in parallel within DST registers.

---

## Scaler CB Setup (Critical Pattern for Reuse)

The **scaler tile** in CB c_2 is prepared by the reader kernel (not the compute
kernel) and persists for the entire program execution. The setup sequence is:

1. **Host side** (program factory, line 131):
   ```cpp
   uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
   ```
   The float scaler is bit-cast to uint32_t and passed as a compile-time arg to the
   reader kernel.

2. **Reader kernel** (dataflow, line 34-35):
   ```cpp
   float scaler_f = __builtin_bit_cast(float, scaler_bits);
   dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f);
   ```

3. **`prepare_reduce_scaler`** implementation:
   - `cb_reserve_back(cb_id, 1)` -- reserve space for one tile
   - `zero_faces<data_format, half_tile>(write_addr)` -- zero the entire tile
   - `fill_row0<data_format, half_tile>(ptr, scaler)` -- fill row 0 of each face
     with the scaler value (in bfloat16: two bf16 values packed per uint32;
     in float32: one float per uint32)
   - `cb_push_back(cb_id, 1)` -- make the tile available

4. **Compute kernel** either:
   - Implicitly waits via the `reduce` library: `cb_wait_front(scaler_cb, 1)` at
     the start of `reduce()`, or
   - Explicitly waits: `cb_scaler_obj.wait_front(1)` in the negate path.

The scaler tile is never popped; it stays in CB c_2 for the entire kernel execution.
The `reduce_tile` instruction reads from it via indexed access (scaler_idx = 0).

### Scaler Data Format Selection (program factory, lines 38-42)

```cpp
tt::DataFormat scaler_cb_data_format =
    (src0_cb_data_format == tt::DataFormat::Float32 && a.device()->arch() != tt::ARCH::BLACKHOLE)
        ? tt::DataFormat::Float32
        : tt::DataFormat::Float16_b;
```

The scaler uses Float16_b by default, upgrading to Float32 only when the input is
Float32 on non-Blackhole architectures. This matches the `reduce_tile` LLK's
expectation for the SRCB operand format.

---

## How Height-Dimension Reduction Works Tile-by-Tile

### The REDUCE_COL Semantic

When the program factory calls `get_defines(math_op, ReduceOpDim::H)`, the defines map
produces `REDUCE_DIM = ReduceDim::REDUCE_COL`. This naming reflects the LLK convention:
- **ReduceOpDim::H** (host-side "reduce height") maps to **ReduceDim::REDUCE_COL**
  (LLK "reduce columns", meaning collapse rows within each column).

### Standard Path: Detailed Tile Flow

Given a tensor with shape [NC, Ht, Wt] in tiles, and chunk_size = DEST_AUTO_LIMIT:

```
For nc = 0 to NC-1:
  For wt = 0 to Wt-1 step chunk_size:
    current_chunk = min(chunk_size, Wt - wt)

    tile_regs_acquire()   // Lock DST registers

    For ht = 0 to Ht-1:
      For i = 0 to current_chunk-1:
        cb_wait_front(c_0, 1)                          // Wait for tile from reader
        reduce_tile<OP, REDUCE_COL>(c_0, c_2, 0, 0, i) // Reduce into DST[i]
        cb_pop_front(c_0, 1)                            // Free input tile

    // At this point, DST[0..current_chunk-1] each contain the fully
    // reduced result for their respective column

    tile_regs_commit()
    tile_regs_wait()

    For i = 0 to current_chunk-1:
      cb_reserve_back(c_3, 1)
      pack_tile(i, c_3)    // Pack DST[i] to output CB
      cb_push_back(c_3, 1)

    tile_regs_release()
```

**Key observations**:
- Each column in the chunk gets its own DST register index (0 through current_chunk-1).
- `reduce_tile` accumulates across height rows: the first call (ht=0) initializes
  DST[i], subsequent calls (ht>0) accumulate into the same DST[i].
- The chunk_size bounds how many columns can be reduced simultaneously, limited by
  available DST registers.
- All Ht rows for a chunk are processed within a single `tile_regs_acquire/release`
  block, meaning the accumulation happens entirely in DST registers without
  spilling to L1.

### Negate Path: Detailed Tile Flow

The negate path implements `reduce_min = -reduce_max(-x)`:

```
For nc = 0 to NC-1:
  For wt = 0 to Wt-1 step chunk_size:
    current_chunk = min(chunk_size, Wt - wt)

    For ht = 0 to Ht-1:
      // Phase 1: Negate input tiles
      tile_regs_acquire()
      cb_input.wait_front(current_chunk)
      copy_tile_init(cb_input)
      negative_tile_init()
      for i in chunk: copy_tile(cb_input, i, i); negative_tile(i)
      tile_regs_commit()
      cb_input.pop_front(current_chunk)
      cb_ineg.reserve_back(current_chunk)
      tile_regs_wait()
      for i in chunk: pack_tile(i, cb_ineg)
      tile_regs_release()
      cb_ineg.push_back(current_chunk)

      // Phase 2: Reduce negated tiles, accumulating across Ht
      tile_regs_acquire()
      if ht > 0: cb_acc.wait_front(current_chunk)  // Reload accumulator
      cb_ineg.wait_front(current_chunk)
      if ht > 0:
        copy_tile_init(cb_acc)
        for i in chunk: copy_tile(cb_acc, i, i)     // Load prev accumulator into DST
      reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc)
      for i in chunk: reduce_tile<OP, DIM>(cb_ineg, cb_scaler, i, 0, i)  // Accumulate
      reduce_uninit(cb_ineg)
      tile_regs_commit()
      cb_ineg.pop_front(current_chunk)
      if ht > 0: cb_acc.pop_front(current_chunk)
      cb_acc.reserve_back(current_chunk)
      tile_regs_wait()
      for i in chunk: pack_tile(i, cb_acc)           // Spill accumulator to L1
      tile_regs_release()
      cb_acc.push_back(current_chunk)

    // Phase 3: Final negate of accumulated result
    tile_regs_acquire()
    cb_acc.wait_front(current_chunk)
    copy_tile_init(cb_acc)
    for i in chunk: copy_tile(cb_acc, i, i)
    negative_tile_init()
    for i in chunk: negative_tile(i)
    tile_regs_commit()
    cb_acc.pop_front(current_chunk)
    cb_output.reserve_back(current_chunk)
    tile_regs_wait()
    for i in chunk: pack_tile(i, cb_output)
    tile_regs_release()
    cb_output.push_back(current_chunk)
```

**Key difference from standard path**: The negate path cannot keep the accumulation
purely in DST registers across Ht iterations because it needs to negate each input
tile first (requiring DST space). Instead, it **spills the accumulator to CB c_4
(cb_acc)** after each Ht iteration, then reloads it before the next iteration. The
reload uses `copy_tile` to bring the previous partial result back into DST before
calling `reduce_tile` to accumulate the current row's contribution.

---

## Index Calculations

### Interleaved Reader: Tile Ordering

The reader produces tiles in the order expected by `REDUCE_COL` + `WaitAndPopPerTile`:

```
N C W_skip H W_chunk order
```

This means: for each chunk of `row_chunk` tile-columns, iterate over all Ht rows,
and within each row emit the chunk's tile-columns left to right.

The reader computes tile IDs using the formula:
- `col_start_tile_id` tracks the base of the current column in tile-linear order
- `curr_id = col_start_tile_id + ht * Wt` strides down by one tile-row (Wt tiles apart)
- When a column wraps past the end of a batch (w == Wt), the reader adjusts:
  `col_start_tile_id = curr_id + (Ht - j - 1) * Wt + 1` to jump to the first column
  of the next batch.

### Runtime Args for Interleaved Reader

```cpp
{a.buffer()->address(),
 (num_cols_read / Wt * HtWt) + (num_cols_read % Wt),  // start tile ID
 num_cols_read % Wt,                                    // column within batch
 num_cols_per_core}                                      // total columns this core
```

The start tile ID places us at the correct (batch, column) position in the tile-linear
address space. `num_cols_read % Wt` tells the reader which column within a batch
we start at, enabling correct wrap-around logic for batch boundaries.

---

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Strided access. Within a chunk, the reader traverses tiles in
  column-major order (stepping by Wt tiles between rows). This is NOT sequential
  in memory -- consecutive tiles in a column are Wt tiles apart.
- **Width-sharded**: Local L1 access. The reader copies from the shard buffer (CB c_1,
  globally allocated) to the input CB (c_0) using local NoC reads.

### Write Pattern
- **Interleaved**: Sequential tile writes. Output tiles are produced in order
  (one per column, columns grouped per core), written sequentially.
- **Width-sharded**: Output is directly in the globally-allocated CB c_3 mapped
  to the output tensor's L1 shard.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major linearization of 2D grid) |
| **Grid dimensions** | Up to compute_with_storage_grid_size.x * compute_with_storage_grid_size.y |
| **Total cores** | min(NC * Wt, available cores) |
| **Work per core** | NC * Wt / num_cores columns (with remainder handling) |
| **Load balancing** | Two core groups: group_1 gets ceil columns, group_2 gets floor columns |

Work is split across cores using `split_work_to_cores()`:
- `num_cols = NC * Wt` (total tile-columns across all batches)
- Cores are divided into `core_group_1` (more work) and `core_group_2` (less work)
- Each core processes its assigned columns completely (all Ht rows per column)

For width-sharded inputs, the core grid is taken from the shard spec, and each core
processes the columns present in its shard.

---

## Arguments

### Compute Kernel Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows in the height dimension |
| 1 | Wt (num_cols_per_core_group) | uint32_t | Number of tile-columns this core group processes |
| 2 | NC | uint32_t | Always 1 (batches handled by treating all columns across batches as one flat list) |

Note: NC is always passed as 1 because the program factory flattens the batch and
column dimensions into `num_cols = NC * Wt`, and each core gets a subset of these
columns. The compute kernel sees its columns as `Wt = num_cols_per_core_group` with
`NC = 1`.

### Reader Kernel Compile-Time Arguments (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles |
| 1 | Wt | uint32_t | Width in tiles (full tensor, not per-core) |
| 2 | HtWt | uint32_t | Ht * Wt (tiles per batch) |
| 3 | scaler_bits | uint32_t | Bit-cast float scaler value |
| 4+ | TensorAccessorArgs | ... | Tensor accessor parameters for DRAM reads |

### Reader Kernel Runtime Arguments (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | col_start_tile_id | uint32_t | Starting tile ID in column-major traversal |
| 2 | curr_col_in_batch | uint32_t | Starting column index within the current batch |
| 3 | num_cols | uint32_t | Number of columns this core processes |

### Writer Kernel Compile-Time Arguments (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_3) |
| 1+ | TensorAccessorArgs | ... | Tensor accessor parameters for DRAM writes |

### Writer Kernel Runtime Arguments (Interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Number of output tiles to write |
| 2 | start_id | uint32_t | Starting output tile index |

### Defines (Passed to Compute Kernel)

| Define | Value | Description |
|--------|-------|-------------|
| REDUCE_OP | PoolType::SUM / PoolType::AVG / PoolType::MAX | Reduction operation type |
| REDUCE_DIM | ReduceDim::REDUCE_COL | Always REDUCE_COL for H reduction |

---

## Kernel Implementations

### Compute Kernel: reduce_h.cpp (Standard Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_h | TRISC (unpack+math+pack) | N/A | CB c_0, CB c_2 | CB c_3 | reduce_tile (FPU) |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp`

**Key Logic**:
- Calls `compute_kernel_hw_startup(c_0, c_2, c_3)` to initialize hardware for
  the input, scaler, and output CBs.
- Delegates entirely to `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(...)`.
- The library handles all chunking (via DEST_AUTO_LIMIT), DST register management,
  and CB synchronization internally.
- `ReduceInputBlockShape::of(Ht, Wt, NC)` tells the library the full tile grid dimensions.

**Exact call signature**:
```cpp
compute_kernel_lib::reduce<
    REDUCE_OP,                                          // e.g., PoolType::SUM
    REDUCE_DIM,                                         // ReduceDim::REDUCE_COL
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    tt::CBIndex::c_0,                                   // input CB
    tt::CBIndex::c_2,                                   // scaler CB
    tt::CBIndex::c_3,                                   // output CB
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));  // tile grid shape
```

### Compute Kernel: reduce_h_neg.cpp (Negate Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_h_neg | TRISC | N/A | CB c_0, CB c_2 | CB c_3 | copy_tile, negative_tile, reduce_tile, pack_tile |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h_neg.cpp`

**Key Logic**:
- Uses `DEST_AUTO_LIMIT` for chunk sizing (same as standard path).
- Three-phase per-ht iteration: (1) negate input -> CB c_5, (2) reduce + accumulate -> CB c_4, (3) after all Ht: negate accumulated result -> CB c_3.
- Manual accumulator management: spills to CB c_4 after each Ht iteration, reloads
  via `copy_tile` before next iteration.
- Calls `reduce_init/reduce_uninit` around each reduce phase to manage LLK state.

**Key API calls**:
- `copy_tile(cb_input, i, i)` -- load tile i from input CB into DST[i]
- `negative_tile(i)` -- negate DST[i] in-place (SFPU operation)
- `reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, i, 0, i)` -- reduce tile i from cb_ineg with scaler tile 0, accumulate into DST[i]
- `pack_tile(i, cb_acc)` -- pack DST[i] into accumulator CB

### Reader Kernel (Interleaved, noted briefly)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`

- Provides tiles to CB c_0 in `N C W_skip H W_chunk` order.
- Prepares scaler tile in CB c_2 via `prepare_reduce_scaler`.
- Uses `DEST_AUTO_LIMIT` for chunk size (shared define with compute kernel).
- Uses TensorAccessor for DRAM reads.

### Reader Kernel (Width-Sharded, noted briefly)

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`

- Reads from local L1 shard (CB c_1, globally allocated).
- Prepares scaler tile when REDUCE_SCALER is defined.
- Provides tiles to CB c_0 in the same column-chunked order.

### Writer Kernel (noted briefly)

**File** (interleaved): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
**File** (sharded): `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`

- Consumes tiles from CB c_3 and writes to DRAM or L1.

---

## DEST_AUTO_LIMIT and Chunk Size

The chunk size (how many columns are reduced simultaneously) is determined by
`DEST_AUTO_LIMIT`, defined in `dest_helpers.hpp`. This value depends on:

1. **Sync mode** (half-sync vs full-sync): Controlled by `ComputeConfig::dst_full_sync_en`
2. **Accumulation mode** (16-bit vs 32-bit): Controlled by `ComputeConfig::fp32_dest_acc_en`

| Sync Mode | Accum Mode | DEST_AUTO_LIMIT |
|-----------|------------|-----------------|
| Half-sync | bfloat16 | 8 |
| Half-sync | fp32 | 4 |
| Full-sync | bfloat16 | 16 |
| Full-sync | fp32 | 8 |

Both reader and compute kernels compute this identically, ensuring they agree on the
tile ordering without explicit host-side coordination (the chunk_size was removed as a
host-passed parameter).

For the **width-sharded** path, chunk_size is forced to 1 regardless of DEST capacity:
```cpp
uint32_t chunk_size = use_width_sharding ? 1 : ttnn::get_dest_reg_count(...);
```
This simplifies the sharded reader logic but means only one column is reduced at a time.

---

## Implementation Notes

### Two Compute Kernel Variants
The operation selects between `reduce_h.cpp` and `reduce_h_neg.cpp` based on the
`negate` flag in operation attributes. The negate variant is used for `reduce_min`
which is implemented as `-reduce_max(-x)`.

### Scaler Purpose
For SUM operations, the scaler is typically 1.0. For AVG, it is 1/N where N is the
reduce factor (Ht * TILE_HEIGHT). The scaler is applied element-wise during `reduce_tile`.

### Column-Major Tile Ordering is Critical
The reader must produce tiles in the exact order expected by the compute kernel's
`REDUCE_COL` + `WaitAndPopPerTile` policy. Any mismatch would cause incorrect
accumulation.

### Flattened Batch Handling
The program factory passes `NC = 1` to the compute kernel and instead increases the
per-core `Wt` argument to include columns from multiple batches. This works because
the reader handles batch boundary wrapping in its tile ID calculations.

### Width-Sharded Optimization
Width-sharded mode uses globally-allocated CBs (c_1 for input shard, c_3 for output),
avoiding DRAM transfers entirely. The reader copies from the shard buffer to the
working CB c_0 tile-by-tile.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does reduce_tile work in the compute API? What are its parameters? How does it accumulate results across multiple calls into the same DST register? Specifically for REDUCE_COL, how does the hardware reduce across the H dimension within a single tile?"
   **Reason**: Needed to understand the core accumulation mechanism that makes height reduction work.
   **Key Findings**: `reduce_tile` performs in-place accumulation into the specified DST register. For `REDUCE_COL`, it collapses rows within a tile and accumulates column-wise into DST. Multiple calls with the same dst_idx accumulate across tiles. Parameters are: `input_cb`, `scaler_cb`, `tile_idx` (in input CB), `scaler_idx` (in scaler CB), `dst_idx` (DST register).

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding DST register semantics, tile_regs_acquire/commit/wait/release protocol, and compute kernel structure.
   **Key Information**: DST registers hold intermediate results. `tile_regs_acquire` locks registers for math, `tile_regs_commit` signals math completion, `tile_regs_wait` waits for pack readiness, `tile_regs_release` frees registers.

2. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Understanding the hardware initialization function called at kernel start.
   **Key Information**: `compute_kernel_hw_startup(icb0, icb1, ocb)` configures unpack, math, and pack hardware. Must be called once at kernel start. Takes input CB A (icb0), input CB B (icb1), and output CB (ocb).

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` and `.inl`
   **Reason**: Understanding the unified reduce library that the standard compute kernel delegates to.
   **Key Information**: The library supports multiple input policies, automatic chunking via DEST_AUTO_LIMIT, and optional accumulation/post-reduce callbacks. For REDUCE_COL with WaitAndPopPerTile, it processes tiles one at a time, using DST register indices 0..chunk_size-1 for simultaneous column reduction.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.inl`
   **Reason**: Understanding how the scaler tile is prepared.
   **Key Information**: `prepare_reduce_scaler<cb_id>(scaler_f)` zeros the tile, then fills row 0 of each face with the scaler value. Format-aware (bfloat16 packs two values per uint32, float32 uses one per uint32).

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST_AUTO_LIMIT calculation.
   **Key Information**: DEST capacity depends on sync mode (half/full) and accumulation mode (16/32-bit). Values range from 4 to 16. Both reader and compute kernels compute this identically via shared defines.

6. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp` (get_defines function)
   **Reason**: Understanding how host-side ReduceOpDim::H maps to kernel-side ReduceDim::REDUCE_COL.
   **Key Information**: `ReduceOpDim::H -> "ReduceDim::REDUCE_COL"`, `ReduceOpDim::W -> "ReduceDim::REDUCE_ROW"`. Also maps ReduceOpMath to PoolType defines.

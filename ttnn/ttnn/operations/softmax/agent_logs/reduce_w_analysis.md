# Reduce W (Width Reduction) Implementation Analysis

## Overview

The **reduce_op_multi_core_w** operation performs reduction along the W (width) dimension of a tiled tensor. Given an input of shape `[N, C, H, W]` in tile layout, it produces an output of shape `[N, C, H, 1t]` where each tile-row of `Wt` input tiles is collapsed into a single output tile. The reduction math can be SUM, AVG, or MAX (controlled by `ReduceOpMath`), with an optional negation variant for implementing MIN via `-MAX(-x)`.

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Compute kernel paths** (selected at program factory time based on `negate` flag):
- Standard: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- Negate variant: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | `Wt` input tiles reduced to 1 output tile |
| **Total units** | `NC * Ht` (all tile-rows across batch and height) |
| **Loop structure** | Outer: rows assigned to core. Inner: `Wt` tiles per row reduced sequentially |

A "work unit" is one tile-row: the set of `Wt` contiguous tiles along the width dimension at a fixed (batch, channel, height-tile) position. Each tile-row produces exactly one output tile.

## Two Code Paths: Reduce-Helper-Library vs Matmul-Based

The compute kernel has **two distinct code paths** selected via the preprocessor define `REDUCE_ROW_SUM_VIA_MM`:

### Path Selection Logic (in `reduce_op_utils::get_defines`)

The `REDUCE_ROW_SUM_VIA_MM` define is set when **both** conditions hold:
1. `reduce_dim == ReduceOpDim::W` (width reduction)
2. `reduce_op == SUM` or `reduce_op == AVG`

For MAX reduction, the define is NOT set, so the reduce-helper-library path is used.

### Path 1: Reduce Helper Library (`#ifndef REDUCE_ROW_SUM_VIA_MM`)

Used for: MAX reduction along W.

```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

compute_kernel_lib::reduce<
    REDUCE_OP,                                             // PoolType::MAX (or SUM/AVG)
    REDUCE_DIM,                                            // ReduceDim::REDUCE_ROW
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    tt::CBIndex::c_0,                                      // input_cb
    tt::CBIndex::c_2,                                      // scaler_cb
    tt::CBIndex::c_3,                                      // output_cb
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));  // block shape
```

**What reduce helper library does internally (REDUCE_ROW path from `reduce_helpers_compute.inl`)**:
1. `reduce_init<reduce_type, REDUCE_ROW>(input_cb, scaler_cb, output_cb)` -- configures unpack, math, pack hardware
2. `cb_wait_front(scaler_cb, 1)` -- waits for scaler tile (already pushed by reader)
3. For each batch `nc` in `[0, NC)`:
   - For each height tile `ht` in `[0, Ht)`:
     - `tile_regs_acquire()` -- acquire DST registers
     - For each width tile `wt` in `[0, Wt)`:
       - `cb_wait_front(input_cb, 1)` -- wait for one input tile
       - `reduce_tile<reduce_type, REDUCE_ROW>(input_cb, scaler_cb, 0, 0, dst_idx=0)` -- accumulate into DST[0]
       - `cb_pop_front(input_cb, 1)` -- consume the tile
     - `cb_reserve_back(output_cb, 1)` -- reserve output space
     - `tile_regs_commit()` / `tile_regs_wait()` -- synchronize DST
     - `pack_tile(0, output_cb)` -- write DST[0] to output CB
     - `tile_regs_release()`
     - `cb_push_back(output_cb, 1)` -- signal output tile ready
4. `reduce_uninit()` -- cleanup

### Path 2: Matmul-Based (`#ifdef REDUCE_ROW_SUM_VIA_MM`)

Used for: SUM and AVG reduction along W.

This path treats the width reduction as a matrix multiply: `input_tile[32x32] * scaler_column[32x1] = result[32x1]`. The scaler tile is prepared as a column vector (row 0 of each face filled with the scaler value).

```cpp
mm_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

cb2.wait_front(1);  // scaler tile from the reader
for (uint32_t nc = 0; nc < NC; nc++) {
    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        acquire_dst();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb0.wait_front(onetile);
            matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0);
            cb0.pop_front(onetile);
        }
        cb3.reserve_back(onetile);
        pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
        cb3.push_back(onetile);
        release_dst();
    }
}
```

The matmul-based path performs the same loop structure but uses `matmul_tiles` instead of `reduce_tile`. The `matmul_tiles` instruction naturally accumulates into DST[0] across the `Wt` iterations, computing the dot product of each row with the scaler column.

### Key Difference for Softmax Reference

For a softmax implementation that needs SUM reduction along W (for the denominator), the **matmul-based path** will be active since `REDUCE_ROW_SUM_VIA_MM` is defined for SUM+W. For MAX reduction (finding the row maximum for numerical stability), the **reduce helper library path** is used with `PoolType::MAX`.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N*C, H, W] (viewed as [NC, Ht, Wt] in tiles) | [N*C, H, 1] (viewed as [NC, Ht, 1] in tiles) |
| **Dimension convention** | NHWC collapsed to [NC, H, W] | [NC, H, 1t] |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (via TensorAccessor) | DRAM (via TensorAccessor) |
| **Data type** | Configurable (bfloat16, float32, etc.) | Configurable (may differ from input) |

### Layout Transformations
- No tilize/untilize within the kernel. The program factory assumes input is already in TILE_LAYOUT.
- The host-level `reduce()` function in `reduce_op.cpp` calls `tilize_with_val_padding()` before invoking the program factory if needed.

## Data Flow Pattern

### Stage 1: Scaler Tile Preparation (Reader Kernel, One-Time)

The reader kernel prepares a scaler tile in CB2 before reading any input data. The scaler preparation depends on the code path:

**Non-MM path** (`prepare_reduce_scaler`):
1. `cb_reserve_back(cb_id_in2, 1)` -- reserve space in CB2
2. Zero all faces of the tile via NoC reads from hardware zeros region
3. Fill row 0 of each face with the scaler value (converted from float to bf16 packed pairs)
4. `cb_push_back(cb_id_in2, 1)` -- push scaler tile

**MM path** (`generate_mm_scaler`):
1. `cb_reserve_back(cb_id_in2, 1)` -- reserve space
2. Zero entire tile via NoC reads from MEM_ZEROS_BASE
3. Fill column 0 of each face with the packed bf16 scaler (every 8th uint32 in face arrays)
4. `cb_push_back(cb_id_in2, 1)` -- push scaler tile

The scaler value itself is passed as a compile-time argument (bit-cast float) to the reader. For SUM, scaler = 1.0f. For AVG, scaler = 1.0f/Wt (set by host). For MAX, scaler = 1.0f.

### Stage 2: Input Tile Streaming (Reader Kernel, Per-Tile)

For each tile from `start_id` to `start_id + num_tiles`:
1. `cb_in0.reserve_back(1)` -- reserve space in CB0
2. `noc.async_read(tensor_accessor, cb_in0, tile_bytes, {page_id=i}, {offset=0})` -- read tile from DRAM
3. `noc.async_read_barrier()` -- wait for read to complete
4. `cb_in0.push_back(1)` -- signal tile is available

Tiles are read sequentially by tile ID. The tile ordering is row-major: for a given core's work, tiles go `[row0_col0, row0_col1, ..., row0_colWt-1, row1_col0, ...]`.

### Stage 3: Compute (Compute Kernel)

For each tile-row (Wt input tiles -> 1 output tile):
1. Acquire DST registers
2. Loop over Wt tiles: wait for input tile in CB0, process via `reduce_tile` or `matmul_tiles`, pop input
3. Pack result from DST[0] to CB3
4. Release DST registers

### Stage 4: Output Writing (Writer Kernel, Per-Tile)

For each output tile:
1. `cb_wait_front(cb_out, 1)` -- wait for compute to produce a tile in CB3
2. Read the L1 address of the tile from CB3
3. `noc_async_write_page(i, tensor_accessor, l1_read_addr)` -- write to DRAM
4. `noc_async_writes_flushed()` -- ensure write dispatched
5. `cb_pop_front(cb_out, 1)` -- free CB3 space

## Circular Buffer Configuration

### Standard Path (non-negate)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler tile for reduce/mm | 2 tiles | 1 tile | Double | Reader | Compute | Program (read once, consumed once, persists via wait_front without pop) |
| c_3 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile-row output) |

### Negate Variant (additional CBs)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_4 | cb_acc | Intermediate accumulator for iterative reduce | 1 tile | 1 tile | Single | Compute (pack) | Compute (copy_tile reload) | Row (persists across wt loop iterations) |
| c_5 | cb_ineg | Negated input tile staging | 1 tile | 1 tile | Single | Compute (pack after negate) | Compute (reduce_tile) | Block (per tile within wt loop) |

**Key insight about CB persistence patterns**:
- **CB0 (input)**: Tiles flow through one at a time. Double-buffered to overlap reader and compute.
- **CB2 (scaler)**: The scaler tile is pushed once by the reader and consumed by `wait_front(1)` in compute. The reduce helper lib (and the MM path) both call `cb_wait_front(scaler_cb, 1)` at the top, before the main loops. The scaler is **never popped** -- it persists for the entire program. Both `reduce_tile` and `matmul_tiles` index it at position 0 repeatedly.
- **CB3 (output)**: One tile produced per tile-row. Double-buffered to overlap compute and writer.
- **CB4 (acc, negate only)**: Used as a ping-pong accumulator in the negate variant. Within the wt loop, the partial reduction result is packed to CB4, then reloaded on the next iteration via `copy_tile`. This is a **multi-pass data reuse pattern** where the same CB serves as both source and destination across loop iterations.
- **CB5 (ineg, negate only)**: Temporary staging for the negated version of each input tile before applying `reduce_tile`.

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Pattern |
|----|----------|------------|---------|
| c_0 | 2 tiles | 1 tile | Double-buffered: Reader can fill tile N+1 while compute processes tile N |
| c_2 | 2 tiles | 1 tile | Persistent: Only 1 tile ever written and never popped |
| c_3 | 2 tiles | 1 tile | Double-buffered: Compute can fill tile N+1 while writer drains tile N |
| c_4 | 1 tile | 1 tile | Single-buffered accumulator: pack then reload (negate path only) |
| c_5 | 1 tile | 1 tile | Single-buffered scratch: produced and consumed within same iteration (negate path only) |

## Index Calculations

### Input Tile Indexing

Tiles are addressed by a flat tile ID. The program factory computes:
- `num_rows = NC * Ht` -- total tile-rows across all batches and height tiles
- Each core is assigned a contiguous range of tile-rows
- `num_tensor_tiles_per_core = num_rows_per_core * Wt` -- total input tiles for the core
- `start_id = num_tiles_read` (accumulated across cores)

The reader kernel uses a simple sequential loop:
```
for i in [start_id, start_id + num_tiles):
    read tile at page_id = i
```

This means input tile index `i` maps to the logical position:
- `row = i / Wt` (which tile-row)
- `col = i % Wt` (which column within the row)
- `batch_channel = row / Ht`
- `height_tile = row % Ht`

### Output Tile Indexing

The writer receives:
- `num_pages = num_tensor_tiles_per_core / Wt` (= num_rows_per_core output tiles)
- `start_id = num_tiles_read / Wt` (output tile start index)

Since the W dimension collapses from `Wt` tiles to 1 tile, the output tile index maps directly to the tile-row index.

## Memory Access Patterns

### Read Pattern
- **Sequential**: Tiles are read in flat tile-ID order. For a given core's range `[start_id, start_id + num_tiles)`, tiles are read one-by-one with `async_read` + barrier per tile.
- **Granularity**: One tile per read operation (no bulk reads).
- **Source**: DRAM via TensorAccessor (interleaved layout, tile pages distributed across DRAM banks).

### Write Pattern
- **Sequential**: Output tiles are written in flat tile-ID order `[start_id, start_id + num_pages)`.
- **Granularity**: One tile per write operation.
- **Destination**: DRAM via TensorAccessor.
- **Flush**: `noc_async_writes_flushed()` after each tile, `noc_async_write_barrier()` at the very end.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D compute grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` |
| **Total cores** | `min(NC * Ht, max_available_cores)` |
| **Work unit** | Tile-row (Wt input tiles -> 1 output tile) |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` tile-rows |
| **Load balancing** | Two-group split via `split_work_to_cores` |
| **Remainder handling** | core_group_1 gets `ceil(num_rows / num_cores)` rows, core_group_2 gets `floor(num_rows / num_cores)` rows |

### How `split_work_to_cores` Works

The function divides `num_rows` work units across available cores:
1. `num_cores = min(num_rows, max_cores_from_grid)`
2. `base = num_rows / num_cores` (floor division)
3. `remainder = num_rows % num_cores`
4. **core_group_1**: first `remainder` cores, each gets `base + 1` rows
5. **core_group_2**: remaining `num_cores - remainder` cores, each gets `base` rows
6. If `remainder == 0`, core_group_2 is empty (all cores equal)

### Core Enumeration

Cores are enumerated via `grid_to_cores(num_cores, grid_x, grid_y, false)` which produces column-major ordering. The per-core runtime arguments use `num_tiles_read` as a running counter to assign contiguous tile ranges.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | scaler_bits | uint32_t | Bit-cast float scaler value (1.0 for SUM/MAX, 1/N for AVG) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for input buffer (page size, bank info, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index to read output tiles from (always `c_3`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Tensor accessor parameters for output buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows per core (varies between core groups) |
| 1 | Wt | uint32_t | Number of tiles in width dimension (tiles to reduce per row) |
| 2 | NC | uint32_t | Always 1 (batch dimension is folded into Ht across cores) |

**Important**: The compute kernel always receives `NC = 1` because the batch*channel dimension is already distributed across cores via the `num_rows_per_core` splitting. Each core processes `num_rows_per_core` tile-rows, regardless of their batch/channel origin. The Ht compile-time arg actually holds `num_rows_per_core_group_N`, not the tensor's true Ht.

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_tensor_tiles_per_core | uint32_t | Total input tiles for this core (`num_rows_per_core * Wt`) |
| 2 | start_id | uint32_t | First tile index to read (= `num_tiles_read` accumulated counter) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Number of output tiles (`num_tensor_tiles_per_core / Wt`) |
| 2 | start_id | uint32_t | First output tile index (`num_tiles_read / Wt`) |

### Preprocessor Defines (passed to all kernels)

| Define | Value | Description |
|--------|-------|-------------|
| `REDUCE_OP` | `PoolType::SUM`, `PoolType::AVG`, or `PoolType::MAX` | Type of reduction math |
| `REDUCE_DIM` | `ReduceDim::REDUCE_ROW` | Always REDUCE_ROW for W reduction |
| `REDUCE_ROW_SUM_VIA_MM` | `1` (only for SUM/AVG + W) | Selects matmul-based code path |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_reduce_universal_start_id | RISCV_0 | NOC0 | DRAM (input tensor) | CB0 (input tiles), CB2 (scaler tile) | Prepare scaler tile, stream input tiles |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`
- **Key Logic**:
  1. Prepares scaler tile in CB2 using either `prepare_reduce_scaler<cb_id_in2>(scaler_f)` (non-MM path) or `generate_mm_scaler(cb_id_in2, packed_bf16)` (MM path).
  2. The scaler value is extracted from compile-time arg 0 via `__builtin_bit_cast(float, scaler_bits)`.
  3. Streams input tiles sequentially from DRAM to CB0, one tile at a time with a read barrier per tile.
  4. Uses `experimental::Noc` and `experimental::CircularBuffer` APIs (newer experimental API style).

### Compute Kernel (Standard Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w | RISCV_2 (unpack+math+pack) | N/A | CB0 (input), CB2 (scaler) | CB3 (output) | reduce_tile or matmul_tiles |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- **Key Logic**:
  - **Non-MM path**: Uses `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, WaitAndPopPerTile, NONE>(c_0, c_2, c_3, ReduceInputBlockShape::of(Ht, Wt, NC))`. This is a single function call that handles the entire computation including all CB synchronization.
  - **MM path**: Manual loop structure with `mm_init`, `acquire_dst`/`release_dst`, `matmul_tiles`, `pack_tile`. The scaler tile in CB2 is waited on once before the main loop and never popped.

### Compute Kernel (Negate Variant)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w_neg | RISCV_2 | N/A | CB0 (input), CB2 (scaler) | CB3 (output) | copy_tile, negative_tile, reduce_tile |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`
- **Key Logic**:
  - Does NOT use `compute_kernel_lib::reduce` helper. Implements the loop manually.
  - For each input tile:
    1. `copy_tile(cb_input, 0, dst_idx)` -- load input to DST
    2. `negative_tile(dst_idx)` -- negate in-place in DST
    3. `pack_tile(dst_idx, cb_ineg)` -- write negated tile to CB5
    4. Reload accumulator from CB4 if `wt > 0` (not first tile in row)
    5. `reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, dst_idx)` -- reduce negated tile
    6. `pack_tile(dst_idx, cb_acc)` -- save partial result to CB4
  - After all Wt tiles processed for a row:
    1. Load final accumulator from CB4
    2. Apply `negative_tile` again (double negation for MIN = -MAX(-x))
    3. Pack to CB3 (output)
  - **Multi-pass reuse pattern**: CB4 (accumulator) is written and re-read within the inner wt loop. This is a feedback loop where the compute kernel is both producer and consumer of CB4.
  - Uses explicit `tile_regs_acquire/commit/wait/release` for each phase (negate phase, reduce phase, final output phase).

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB3 (output tiles) | DRAM (output tensor) | Write tiles to DRAM |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**:
  - Generic writer kernel shared across multiple operations.
  - The output CB index is a compile-time argument (index 0 = `c_3`).
  - Uses `TensorAccessor` for DRAM writes.
  - Sequential one-tile-at-a-time: `cb_wait_front`, `get_read_ptr`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front`.
  - Final `noc_async_write_barrier()` after all tiles written.

## Implementation Notes

### Scaler Tile Format Differences Between Code Paths

The two code paths prepare the scaler tile differently:
- **`prepare_reduce_scaler`** (for reduce_tile path): Fills row 0 of each face with the scaler value. The `reduce_tile` instruction reads this as a column vector to scale the reduction.
- **`generate_mm_scaler`** (for matmul path): Fills column 0 of each face (every 8th uint32 position in bf16 layout, representing the first element of each row). This creates a column vector for the matrix multiply.

### Why NC = 1 in Compute Args

The program factory sets `NC = 1` for the compute kernel compile-time args because it folds the batch*channel dimension into the row count distributed across cores. Each core's `Ht` compile-time arg is actually `num_rows_per_core`, which may span multiple batch/channel combinations. This simplification works because all tile-rows are independent of each other for width reduction.

### Negate Variant Design

The negate variant (`reduce_w_neg.cpp`) implements `reduce_min(x) = -reduce_max(-x)`:
1. Negate each input tile
2. Perform MAX reduction on negated tiles (using CB4 as iterative accumulator)
3. Negate the final result

This requires two extra CBs (c_4, c_5) and cannot use the simple `compute_kernel_lib::reduce` helper because it needs to interleave negation with reduction. The accumulator pattern (CB4) is noteworthy: it uses a single-tile CB as a feedback loop where the compute kernel writes a partial result, then reads it back on the next iteration.

### Relevance to Softmax Implementation

For a softmax operation (`exp(x_i) / sum(exp(x_j))`), the key patterns from this analysis are:

1. **Width reduction loop structure**: The `for ht / for wt` nesting with `acquire_dst` at the row level and `pack_tile` after completing all `Wt` tiles is the canonical pattern for accumulating across width.

2. **Scaler tile as persistent CB**: The scaler tile is pushed once and persists for the entire kernel run (never popped). This is important for softmax where a constant like 1.0 may be needed throughout.

3. **Reduce helper library API**: `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile, NONE>(cb_in, cb_scaler, cb_out, ReduceInputBlockShape::of(Ht, Wt, NC))` is the one-liner for performing complete width reduction. For softmax, one could use `WaitUpfrontNoPop` policy to keep input tiles in CB for reuse (e.g., for both max and sum phases).

4. **Post-reduce callback**: The reduce helper supports a post-reduce lambda (e.g., `recip_tile` for computing 1/sum). This is documented in the header as being ideal for softmax patterns.

5. **CB4 as accumulator feedback loop**: The negate variant's pattern of using a CB as an iterative accumulator (write partial result, read it back next iteration) may be useful for multi-phase softmax computations.

6. **`REDUCE_ROW_SUM_VIA_MM` path**: When the softmax needs `sum(exp(x_j))` along W, the matmul-based reduction path will be activated. Understanding both paths is important.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What does compute_kernel_hw_startup do in tt-metal compute kernels?"
   **Reason**: Understanding the hardware initialization required before any compute operations.
   **Key Findings**: `compute_kernel_hw_startup(icb0, icb1, ocb)` configures the unpack, math, and pack RISC-V processors. Specifically:
   - UNPACK: `llk_unpack_hw_configure` -- sets up unpacker for input CBs
   - MATH: `llk_math_pack_sync_init` + `llk_math_hw_configure` -- configures FPU/SFPU
   - PACK: `llk_pack_hw_configure` + `llk_pack_init` + `llk_pack_dest_init` -- configures packer for output CB
   Must be called once at kernel start before any operation-specific init.

2. **Query**: "How does reduce_tile work in tt-metal? How does REDUCE_ROW work?"
   **Reason**: Understanding the core reduce instruction parameters and behavior.
   **Key Findings**: DeepWiki query failed. Information was derived from code analysis: `reduce_tile<reduce_type, reduce_dim>(input_cb, scaler_cb, tile_idx, scaler_idx, dst_idx)` reads from input_cb at tile_idx, applies the reduction operation with scaling, and accumulates into DST[dst_idx]. For REDUCE_ROW, it collapses all 32 columns of the tile into a single column by applying the reduce operation (sum/max) across each row and multiplying by the scaler.

3. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: Understanding work distribution across cores.
   **Key Findings**: DeepWiki query failed. Derived from `work_split.hpp` header comments: Returns (num_cores, all_cores, core_group_1, core_group_2, work_per_group_1, work_per_group_2). core_group_1 gets ceil(total/cores) work, core_group_2 gets floor(total/cores). If work divides evenly, core_group_2 is empty.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
   **Reason**: Understanding the reduce helper library API, input policies, and accumulation patterns
   **Key Information**: The `reduce<>()` template function with `ReduceInputPolicy` (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), `ReduceDataFormatReconfigMode`, `AccumulateT`, and `PostReduceOp` parameters. Softmax-specific examples included in docstrings.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`
   **Reason**: Understanding the exact implementation of the REDUCE_ROW path
   **Key Information**: The REDUCE_ROW path loops `nc -> ht -> wt`, with tile_regs_acquire per row, reduce_tile per tile, and pack_tile per completed row. post_reduce_op is called with dst_idx before packing.

3. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp` (lines 20-42)
   **Reason**: Understanding how `get_defines` generates preprocessor defines
   **Key Information**: `REDUCE_ROW_SUM_VIA_MM` is only set for `(SUM or AVG) + W` combinations. The `REDUCE_OP` and `REDUCE_DIM` defines use the `PoolType::` and `ReduceDim::` enum namespaces.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp/.inl`
   **Reason**: Understanding how the scaler tile is prepared on the reader side
   **Key Information**: `prepare_reduce_scaler<cb_id>(scaler_f)` zeros the tile via NoC reads from hardware zeros region, then fills row 0 of each face with the float-to-bf16-converted scaler value. Format is auto-detected from the CB data format.

5. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp`
   **Reason**: Understanding the matmul scaler tile format
   **Key Information**: Creates a column vector by writing packed bf16 values to every 8th position in each face (column 0 only). Different from `prepare_reduce_scaler` which fills entire row 0.

6. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity and `DEST_AUTO_LIMIT`
   **Key Information**: DEST capacity varies: SyncFull+fp16=16, SyncFull+fp32=8, SyncHalf+fp16=8, SyncHalf+fp32=4. `DEST_AUTO_LIMIT = get_dest_limit()` is used by REDUCE_COL for chunking but not relevant to REDUCE_ROW (which only uses DST[0]).

7. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Understanding the work splitting utility
   **Key Information**: `split_work_to_cores(grid_size, units_to_divide)` returns a 6-tuple with two core groups. The function ensures work is divided as evenly as possible with at most a difference of 1 unit between the two groups.

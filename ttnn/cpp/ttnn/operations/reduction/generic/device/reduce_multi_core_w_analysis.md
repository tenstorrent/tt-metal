# Reduce Multi-Core W Implementation Analysis

## Overview

This operation performs reduction along the **W (width) dimension** of a 4D tensor using multiple Tensix cores in parallel. Each row of tiles is reduced to a single output tile, collapsing the W dimension while preserving N, C, and H dimensions.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Supported Operations**:
- **SUM**: Sums all elements along W (uses matmul-based reduction for efficiency)
- **MAX**: Finds maximum element along W (uses reduce_tile)

**Key Architectural Pattern**: For SUM reductions, this factory uses the `REDUCE_ROW_SUM_VIA_MM` optimization, which implements row summation via matrix multiplication with a scaler tile containing the scaling factor. This leverages the hardware's optimized matrix multiply units.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Row of tiles (tile-row) |
| **Unit size** | Wt tiles input, 1 tile output |
| **Total units** | NC * Ht (number of tile-rows in tensor) |
| **Loop structure** | For each row: read Wt tiles, reduce to 1 tile, write 1 tile |

A **work unit** is one tile-row: reading `Wt` input tiles and producing `1` output tile. The reduction collapses a row of `Wt` tiles along the width dimension into a single tile.

**Reduction Ratio**: `Wt : 1` (e.g., if W=1024 with TILE_WIDTH=32, then Wt=32 tiles become 1 tile)

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW (batch, channel, height, width) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32 (configurable) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, 1] (W dimension collapsed to 1 tile width) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input or configurable via output_dtype |

### Shape Transformation

```
Input:  [N, C, H, W]           with padded shape in tiles: [N, C, Ht, Wt]
Output: [N, C, H, TILE_WIDTH]  with padded shape in tiles: [N, C, Ht, 1]
```

The output has exactly `Ht` tiles per NC batch, where each tile contains the reduced result of one tile-row.

## Data Flow Pattern

### High-Level Flow

```
DRAM Input                    Reader Kernel              Compute Kernel            Writer Kernel              DRAM Output
    |                             |                           |                          |                         |
    |  read Wt tiles per row      |                           |                          |                         |
    |------------------------->   |  push to CB_in0           |                          |                         |
    |                             |-------------------------->|                          |                         |
    |  generate scaler tile       |                           |                          |                         |
    |  (one-time at start)        |  push to CB_scaler        |                          |                         |
    |                             |-------------------------->|                          |                         |
    |                             |                           |  reduce Wt tiles -> 1    |                         |
    |                             |                           |  push to CB_out          |                         |
    |                             |                           |------------------------->|                          |
    |                             |                           |                          |  write 1 tile to DRAM   |
    |                             |                           |                          |------------------------>|
```

### Detailed Stage Flow

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 0 | Reader | (local) | CB_scaler (c_2) | generate_reduce_scaler: reserve_back, push_back (once at start) |
| 1 | Reader | DRAM | CB_in0 (c_0) | reserve_back(1), noc_async_read_page, push_back(1) per tile |
| 2 | Compute | CB_in0 (c_0), CB_scaler (c_2) | CB_out (c_3) | wait_front, reduce/matmul, pop_front per tile; reserve_back, pack_tile, push_back for output |
| 3 | Writer | CB_out (c_3) | DRAM | wait_front(1), noc_async_write_page, pop_front(1) per tile |

### Scaler Tile Generation (Critical for Mean Computation)

The reader kernel generates a **scaler tile** at startup that is used throughout the reduction:

1. **Purpose**: The scaler value is multiplied with each input tile during reduction
2. **For SUM**: scaler = 1.0 (or user-provided value)
3. **For MEAN**: scaler = 1.0/W (to compute average)
4. **Format**: bfloat16, packed as two bfloat16 values in a uint32
5. **Generation**: The `generate_reduce_scaler()` function creates a tile where the first 8 elements of each face are set to the scaler value, rest are zeros

```cpp
// In program factory:
bfloat16 bfloat_scaler_value = bfloat16::truncate(operation_attributes.scaler);
uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

// In reader kernel:
generate_reduce_scaler(cb_id_in2, scaler);  // Creates scaler tile in CB_scaler
```

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_2 | cb_scaler | Scaler tile | 2 tiles | 1 tile | Single (used as 1) | Reader | Compute | Program |
| c_3 | cb_out | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |

### CB Design Notes

1. **CB_in0 (c_0)**: Double-buffered to allow reader to prefetch next tile while compute processes current
2. **CB_scaler (c_2)**: Generated once at kernel start, persists for entire program. Compute kernel calls `cb_wait_front(c_2, 1)` once to ensure scaler is ready
3. **CB_out (c_3)**: Double-buffered to allow compute to produce next tile while writer sends current

### Tile Sizes

- **Input tile size**: `tt::tile_size(src0_cb_data_format)` - typically 2048 bytes for bfloat16
- **Scaler tile size**: Always 2048 bytes (hardcoded bfloat16)
- **Output tile size**: `tt::tile_size(dst_cb_data_format)` - depends on output dtype

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Ratio | Pattern |
|----|----------|------------|-------|---------|
| c_0 (input) | 2 tiles | 1 tile | 2:1 | Double-buffered |
| c_2 (scaler) | 2 tiles | 1 tile | 2:1 | Single-buffered (one-time fill) |
| c_3 (output) | 2 tiles | 1 tile | 2:1 | Double-buffered |

**Pipeline Overlap**: Reader and writer can overlap with compute due to double buffering on input and output CBs.

## Index Calculations

### Reader Kernel Index Calculation

```cpp
// Per-core starting tile index
uint32_t start_id = get_arg_val<uint32_t>(2);  // Runtime arg: starting tile index

// Loop through assigned tiles
for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
    noc_async_read_page(i, tensor_accessor, l1_write_addr);
}
```

The reader uses **linear tile indexing** where tiles are numbered sequentially in row-major order across the entire tensor:
- Tile index `i` maps to position `(n, c, h, w)` where `i = n*C*Ht*Wt + c*Ht*Wt + h*Wt + w`

### Compute Kernel Index Mapping

The compute kernel processes tiles in row-major order within its assigned rows:
```cpp
for (uint32_t ht = 0; ht < Ht; ++ht) {
    // Process one row: Wt tiles -> 1 output tile
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        cb_wait_front(CB_in0, 1);
        matmul_tiles(CB_in0, CB_scaler, 0, 0, 0);  // or reduce_tile
        cb_pop_front(CB_in0, 1);
    }
    // Pack result
    pack_tile(0, CB_out);
}
```

### Writer Kernel Index Calculation

```cpp
// Output tiles are 1/Wt of input tiles (reduction ratio)
uint32_t num_output_tiles = num_input_tiles / Wt;
uint32_t output_start_id = input_start_id / Wt;

for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
    noc_async_write_page(i, tensor_accessor, l1_read_addr);
}
```

## Memory Access Patterns

### Read Pattern

- **Pattern Type**: Sequential tile reads
- **Ordering**: Row-major (contiguous tiles along W, then H, then C, then N)
- **Granularity**: One tile per NOC transaction
- **Barrier**: `noc_async_read_barrier()` after each tile (could be optimized)

### Write Pattern

- **Pattern Type**: Sequential tile writes
- **Ordering**: Row-major (one output tile per Wt input tiles)
- **Granularity**: One tile per NOC transaction
- **Barrier**: `noc_async_write_barrier()` after each tile

### Access Locality

Input tiles for one reduction (one tile-row) are contiguous in memory, providing good spatial locality. However, the reader processes one tile at a time rather than batching reads.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (cores enumerated linearly) |
| **Grid dimensions** | Up to device compute_with_storage_grid_size |
| **Total cores** | min(num_rows, max_cores) |
| **Work per core** | num_rows_per_core tile-rows |
| **Load balancing** | Two-group split (group_1 gets +1 row if uneven) |

### Work Distribution Algorithm

```cpp
uint32_t num_rows = NC * Ht;  // Total tile-rows to process

// split_work_to_cores returns:
// - core_group_1: cores with num_rows_per_core_group_1 rows each
// - core_group_2: cores with num_rows_per_core_group_2 rows each (one less)

// If num_rows % num_cores != 0:
//   - group_1 gets ceiling(num_rows/num_cores) rows
//   - group_2 gets floor(num_rows/num_cores) rows
```

### Per-Core Tile Assignment

```cpp
// Input tiles per core
uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;

// Output tiles per core
uint32_t num_output_tiles = num_tensor_tiles_per_core / Wt;  // = num_rows_per_core
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scaler_value | uint32_t | Two bfloat16 scaler values packed into uint32 |
| 1+ | tensor_accessor_args | TensorAccessorArgs | Buffer access configuration (rank, num_banks, shapes, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_3) |
| 1+ | tensor_accessor_args | TensorAccessorArgs | Buffer access configuration |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows per core (num_rows_per_core_group_X) |
| 1 | Wt | uint32_t | Number of tiles along W dimension |
| 2 | NC | uint32_t | Always 1 (batch dimension handled by core distribution) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer address in DRAM/L1 |
| 1 | num_tiles | uint32_t | Total input tiles for this core (num_rows * Wt) |
| 2 | start_id | uint32_t | Starting tile index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer address in DRAM/L1 |
| 1 | num_tiles | uint32_t | Output tiles for this core (num_rows) |
| 2 | start_id | uint32_t | Starting output tile index (input_start_id / Wt) |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_reduce_universal_start_id | RISCV_0 | NOC0 | DRAM | CB_in0, CB_scaler | Read tiles, generate scaler |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`

**Key Logic**:
1. Generate scaler tile via `generate_reduce_scaler()` - fills CB_scaler with scaling factor
2. Loop over assigned tiles, reading each via `noc_async_read_page()` into CB_in0
3. Uses `TensorAccessor` for address calculation based on tile index

```cpp
// One-time scaler generation
generate_reduce_scaler(cb_id_in2, scaler);  // scaler is packed_scaler_value

// Tile read loop
for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
    cb_reserve_back(cb_id_in0, onetile);
    noc_async_read_page(i, tensor_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, onetile);
}
```

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w | TRISC | N/A | CB_in0, CB_scaler | CB_out | Reduce/matmul tiles |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`

**Key Logic**:

Two implementations controlled by `REDUCE_ROW_SUM_VIA_MM` define:

**Path 1: Matmul-based reduction (SUM only)**
```cpp
mm_init(CB_in0, CB_scaler, CB_out);
cb_wait_front(CB_scaler, 1);  // Wait for scaler once

for (uint32_t ht = 0; ht < Ht; ++ht) {
    acquire_dst();
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        cb_wait_front(CB_in0, 1);
        matmul_tiles(CB_in0, CB_scaler, 0, 0, 0);  // Accumulates in DST[0]
        cb_pop_front(CB_in0, 1);
    }
    cb_reserve_back(CB_out, 1);
    pack_tile(0, CB_out);  // Pack DST[0] to output
    cb_push_back(CB_out, 1);
    release_dst();
}
```

**Path 2: reduce_helpers library (generic)**
```cpp
compute_kernel_hw_startup(CB_in0, CB_scaler, CB_out);
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, STREAMING>(
    CB_in0, CB_scaler, CB_out, TileShape::grid(Ht, Wt, NC));
```

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_universal_start_id | RISCV_1 | NOC1 | CB_out | DRAM | Write tiles |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_universal_start_id.cpp`

**Key Logic**:
```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, onetile);
    noc_async_write_page(i, tensor_accessor, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, onetile);
}
```

## Implementation Notes

### REDUCE_ROW_SUM_VIA_MM Optimization

For SUM reductions along W, this factory uses matrix multiplication instead of the generic `reduce_tile` function:
- **Rationale**: The matrix multiply units are highly optimized on Tenstorrent hardware
- **Mechanism**: Each input tile is multiplied with the scaler tile, accumulating into DST[0]
- **Enabled by**: `defines["REDUCE_ROW_SUM_VIA_MM"] = "1"` when `reduce_dim == W && reduce_op == SUM`

### Scaler Tile Structure

The scaler tile created by `generate_reduce_scaler()`:
- **Tile dimensions**: 32x32 elements (standard tile)
- **Non-zero elements**: Only first 8 elements of each 16x16 face contain the scaler value
- **Memory format**: Row of 8 scalers repeated across 4 faces
- **Total size**: 2048 bytes (bfloat16)

### Implications for Variance Computation

For the variance operation (mean of squared differences):
1. **Mean computation**: Use scaler = 1.0/W to compute mean = sum(x)/W
2. **The scaler CB persists**: Can be reused for the final mean-of-squares reduction
3. **Broadcasting requirement**: After computing mean, need to broadcast it back to all W positions for subtraction
4. **Additional CBs needed**: For intermediate results (mean, diff, squared)

### Two Core Groups

The factory creates potentially two different compute kernels:
- **core_group_1**: Gets `num_rows_per_core_group_1` rows
- **core_group_2**: Gets `num_rows_per_core_group_2` rows (one less)

This handles non-uniform work distribution when `num_rows % num_cores != 0`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the reduce_tile function work in compute kernels? What are the parameters and how does it perform row reduction (REDUCE_ROW) for the W dimension?"
   **Reason**: Needed to understand the core reduction primitive
   **Key Findings**: reduce_tile takes input CB, scaler CB, tile indices, and DST index. REDUCE_ROW collapses W dimension. Requires reduce_init before use.

2. **Query**: "What is the REDUCE_ROW_SUM_VIA_MM approach for row reduction? How does it use matrix multiplication instead of reduce_tile for summing along the W dimension?"
   **Reason**: The code shows conditional compilation for matmul-based reduction
   **Key Findings**: REDUCE_ROW_SUM_VIA_MM is an optimization that uses matmul_tiles for SUM operations, leveraging the hardware's matrix multiply units for better performance.

3. **Query**: "How does TensorAccessor work in tt-metal dataflow kernels? What is TensorAccessorArgs and noc_async_read_page?"
   **Reason**: Reader kernel uses TensorAccessor for memory access
   **Key Findings**: TensorAccessor abstracts memory bank distribution, TensorAccessorArgs configures compile-time vs runtime parameter passing, noc_async_read_page reads a tile via NOC using TensorAccessor for address calculation.

4. **Query**: "How does split_work_to_cores work in tt-metal? What do core_group_1 and core_group_2 represent?"
   **Reason**: Program factory uses this for work distribution
   **Key Findings**: split_work_to_cores distributes work units across cores, returning two groups - group_1 gets ceiling work units, group_2 gets floor work units when work doesn't divide evenly.

5. **Query**: "What is the tile layout and data format in Tenstorrent hardware? How is data stored in bfloat16 tiles and what is the tile size?"
   **Reason**: Need to understand tile structure for CB sizing
   **Key Findings**: 32x32 tiles with four 16x16 faces, bfloat16 tile is 2048 bytes, row-major face storage by default.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
   **Reason**: Understanding the reduce library abstraction
   **Key Information**: Provides unified reduce function with multiple modes (STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT), handles DST register management, CB operations, and packing automatically.

2. **Source**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
   **Reason**: Understanding scaler tile generation
   **Key Information**: Creates a tile where first 8 elements of each face contain the packed scaler value, rest are zeros. Uses NOC reads from zero memory for initialization.

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register limits
   **Key Information**: DEST capacity depends on sync mode and accumulation mode (4-16 tiles). DEST_AUTO_LIMIT provides compile-time constant.

4. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
   **Reason**: Understanding how defines are generated
   **Key Information**: get_defines() maps ReduceOpDim::W to "ReduceDim::REDUCE_ROW" and sets REDUCE_ROW_SUM_VIA_MM for SUM+W combination.

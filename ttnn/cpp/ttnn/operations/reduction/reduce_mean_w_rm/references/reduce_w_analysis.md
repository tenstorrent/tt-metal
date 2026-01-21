# Reduce Multi-Core Width Implementation Analysis

## Overview

This analysis documents the implementation of the **Width-dimension Reduction** (reduce W) operation in TTNN, which reduces tensor data along the width dimension (innermost dimension in row-major layout). The operation supports SUM, MAX, and AVG reduction types.

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Associated Kernel Files**:
- **Reader**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`
- **Compute**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- **Writer**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total input units** | NC * Ht * Wt tiles |
| **Total output units** | NC * Ht tiles |
| **Loop structure** | For each row of tiles (Ht), reduce all Wt tiles to 1 output tile |

**What constitutes one unit of work**: A "tile-row" - reducing Wt input tiles to produce 1 output tile. Work is distributed by assigning groups of tile-rows to different cores.

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Any (converted to DataFormat via `datatype_to_dataformat_converter`) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | [N, C, H, 1] (collapsed to 1 tile width) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Configurable (defaults to input dtype) |

### Layout Transformations

- **No layout transformation** within this operation - input and output are both TILE_LAYOUT
- **Dimensional collapse**: Width dimension reduces from Wt tiles to 1 tile
- **Tile count reduction**: Output has NC * Ht tiles vs input's NC * Ht * Wt tiles

## Data Flow Pattern

### Step-by-Step Flow

1. **Reader Kernel** (RISCV_0 / BRISC):
   - Generates scaler tile in CB2 for reduce operation (1.0 for sum/max, 1/W for mean)
   - Reads input tiles sequentially from DRAM using TensorAccessor
   - Pushes tiles one-at-a-time to CB0 (input circular buffer)
   - Each core reads `num_rows_per_core * Wt` tiles starting from `start_id`

2. **Compute Kernel** (TRISC):
   - Waits for scaler tile in CB2 (once at start)
   - For each tile-row (Ht iterations):
     - Acquires DEST register
     - Reduces Wt tiles from CB0 to single result in DEST
     - Packs result to CB3 (output circular buffer)
     - Releases DEST register

3. **Writer Kernel** (RISCV_1 / NCRISC):
   - Reads reduced tiles from CB3
   - Writes tiles sequentially to output DRAM buffer
   - Each core writes `num_rows_per_core` tiles

### Data Flow Diagram

```
DRAM Input                     L1 CBs                              DRAM Output
+--------+                  +----------+                           +--------+
| Tile 0 |--+               |   CB0    |                           | Out 0  |
| Tile 1 |  |   Reader      | (input)  |   Compute      Writer     +--------+
| ...    |  +-------------->| 2 tiles  |------------->+--------+-->| Out 1  |
| Tile W |  | (Wt tiles     +----------+   reduce     |  CB3   |   +--------+
+--------+  |  per row)                    Wt->1      |(output)|   | ...    |
            |               +----------+              | 2 tiles|   +--------+
            |               |   CB2    |              +--------+
            +-------------->| (scaler) |
                            |  2 tiles |
                            +----------+
```

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 (0) | cb_src0 | Input tiles staging | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_2 | cb_scaler | Scaler tile for reduce | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_3 | cb_output | Output tiles staging | 2 tiles | 1 tile | Double | Compute | Writer | Block |

### CB Configuration Details

- **cb_src0 (index 0)**: Input tiles from DRAM. Double-buffered (2 tiles) to allow overlap between reader and compute.
- **cb_scaler (index 2)**: Contains the scaler tile used in reduction. Generated once by reader, consumed throughout by compute.
- **cb_output (index 3)**: Output tiles after reduction. Double-buffered to allow overlap between compute and writer.

**Scaler Tile Contents**:
- For SUM/MAX operations: All 1.0 values
- For MEAN operation: 1/W (reciprocal of width) for averaging

## Pipeline Pattern Summary

All CBs use **Double-buffering** (capacity = 2 * block_size):
- Allows producer-consumer overlap
- Reader can fill next tile while compute processes current
- Compute can pack output while writer drains previous

**Pipeline Overlap**: Yes - producer/consumer overlap is enabled due to 2-tile capacity vs 1-tile block size.

## Index Calculations

### TensorAccessor Usage

The operation uses `TensorAccessor` for both reading input and writing output:

```cpp
// Host side - compile time args
TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

// Kernel side - page-based access
auto tensor_accessor = TensorAccessor(tensor_args, src_addr, tile_bytes);
noc_async_read_page(i, tensor_accessor, l1_write_addr);
```

### Index Mapping

**Input tile index**: Sequential page index in row-major order
- `tile_index = batch * (Ht * Wt) + row * Wt + col`
- Reader reads tiles `[start_id, start_id + num_tensor_tiles_per_core)`

**Output tile index**:
- `out_tile_index = start_id / Wt` (divides by width to get row index)
- Each Wt input tiles produce 1 output tile

### Per-Core Index Assignment

```cpp
// Reader runtime args
uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
SetRuntimeArgs(reader, core, {
    src_addr,
    num_tensor_tiles_per_core,
    num_tiles_read  // start_id
});

// Writer runtime args
SetRuntimeArgs(writer, core, {
    dst_addr,
    num_tensor_tiles_per_core / Wt,  // num output tiles
    num_tiles_read / Wt              // output start_id
});
```

## Memory Access Patterns

### Read Pattern

| Attribute | Value |
|-----------|-------|
| **Pattern** | Sequential |
| **Memory type** | DRAM |
| **Access granularity** | 1 tile per NoC transaction |
| **Stride** | Contiguous (tiles in row-major order) |
| **Blocking** | Yes (cb_reserve_back + barrier after each tile) |

The reader fetches tiles sequentially using `noc_async_read_page` with immediate barrier. This is simple but not optimal for throughput - a batched approach could improve performance.

### Write Pattern

| Attribute | Value |
|-----------|-------|
| **Pattern** | Sequential |
| **Memory type** | DRAM |
| **Access granularity** | 1 tile per NoC transaction |
| **Stride** | Contiguous (reduced tiles in row-major order) |
| **Blocking** | Yes (barrier after each tile) |

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (uses compute_with_storage_grid_size) |
| **Grid dimensions** | Device-dependent (e.g., 8x8 on Wormhole) |
| **Total cores** | min(num_rows, max_available_cores) |
| **Work unit** | tile-row (Wt input tiles -> 1 output tile) |
| **Load balancing** | Round-robin with remainder handling |
| **Remainder handling** | core_group_1 gets +1 row vs core_group_2 |

### Work Splitting Details

The operation uses `tt::tt_metal::split_work_to_cores()` to distribute tile-rows across cores:

```cpp
uint32_t num_rows = NC * Ht;  // Total tile-rows to process

std::tie(num_cores, all_cores, core_group_1, core_group_2,
         num_rows_per_core_group_1, num_rows_per_core_group_2) =
    split_work_to_cores(compute_with_storage_grid_size, num_rows);
```

**Result**:
- `core_group_1`: Cores that process `num_rows_per_core_group_1` rows each
- `core_group_2`: Cores that process `num_rows_per_core_group_2` rows each (one less)
- If evenly divisible: `core_group_2` is empty

### Core Iteration Pattern

Cores are iterated column-wise (row_wise=false by default):

```cpp
cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, false);
// Pattern: (0,0), (0,1), (0,2), ..., (1,0), (1,1), ...
```

Or if `sub_core_grids` is specified, cores are extracted from the custom grid:

```cpp
for (const auto& range : all_cores.ranges()) {
    for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
        for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
            cores.emplace_back(x, y);
        }
    }
}
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scaler_value | uint32_t | Scaler packed as 2x bfloat16 in uint32 |
| 1+ | tensor_accessor_args | TensorAccessorArgs | Buffer metadata for NoC addressing |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index to read output from (c_3) |
| 1+ | tensor_accessor_args | TensorAccessorArgs | Buffer metadata for NoC addressing |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows per core (num_rows_per_core) |
| 1 | Wt | uint32_t | Number of tile-columns to reduce |
| 2 | NC | uint32_t | Always 1 (batch loop handled by host distribution) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address |
| 1 | num_tiles | uint32_t | Total tiles to read (num_rows * Wt) |
| 2 | start_id | uint32_t | Starting tile index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address |
| 1 | num_pages | uint32_t | Number of output tiles (num_rows) |
| 2 | start_id | uint32_t | Starting output tile index |

## Kernel Implementations

### Reader Kernel: reader_unary_reduce_universal_start_id.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM src_buffer | CB0, CB2 | Read tiles, generate scaler |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`

**Key Logic**:
1. **Scaler Generation**: Uses `generate_reduce_scaler()` to create a tile filled with the packed scaler value
   - Creates a tile with zeros first (via NoC read from MEM_ZEROS_BASE)
   - Fills specific positions with the scaler value for the reduce hardware
2. **Tile Reading Loop**: Simple sequential read with per-tile barrier
   - `cb_reserve_back` -> `noc_async_read_page` -> `noc_async_read_barrier` -> `cb_push_back`

**Conditional Compilation**:
- `REDUCE_ROW_SUM_VIA_MM`: Uses matrix-multiplication-based scaler generation when enabled

### Compute Kernel: reduce_w.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | TRISC (2,3,4) | N/A | CB0, CB2 | CB3 | reduce_tile / matmul |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`

**Key Logic**:

**Standard Path** (when REDUCE_ROW_SUM_VIA_MM is NOT defined):
```cpp
compute_kernel_hw_startup(cb_in0, cb_scaler, cb_out);
compute_kernel_lib::reduce<
    REDUCE_OP,    // SUM or MAX
    REDUCE_DIM,   // REDUCE_ROW for W-reduction
    ReduceInputMode::STREAMING,
    ReduceDataFormatReconfig::NONE>(
    cb_in, cb_scaler, cb_out,
    TileShape::grid(Ht, Wt, NC));
```

Uses the `reduce_helpers.hpp` library which:
- Handles DEST register acquisition/release
- Processes tiles in STREAMING mode (one-at-a-time wait/pop)
- Manages reduce_init/uninit calls

**REDUCE_ROW_SUM_VIA_MM Path** (for W-dimension SUM only):
```cpp
mm_init(cb_in0, cb_scaler, cb_out);
for (ht = 0; ht < Ht; ++ht) {
    acquire_dst();
    for (wt = 0; wt < Wt; ++wt) {
        cb_wait_front(cb_in0, 1);
        matmul_tiles(cb_in0, cb_scaler, 0, 0, 0);
        cb_pop_front(cb_in0, 1);
    }
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    release_dst();
}
```

Uses matrix multiplication to perform row-sum reduction, leveraging the hardware's optimized matmul path.

**Why Two Paths**:
- `REDUCE_ROW_SUM_VIA_MM` is an optimization for W-dimension SUM operations
- Matrix multiplication hardware is highly optimized on Tenstorrent
- Row reduction can be expressed as matmul: `[1xW] * [Wx1] = [1x1]` per row

### Writer Kernel: writer_unary_interleaved_start_id.cpp

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB3 | DRAM dst_buffer | Write tiles |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Simple sequential write loop
- `cb_wait_front` -> `noc_async_write_page` -> `noc_async_write_barrier` -> `cb_pop_front`
- Supports `BACKWARDS` mode for reverse iteration (not used in reduce)
- Supports `OUT_SHARDED` mode (waits for all pages, no write loop)

**Note**: This is a generic writer kernel shared across many operations (eltwise unary, reduce, etc.)

## Implementation Notes

### Design Decisions

1. **REDUCE_ROW_SUM_VIA_MM Optimization**:
   - Enabled only for `ReduceOpDim::W` + `ReduceOpMath::SUM`
   - Matrix multiplication hardware provides better throughput for sum reductions
   - Other reduction types (MAX) use the standard `reduce_tile` path

2. **Scaler as Compile-Time Argument**:
   - Packed scaler value is a compile-time arg for efficiency
   - Avoids runtime overhead of passing scaler per-core
   - Scaler tile is generated once per core by the reader

3. **Single-Tile Streaming**:
   - Reader uses 1-tile-at-a-time read with immediate barrier
   - Simple but potentially suboptimal - no read pipelining
   - Double-buffered CBs allow compute/reader overlap despite sequential reads

4. **Generic Writer Kernel**:
   - Reuses `writer_unary_interleaved_start_id.cpp` from eltwise
   - Reduces code duplication
   - Supports multiple modes (forward, backward, sharded)

5. **TensorAccessor for Memory Access**:
   - Uses `TensorAccessorArgs` for compile-time buffer metadata
   - Enables proper NoC address calculation for interleaved buffers
   - Abstracts bank distribution details from kernel code

### Pain Points and Considerations

1. **No Batched Reading**: Reader issues individual NoC reads with barriers. Batched reads could improve throughput for large tensors.

2. **Fixed CB Sizes**: CB capacities (2 tiles) are hardcoded. Larger buffers might improve throughput for high-latency DRAM access.

3. **Compile-Time Kernel Variants**: Different compute kernels for core_group_1 vs core_group_2 when row counts differ. Creates additional kernel compilation overhead.

4. **No Sharding Support**: This factory only handles interleaved memory. Sharded variants would require different data flow patterns.

5. **Scaler Tile Format**: Hardcoded to bfloat16 (`Float16_b`). May cause precision issues with float32 inputs/outputs.

### Edge Cases

- **Empty tensor**: Not explicitly handled - would result in 0 cores and empty result
- **Single tile width (Wt=1)**: Reduction becomes identity - still works correctly
- **Uneven work distribution**: Handled via core_group_1/core_group_2 split

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the TensorAccessor and TensorAccessorArgs system in tt-metal?"
   **Reason**: Understanding how tensor data is accessed in kernels via NoC
   **Key Findings**: TensorAccessor abstracts page-to-bank mapping for interleaved/sharded buffers. Host creates TensorAccessorArgs that get serialized as compile-time args. Kernel creates TensorAccessor instance to calculate NoC addresses.

2. **Query**: "What does the reduce_tile function do in tt-metal compute kernels?"
   **Reason**: Understanding the core reduction primitive
   **Key Findings**: reduce_tile performs hardware-accelerated reduction on tiles. Template parameters specify reduction type (SUM/MAX/AVG) and dimension (ROW/COL/SCALAR). Uses LLK functions llk_math_reduce and llk_unpack_AB internally.

3. **Query**: "What does split_work_to_cores do in tt-metal?"
   **Reason**: Understanding work distribution across cores
   **Key Findings**: Divides work units among available cores. Returns two core groups - one that does floor(work/cores) units, one that does floor+1 when not evenly divisible. Handles both rectangular grid and CoreRangeSet inputs.

4. **Query**: "What is the REDUCE_ROW_SUM_VIA_MM optimization?"
   **Reason**: Understanding why matmul is used for reduction
   **Key Findings**: Row-sum reduction can be expressed as matrix multiplication, which is highly optimized on Tenstorrent hardware. This optimization is enabled only for W-dimension SUM operations.

5. **Query**: "What is the difference between ReduceDim::REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR?"
   **Reason**: Understanding reduction dimension semantics
   **Key Findings**: REDUCE_ROW collapses width (W dimension), REDUCE_COL collapses height (H dimension), REDUCE_SCALAR collapses both. Maps to ReduceOpDim::W, H, HW respectively.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
   **Reason**: Understanding the unified reduce library used by compute kernel
   **Key Information**: Provides templated reduce() function handling all reduction patterns. Manages DEST registers, CB operations, and supports multiple input modes (STREAMING, PRELOADED, etc.)

2. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Understanding the work distribution API
   **Key Information**: split_work_to_cores returns 6-tuple: (num_cores, all_cores, core_group_1, core_group_2, units_per_group_1, units_per_group_2)

3. **Source**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
   **Reason**: Understanding scaler tile generation
   **Key Information**: Creates a tile filled mostly with zeros, with specific positions containing the packed scaler value for hardware reduce operations.

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity detection
   **Key Information**: DEST capacity varies by sync mode (Full/Half) and accumulation mode (16/32-bit). Ranges from 4 to 16 tiles. Affects chunking in column reduction.

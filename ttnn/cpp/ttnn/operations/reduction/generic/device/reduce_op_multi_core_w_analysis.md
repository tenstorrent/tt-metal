# Reduce W (Width Reduction) Implementation Analysis

## Overview

The `reduce_w` operation performs reduction along the width (W) dimension of a tensor using multi-core parallelization. It reduces each tile row to a single output tile, effectively collapsing the W dimension while preserving N, C, and H dimensions.

**Program Factory Path**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Key Characteristics**:
- Supports SUM, AVG, and MAX reduction operations via scaler
- Uses `reduce_helpers.hpp` kernel library for compute (or matmul fallback for SUM)
- Distributes work by tile rows (NC * Ht) across cores
- Interleaved memory layout for both input and output

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile row |
| **Unit size** | Wt tiles (one complete tile row) |
| **Total units** | NC * Ht (number of tile rows) |
| **Loop structure** | Outer: tile rows (distributed to cores), Inner: Wt tiles per row |

A work unit is one complete tile row. Each core processes `num_rows_per_core` complete tile rows, where each row contains Wt input tiles that reduce to 1 output tile.

**Dimension Definitions** (from program factory lines 26-29):
- `W = shape[3]` - Width dimension
- `H = shape[2]` - Height dimension
- `NC = shape[1] * shape[0]` - Batch dimensions (N * C)
- `Wt = W / TILE_WIDTH` - Width in tiles
- `Ht = H / TILE_HEIGHT` - Height in tiles
- `num_rows = NC * Ht` - Total tile rows to process

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, 1] |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | Any supported (bfloat16, float32, etc.) | Same as input or specified |

### Layout Transformations

- **Input**: Must be in TILE_LAYOUT (32x32 tiles). If row-major, caller tilizes beforehand.
- **Output**: Remains in TILE_LAYOUT. Width dimension is reduced but tile structure preserved.
- **Reduction ratio**: `Wt:1` - Wt input tiles produce 1 output tile per row.

## Data Flow Pattern

### High-Level Flow

```
Input Tensor (DRAM) --> Reader Kernel --> CB_in0 (input tiles)
                                     --> CB_2 (scaler tile, generated once)
                                            |
                                            v
                                     Compute Kernel (reduce along W)
                                            |
                                            v
                                      CB_out (c_3) --> Writer Kernel --> Output Tensor (DRAM)
```

### Detailed Step-by-Step Flow

1. **Reader Kernel** (`reader_unary_reduce_universal_start_id.cpp`):
   - Generates scaler tile once into CB_2 using `generate_reduce_scaler()`
   - Reads input tiles sequentially from DRAM using TensorAccessor
   - Pushes tiles one at a time to CB_in0

2. **Compute Kernel** (`reduce_w.cpp`):
   - For each tile row (Ht iterations per batch):
     - Accumulates Wt tiles using `reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
     - Scaler is applied during reduction (for AVG, scaler = 1/Wt)
     - Packs single output tile to CB_out

3. **Writer Kernel** (`writer_unary_universal_start_id.cpp`):
   - Reads output tiles from CB_out one at a time
   - Writes to DRAM using TensorAccessor

### Reduction Mathematics

For **SUM** reduction: `output[h] = sum(input[h][0:Wt]) * scaler`

For **AVG** reduction: `output[h] = sum(input[h][0:Wt]) * (1/Wt)`

For **MAX** reduction: `output[h] = max(input[h][0:Wt])`

The scaler is baked into a tile by the reader and applied during the reduce_tile operation.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_2 | cb_scaler | Scaler tile | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_3 | cb_out | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |

### CB Configuration Details

**CB c_0 (Input)** (lines 62-67):
- Holds input tiles for streaming processing
- Double-buffered for overlap between reader and compute
- Data format matches input tensor dtype

**CB c_2 (Scaler)** (lines 69-73):
- Holds the scaler tile (bfloat16 format, hardcoded)
- Generated once at kernel start, persists for entire program
- Scaler value packed as two bfloat16 values in uint32

**CB c_3 (Output)** (lines 75-80):
- Holds reduced output tiles
- Double-buffered for overlap between compute and writer
- Data format matches output tensor dtype

## Pipeline Pattern Summary

All three circular buffers are double-buffered (capacity = 2 * block_size), enabling:
- **Reader-Compute overlap**: Reader can prepare next tile while compute processes current
- **Compute-Writer overlap**: Writer can write previous tile while compute produces next

This is a standard **streaming pipeline** pattern where tiles flow through reader -> compute -> writer with double buffering at each stage boundary.

## Index Calculations

### Input Tile Index Mapping

The input tensor is accessed sequentially in row-major tile order:
```
tile_index = start_id + i  (for i in 0..num_tensor_tiles_per_core)
```

Where:
- `start_id = num_tiles_read` accumulated across cores
- `num_tensor_tiles_per_core = num_rows_per_core * Wt`

### Output Tile Index Mapping

Output tiles are written at reduced indices:
```
output_tile_index = start_id / Wt + j  (for j in 0..num_output_tiles)
```

Where:
- `out_dim_divider = Wt` (reduction factor)
- `num_output_tiles = num_tensor_tiles_per_core / Wt`

### TensorAccessor Usage

Both reader and writer use `TensorAccessor` for address generation:

**Reader** (line 33, kernel):
```cpp
auto tensor_accessor = TensorAccessor(tensor_args, src_addr, tile_bytes);
noc_async_read_page(i, tensor_accessor, l1_write_addr);
```

**Writer** (line 23, kernel):
```cpp
auto tensor_accessor = TensorAccessor(tensor_args, dst_addr, tile_bytes);
noc_async_write_page(i, tensor_accessor, l1_read_addr);
```

## Memory Access Patterns

### Read Pattern

**Pattern**: Sequential, tile-by-tile streaming

For each core:
1. Start at `start_id = num_tiles_read` (cumulative offset)
2. Read tiles sequentially: `start_id, start_id+1, ..., start_id+num_tensor_tiles_per_core-1`
3. Each tile is read, processed, and popped before next is needed

**Access granularity**: Single tile per NoC read
**Synchronization**: `noc_async_read_barrier()` after each tile

### Write Pattern

**Pattern**: Sequential, tile-by-tile streaming

For each core:
1. Start at `output_start_id = start_id / Wt`
2. Write tiles sequentially: one output tile per Wt input tiles
3. Each tile is written and popped before next is produced

**Access granularity**: Single tile per NoC write
**Synchronization**: `noc_async_write_barrier()` after each tile

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized 2D grid) |
| **Grid dimensions** | Up to device compute grid size |
| **Total cores** | min(num_rows, max_cores) |
| **Work per core** | num_rows_per_core_group_1 or num_rows_per_core_group_2 rows |
| **Load balancing** | Two-group split for remainder handling |

### Work Distribution (lines 48-60)

Uses `tt::tt_metal::split_work_to_cores()` to distribute `num_rows = NC * Ht` across available cores:

```cpp
std::tie(num_cores, all_cores, core_group_1, core_group_2,
         num_rows_per_core_group_1, num_rows_per_core_group_2) =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
```

**Two-Group Strategy**:
- `core_group_1`: Gets `num_rows / num_cores + 1` rows (handles remainder)
- `core_group_2`: Gets `num_rows / num_cores` rows

**Core Enumeration** (lines 141-152):
- If `sub_core_grids` specified: iterate through ranges in order
- Otherwise: use `grid_to_cores()` for standard row-major enumeration

### Per-Core Work Assignment (lines 153-184)

```cpp
for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
    // Determine rows for this core
    uint32_t num_rows_per_core = core_group_1.contains(core) ?
        num_rows_per_core_group_1 : num_rows_per_core_group_2;

    uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;

    // Reader: reads all input tiles for assigned rows
    SetRuntimeArgs(reader_kernel_id, core, {
        src_addr, num_tensor_tiles_per_core, num_tiles_read
    });

    // Writer: writes reduced output tiles
    SetRuntimeArgs(writer_kernel_id, core, {
        dst_addr, num_tensor_tiles_per_core / Wt, num_tiles_read / Wt
    });

    num_tiles_read += num_tensor_tiles_per_core;
}
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scaler_value | uint32_t | Two bfloat16 scaler values packed into uint32 |
| 1+ | tensor_args | TensorAccessorArgs | Input tensor access parameters |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer ID (c_3) |
| 1+ | tensor_args | TensorAccessorArgs | Output tensor access parameters |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile rows to process (num_rows_per_core_group_X) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row to reduce) |
| 2 | NC | uint32_t | Always 1 (batch dimension handled by row distribution) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM address |
| 1 | num_tiles | uint32_t | Total tiles to read (num_rows * Wt) |
| 2 | start_id | uint32_t | Starting tile index |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | num_tiles | uint32_t | Tiles to write (num_rows) |
| 2 | start_id | uint32_t | Starting output tile index |

### Kernel Defines

Generated by `reduce_op_utils::get_defines()` (from reduce_op.cpp lines 20-38):

| Define | Value (for W reduction) | Description |
|--------|-------------------------|-------------|
| REDUCE_OP | PoolType::SUM or PoolType::MAX | Reduction operation type |
| REDUCE_DIM | ReduceDim::REDUCE_ROW | Reduction dimension (W maps to REDUCE_ROW) |
| REDUCE_ROW_SUM_VIA_MM | 1 (only for SUM) | Enables matmul-based implementation |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_reduce_universal_start_id | RISCV_0 | NOC0 | DRAM | CB_0, CB_2 | Generate scaler, read tiles |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`

**Key Logic**:
1. Generate scaler tile once using `generate_reduce_scaler(cb_id_in2, scaler)` (line 22)
2. Loop through assigned tiles (lines 36-42):
   - `cb_reserve_back(cb_id_in0, 1)` - Reserve space
   - `noc_async_read_page(i, tensor_accessor, l1_write_addr)` - Read tile
   - `noc_async_read_barrier()` - Wait for completion
   - `cb_push_back(cb_id_in0, 1)` - Make available to compute

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w | RISCV_2+ | N/A | CB_0, CB_2 | CB_3 | W-dimension reduction |

**File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`

**Key Logic** (using reduce_helpers.hpp, lines 20-30):
```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    tt::CBIndex::c_0,  // input CB
    tt::CBIndex::c_2,  // scaler CB
    tt::CBIndex::c_3,  // output CB
    Ht, Wt, NC);
```

**Reduction Flow** (from reduce_helpers.hpp REDUCE_ROW path, lines 266-325):
1. For each tile row (ht in 0..Ht):
   - `tile_regs_acquire()` - Acquire DEST registers
   - For each tile in row (wt in 0..Wt):
     - `cb_wait_front(icb, 1)` - Wait for input tile
     - `reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(icb, icb_scaler, 0, 0, 0)` - Accumulate
     - `cb_pop_front(icb, 1)` - Free input tile
   - `cb_reserve_back(ocb, 1)` - Reserve output space
   - `tile_regs_commit(); tile_regs_wait()` - Sync DEST
   - `pack_tile(0, ocb)` - Pack result to output CB
   - `tile_regs_release()` - Release DEST
   - `cb_push_back(ocb, 1)` - Make output available

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_universal_start_id | RISCV_1 | NOC1 | CB_3 | DRAM | Write tiles |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_universal_start_id.cpp`

**Key Logic** (lines 29-37):
```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    noc_async_write_page(i, tensor_accessor, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, 1);
}
```

## Implementation Notes

### Scaler Generation

The scaler tile is generated in-kernel by the reader using `generate_reduce_scaler()`:

1. Tile is filled with zeros using NoC reads from `MEM_ZEROS_BASE`
2. First 8 positions of each 128-element face are set to the packed scaler value
3. This creates a tile where only the diagonal elements have the scaler value
4. For AVG: scaler = 1/W, applied during reduction
5. For SUM: scaler = 1.0 (no scaling)
6. For MAX: scaler is ignored

### Alternative Matmul Path

For SUM reduction, `REDUCE_ROW_SUM_VIA_MM` define enables a matmul-based implementation (lines 32-55 of reduce_w.cpp):
- Uses `matmul_tiles()` with scaler tile as multiplier
- Simpler accumulation loop without reduce primitives
- May have different performance characteristics

### Two Core Groups

The split into core_group_1 and core_group_2 handles non-divisible work:
- Separate compute kernels are created for each group with different `Ht` values
- This ensures all work is completed even when num_rows doesn't divide evenly

### Program Caching

The factory implements `override_runtime_arguments()` for efficient tensor address updates:
- Only buffer addresses are updated on cache hit
- Kernel recompilation not needed for same-shape tensors

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the reduce operation work in tt-metal? Specifically, what is the reduce_tile function and how does it perform reduction along different dimensions?"
   **Reason**: Needed to understand the core reduction mechanism and how REDUCE_ROW maps to W-dimension reduction
   **Key Findings**: `reduce_tile` accumulates values in DEST registers. REDUCE_ROW reduces width, storing result in first column. Requires `reduce_init` before and `reduce_uninit` after. Scaler CB is important for AVG operations.

2. **Query**: "What is the split_work_to_cores function in tt-metal and how does it distribute work across cores?"
   **Reason**: Understanding core distribution strategy for multi-core operations
   **Key Findings**: Returns two core groups - core_group_1 handles remainder (gets +1 work units), core_group_2 gets base amount. Returns CoreRangeSets for kernel assignment.

3. **Query**: "What is TensorAccessor in tt-metal? How does it work with TensorAccessorArgs?"
   **Reason**: Understanding how tile indices map to physical DRAM addresses
   **Key Findings**: TensorAccessor abstracts tile-to-address mapping. TensorAccessorArgs configures compile-time vs runtime parameters. `noc_async_read_page()` and `noc_async_write_page()` use it for address generation.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
   **Reason**: Understanding the reduce helper library used by compute kernel
   **Key Information**: Unified `reduce<PoolType, ReduceDim>()` function handles all reduction patterns. Supports STREAMING input mode with automatic CB management. Handles DEST register acquire/commit/wait/release cycle.

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST register capacity limits
   **Key Information**: DEST capacity varies by sync mode and accumulation mode (4-16 tiles). Auto-detected via JIT headers.

3. **Source**: `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
   **Reason**: Understanding how scaler tile is created
   **Key Information**: Zeros tile with scaler values in specific positions. Scaler is double-packed bfloat16. Uses NoC reads from MEM_ZEROS_BASE for efficient zeroing.

4. **Source**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp`
   **Reason**: Understanding reduce operation defines and configuration
   **Key Information**: `get_defines()` maps ReduceOpDim::W to ReduceDim::REDUCE_ROW. REDUCE_ROW_SUM_VIA_MM enables matmul path for SUM operations.

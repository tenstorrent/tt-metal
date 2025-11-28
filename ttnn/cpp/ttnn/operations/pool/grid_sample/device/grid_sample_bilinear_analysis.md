# Grid Sample Bilinear Implementation Analysis

## Overview

The `grid_sample_bilinear` operation performs spatial sampling of an input tensor using bilinear interpolation at locations specified by a grid tensor. This is commonly used in computer vision for spatial transformations such as affine transforms, perspective warping, and optical flow-based image warping.

**Program Factory Path**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp`

### Operation Summary
- **Input Tensor**: Shape (N, H_in, W_in, C) - the source image/feature map to sample from
- **Grid Tensor**: Shape (N, H_grid, W_grid, 2*K) or (N, H_grid, W_grid, 6*K) for precomputed mode
- **Output Tensor**: Shape (N, H_grid, W_grid, C) or (N, H_grid, W_grid*K, C) depending on `batch_output_channels` flag

The operation samples the input tensor at normalized coordinates specified in the grid tensor, using bilinear interpolation to compute weighted averages of the four nearest neighbor pixels.

## Work Unit Definition

**One work unit = One grid point (spatial location)**

For each grid point:
1. Read the (x, y) sampling coordinates from the grid tensor
2. Convert normalized coordinates [-1, 1] to image coordinates
3. Identify the four neighboring pixels (NW, NE, SW, SE)
4. Calculate bilinear interpolation weights
5. Read the four input sticks (entire channel dimension for each corner)
6. Perform weighted reduction to produce one output stick

The work is distributed across cores by dividing the total number of grid sticks (H_grid * W_grid * N) among available cores.

## Data Flow Pattern

### High-Level Flow

```
                    Grid Tensor (in L1 or DRAM)
                              |
                              v
        +---------------------+---------------------+
        |                Reader Kernel              |
        |  - Read grid coordinates                  |
        |  - Calculate image coordinates & weights  |
        |  - Read 4 corner input sticks from DRAM   |
        +---------------------+---------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
        Input CB (4 sticks)            Scalar CB (weights)
              |                               |
              +---------------+---------------+
                              |
                              v
        +---------------------+---------------------+
        |              Compute Kernel               |
        |  - Tilize input sticks                    |
        |  - Apply weights (scalar multiply)        |
        |  - Reduce across 4 corners (column sum)   |
        |  - Untilize output                        |
        +---------------------+---------------------+
                              |
                              v
                        Output CB
                              |
                              v
        +---------------------+---------------------+
        |              Writer Kernel                |
        |  - Write output sticks to DRAM            |
        |  (Interleaved mode only - sharded mode    |
        |   writes directly from output CB to L1)   |
        +---------------------+---------------------+
```

### Detailed Data Flow

1. **Grid Coordinate Processing**:
   - For standard grids: Read (x, y) pair in [-1, 1] range, convert to image coordinates
   - For precomputed grids: Read 6 values per point (h0, w0, weight_nw, weight_ne, weight_sw, weight_se)

2. **Input Data Fetching**:
   - Calculate 4 corner indices: (h0,w0), (h0,w1), (h1,w0), (h1,w1)
   - Read 4 input sticks via NOC from DRAM/L1
   - Each stick contains the full channel dimension (C elements)

3. **Compute Phase**:
   - Input data is tilized with scalar weights
   - Column reduction (SUM) produces weighted average across the 4 corners
   - Output is untilized to row-major format

4. **Output Writing**:
   - For interleaved: Writer kernel writes to DRAM
   - For sharded: Output CB is directly mapped to output shard in L1

## Circular Buffer Configuration

### CB_0 (grid_cb_index): Grid Data Buffer
- **Size**:
  - Interleaved: `grid_stick_size * 1` (single-buffered)
  - Sharded: `grid_stick_size * grid_nsticks_per_core` (entire shard)
- **Producer**: Reader kernel (interleaved) / Pre-loaded (sharded)
- **Consumer**: Reader kernel (reads coordinates to calculate indices)
- **Usage Pattern**: For sharded mode, the grid CB points directly to the grid tensor's L1 buffer. For interleaved mode, grid sticks are read one at a time from DRAM.
- **Data Format**: `grid_cb_data_format` (matches grid tensor dtype)

### CB_1 (input_cb_index_0): Primary Input Data Buffer
- **Size**: `in_ntiles_c * TILE_HW * element_size * BUFFERING_FACTOR(2)`
- **Producer**: Reader kernel (RISCV_0 / BRISC)
- **Consumer**: Compute kernel
- **Usage Pattern**: Double-buffered to overlap compute with data movement. Contains 4 input sticks (4 corners * C channels) laid out for tilized access.
- **Data Format**: `input_cb_data_format` (matches input tensor dtype)

### CB_2 (input_cb_index_1): Secondary Input Data Buffer (Split Reader Only)
- **Size**: Same as CB_1
- **Producer**: Reader kernel (RISCV_1 / NCRISC) when split reader is enabled
- **Consumer**: Compute kernel
- **Usage Pattern**: Only created for sharded mode with split reader. Allows two RISC-V cores to read input data in parallel.
- **Condition**: Created only when `is_sharded && enable_split_reader`

### CB_3 (scalar_cb_index_0): Primary Interpolation Weights Buffer
- **Size**: `TILE_HW * element_size * BUFFERING_FACTOR(2)`
- **Producer**: Reader kernel (RISCV_0)
- **Consumer**: Compute kernel
- **Usage Pattern**: Contains 4 bilinear weights (NW, NE, SW, SE) packed into a tile format for the reduction operation.
- **Data Format**: Same as input data format

### CB_4 (scalar_cb_index_1): Secondary Weights Buffer (Split Reader Only)
- **Size**: Same as CB_3
- **Producer**: Reader kernel (RISCV_1) when split reader is enabled
- **Consumer**: Compute kernel
- **Condition**: Created only when `is_sharded && enable_split_reader`

### CB_5 (output_cb_index): Output Data Buffer
- **Size**:
  - Interleaved: `FACE_WIDTH * element_size * out_ntiles_c * BUFFERING_FACTOR(2)`
  - Sharded: `FACE_WIDTH * element_size * output_nsticks_per_core * out_ntiles_c`
- **Producer**: Compute kernel
- **Consumer**: Writer kernel (interleaved) / Direct L1 output (sharded)
- **Usage Pattern**:
  - For interleaved: Double-buffered for pipelining
  - For sharded: Maps directly to output tensor buffer in L1
- **Data Format**: `output_cb_data_format` (matches output tensor dtype)

## Index Calculations

### Coordinate Transformation (Standard Grid Mode)

Grid coordinates are in normalized range [-1, 1]. The transformation to image coordinates depends on the `align_corners` setting (this implementation uses `align_corners=False`):

```cpp
// Compute scaling factors
const float height_scale = input_height * 0.5f;
const float height_offset = height_scale - 0.5f;
const float width_scale = input_width * 0.5f;
const float width_offset = width_scale - 0.5f;

// Transform grid coordinates to image coordinates
h_coord_image = h_coord_rel * height_scale + height_offset;
w_coord_image = w_coord_rel * width_scale + width_offset;

// Get corner pixel indices
h0 = floor(h_coord_image);  // North row
h1 = h0 + 1;                 // South row
w0 = floor(w_coord_image);  // West column
w1 = w0 + 1;                 // East column
```

### Bilinear Weight Calculation

```cpp
// Fractional parts
h_frac = h_coord_image - h0;
w_frac = w_coord_image - w0;
h_frac_inv = 1.0 - h_frac;
w_frac_inv = 1.0 - w_frac;

// Bilinear weights (with boundary validation)
weight_nw = h_frac_inv * w_frac_inv;  // (1-h_frac) * (1-w_frac)
weight_ne = h_frac_inv * w_frac;       // (1-h_frac) * w_frac
weight_sw = h_frac * w_frac_inv;       // h_frac * (1-w_frac)
weight_se = h_frac * w_frac;           // h_frac * w_frac
```

### Input Stick Index Calculation

```cpp
// For each corner (e.g., North-West)
stick_index = batch_offset + (h0 * input_width) + w0;

// batch_offset advances through batches as grid points are processed
batch_offset = curr_batch * input_height * input_width;
```

### Grid Batching Factor

The grid tensor can contain multiple coordinate pairs per spatial location (K pairs), allowing batch processing of multiple sampling points:

```cpp
// For standard grid: 2 elements per point (x, y)
// For precomputed grid (bilinear): 6 elements per point (h0, w0, weight_nw, weight_ne, weight_sw, weight_se)
grid_batching_factor = grid_tensor.logical_shape()[-1] / elements_per_point;
```

## Memory Access Patterns

### Read Pattern

**Grid Tensor Access**:
- Interleaved: Sequential stick-by-stick reads from DRAM using TensorAccessor
- Sharded: Grid data pre-loaded in L1, accessed via direct pointer

**Input Tensor Access**:
- Random access pattern based on grid coordinates
- For each grid point, 4 potentially non-contiguous sticks are read
- Uses TensorAccessor for DRAM bank mapping
- Sticks are read in order: NW, NE, SW, SE (4 corners)

**Pattern per grid point**:
```
Input Read Order:
1. North-West corner: (h0, w0) -> stick[batch * H * W + h0 * W + w0]
2. North-East corner: (h0, w1) -> stick[batch * H * W + h0 * W + w1]
3. South-West corner: (h1, w0) -> stick[batch * H * W + h1 * W + w0]
4. South-East corner: (h1, w1) -> stick[batch * H * W + h1 * W + w1]
```

### Write Pattern

**Interleaved Mode**:
- Sequential writes of output sticks to DRAM
- Each stick written after compute produces one output

**Sharded Mode**:
- Output written directly to L1 shard
- No explicit writer kernel needed - output CB maps to output buffer

### Boundary Handling (Zero Padding Mode)

When sampling coordinates fall outside the input bounds:
1. Input CB is pre-zeroed at kernel start using `zero_out_tiles()`
2. Out-of-bounds corner reads are skipped (no NOC read issued)
3. Corresponding weights are set to zero
4. The reduction naturally produces correct result with zero-weighted corners

## Core Distribution Strategy

### Interleaved Mode

Uses `tt::tt_metal::split_work_to_cores()` to distribute grid sticks:

```cpp
auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range,
      num_sticks_1, num_sticks_2] = split_work_to_cores(compute_grid_size, total_grid_nsticks);
```

- **Core Group 1**: Processes `num_sticks_per_core_group_1` grid sticks each
- **Core Group 2**: Processes `num_sticks_per_core_group_2` grid sticks each (handles remainder)
- Work is divided to minimize imbalance

### Sharded Mode

Uses the grid tensor's shard specification:

```cpp
const auto grid_shard_spec = grid_tensor.shard_spec().value();
grid_nsticks_per_core = grid_shard_spec.shape[0];
num_cores = (total_grid_nsticks + grid_nsticks_per_core - 1) / grid_nsticks_per_core;
all_cores = grid_shard_spec.grid;
```

- Each core processes its assigned shard of grid sticks
- Grid data is already in L1 (no reader fetch needed)

### Split Reader Pattern

When enabled (`enable_split_reader = true`), two reader kernels run in parallel:

**Reader 0 (RISCV_0 / BRISC)**:
- Processes even-indexed grid points within each shard
- Uses NOC0 for input reads
- Writes to `input_cb_index_0` and `scalar_cb_index_0`

**Reader 1 (RISCV_1 / NCRISC)**:
- Processes odd-indexed grid points within each shard
- Uses NOC1 for input reads
- Writes to `input_cb_index_1` and `scalar_cb_index_1`

Split reader is enabled when:
1. Grid tensor is sharded AND
2. Either: grid is not precomputed (coordinate computation dominates), OR
3. Input tensor is in L1 (not DRAM) AND
4. Architecture-specific conditions (always on WH_B0, channel-dependent on Blackhole)

## Arguments

### Compile-Time Arguments

#### Reader Kernel (Interleaved & Sharded)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `input_cb_index_0` | CB index for input data (primary reader) |
| 1 | `grid_cb_index` | CB index for grid data |
| 2 | `scalar_cb_index_0` | CB index for interpolation weights (primary) |
| 3 | `input_stick_nbytes` | Size of one input stick in bytes |
| 4 | `grid_stick_nbytes` | Size of one grid stick in bytes |
| 5 | `input_batch` | Batch dimension of input tensor |
| 6 | `input_height` | Height dimension of input tensor |
| 7 | `input_width` | Width dimension of input tensor |
| 8 | `grid_batching_factor` | Number of coordinate pairs per grid stick |
| 9 | `grid_dtype` | Data type of grid tensor (0=BF16, 1=FP32) |
| 10 | `grid_hw` | H_grid * W_grid (grid spatial size) |
| 11 | `use_precomputed_grid` | Whether grid contains precomputed coordinates/weights |
| 12+ | TensorAccessorArgs | Input tensor accessor parameters |
| (interleaved only) | TensorAccessorArgs | Grid tensor accessor parameters |

**Additional sharded-only arguments**:
| Index | Name | Description |
|-------|------|-------------|
| 12 | `enable_split_reader` | Whether split reader is active |
| 13 | `reader_id` | 0 for RISCV_0, 1 for RISCV_1 |
| 14 | `grid_nsticks_per_core` | Number of grid sticks in this core's shard |

#### Compute Kernel

| Index | Name | Description |
|-------|------|-------------|
| 0 | `in_ntiles_c` | Number of tiles in channel dimension |
| 1 | `REDUCTION_SIZE` | Window size for reduction (4 for bilinear) |
| 2 | `enable_split_reader` | Whether split reader is active |
| 3 | `total_interpolations` | Total grid points to process on this core |
| 4 | `channels_per_shard` | Channel dimension size |
| 5 | `in_nblocks_c` | Number of channel blocks |
| 6 | `MAX_ROWS_FOR_REDUCTION` | 16 (face height) |
| 7-10 | CB indices | Input and scalar CB indices for both readers |
| 11-16 | Unused | Reserved for other pool operations |
| 17 | `output_cb_index` | Output circular buffer index |
| 18 | Unused | Reserved |
| 19 | `ONE_SCALAR_PER_CORE` | False for grid_sample |
| 20 | `pre_tilize_cb_id` | Unused (32) for row-major output |
| 21 | `is_output_tiled` | False for grid_sample |
| 22 | `is_output_block_format` | False for grid_sample |
| 23-30 | Unused | Reserved for pooling operations |

**Defines**:
- `REDUCE_OP = PoolType::SUM` (weighted average via sum reduction)
- `REDUCE_DIM = ReduceDim::REDUCE_COL` (reduce across the 4 corners)

#### Writer Kernel (Interleaved Only)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `output_cb_index` | Output circular buffer index |
| 1 | `output_stick_size` | Size of one output stick in bytes |
| 2 | `out_ntiles_c` | Number of tiles per output stick |
| 3+ | TensorAccessorArgs | Output tensor accessor parameters |

### Runtime Arguments

#### Reader Kernel

**Interleaved Mode**:
| Index | Name | Description |
|-------|------|-------------|
| 0 | `input_buffer_address` | Base address of input tensor in DRAM |
| 1 | `grid_buffer_address` | Base address of grid tensor in DRAM |
| 2 | `grid_sticks` | Number of grid sticks to process |
| 3 | `grid_processed` | Starting grid stick index |

**Sharded Mode**:
| Index | Name | Description |
|-------|------|-------------|
| 0 | `input_buffer_address` | Base address of input tensor |
| 1 | `grid_stick_offset` | Global starting grid stick index for this core |

#### Writer Kernel (Interleaved Only)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `output_buffer_address` | Base address of output tensor in DRAM |
| 1 | `output_sticks` | Number of output sticks to write |
| 2 | `output_processed` | Starting output stick index |

## Kernel Implementations

### Reader Kernel (Interleaved): `reader_grid_sample_interleaved_start_id.cpp`

**File**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp`

**Responsibilities**:
1. Initialize input CB to zeros (for boundary handling)
2. Read grid sticks sequentially from DRAM
3. For each grid stick, process `grid_batching_factor` coordinate pairs
4. Calculate image coordinates and bilinear weights
5. Read 4 corner input sticks from DRAM
6. Store weights in scalar CB

**Input**: Grid tensor address, input tensor address
**Output**: Populated input CB and scalar CB

**Key Logic**:
```cpp
// Main processing loop
for (uint32_t spatial_pos = start_page_id; spatial_pos < end_id; ++spatial_pos) {
    // Read grid stick
    noc_async_read(grid_noc_addr, l1_write_grid_addr, grid_stick_nbytes);

    // Process each coordinate pair in the grid stick
    for (uint32_t grid_idx = 0; grid_idx < grid_batches; ++grid_idx) {
        process_grid_point<...>(grid_ptr, grid_idx, input_tensor_accessor, batch_offset);
    }
}
```

### Reader Kernel (Sharded): `reader_grid_sample_sharded.cpp`

**File**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded.cpp`

**Responsibilities**:
1. Initialize input CB to zeros
2. Access pre-loaded grid data in L1
3. Support split reader pattern (interleaved grid point processing)
4. Process only assigned grid points (even or odd based on `reader_id`)

**Input**: Grid data in L1 (via sharded CB), input tensor address
**Output**: Populated input CB and scalar CB

**Key Logic**:
```cpp
// Split reader: advance at start if reader_id == 1
if constexpr (split_reader && reader_id == 1) {
    advance_grid_index_bounded(...);  // Skip to first odd-indexed point
}

while (grid_stick_idx < grid_nsticks_per_core) {
    process_grid_point<...>(grid_stick_ptr, in_grid_row_idx, input_tensor_accessor, batch_offset);

    advance_grid_index_bounded(...);  // Move to next point
    if constexpr (split_reader) {
        advance_grid_index_bounded(...);  // Skip the other reader's point
    }
}
```

### Common Reader Logic: `grid_sample_reader_common.hpp`

**File**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp`

**Key Functions**:

1. **`GridCoordinateReader::read_grid_point`**:
   - Reads grid coordinates (standard or precomputed)
   - Calculates corner pixel indices
   - Computes bilinear interpolation weights

2. **`read_four_corner_inputs`**:
   - Issues 4 NOC async reads for the corner pixels
   - Skips reads for out-of-bounds coordinates

3. **`process_grid_point`**:
   - Combines coordinate reading and input fetching
   - Manages CB space reservation
   - Handles boundary validation and weight zeroing

### Compute Kernel: `compute_pool_2d.cpp`

**File**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp`

**Responsibilities**:
1. Wait for input data and weights from reader(s)
2. Tilize input sticks with scalar multiplication
3. Perform column reduction (weighted sum across 4 corners)
4. Untilize output to row-major format
5. Push results to output CB

**Input**: Input CB (4 sticks), Scalar CB (4 weights)
**Output**: Output CB (1 output stick)

**Key Logic**:
```cpp
for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
    // Select CB based on split reader (alternates for n even/odd)
    const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
    const uint32_t curr_scalar_cb_id = !reader0 ? in_scalar_cb_id_1 : in_scalar_cb_id_0;

    cb_wait_front(curr_scalar_cb_id, 1);  // Wait for weights

    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
        tile_regs_acquire();

        for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
            cb_wait_front(curr_in_cb_id, 1);  // Wait for input

            // Unpack and tilize input with scalar multiply
            unpack_tilizeA_B_block<...>(curr_in_cb_id, curr_scalar_cb_id, ...);

            // Reduce across the 4 corners (column reduction)
            for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                reduce_tile_math(math_tile_idx, num_faces_in_input_tile);
            }

            cb_pop_front(curr_in_cb_id, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack and untilize result to output CB
        pack_untilize_dest<...>(out_cb_id, ...);
        cb_push_back(out_cb_id, output_faces);

        tile_regs_release();
    }

    cb_pop_front(curr_scalar_cb_id, 1);
}
```

### Writer Kernel: `writer_grid_sample_interleaved.cpp`

**File**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp`

**Responsibilities**:
1. Wait for output data from compute kernel
2. Write output sticks to DRAM using TensorAccessor

**Input**: Output CB with computed results
**Output**: Output tensor in DRAM

**Key Logic**:
```cpp
for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; stick_id++) {
    // Wait for one complete output stick (ntiles_c pages)
    cb_wait_front(cb_id_out0, ntiles_c);

    uint64_t base_l1_read_addr = get_read_ptr(cb_id_out0);
    uint64_t dst_noc_addr = s0.get_noc_addr(stick_id);

    // Write the complete stick to DRAM
    noc_async_write(base_l1_read_addr, dst_noc_addr, output_stick_size);
    noc_async_write_barrier();

    // Free the consumed pages
    cb_pop_front(cb_id_out0, ntiles_c);
}
```

## Implementation Notes

### Optimization: Split Reader Pattern

The split reader pattern doubles the effective data movement bandwidth by utilizing both RISC-V cores (BRISC and NCRISC) for reading input data:

**Benefits**:
- Halves coordinate computation time (both cores compute in parallel)
- Doubles NoC read bandwidth (NOC0 and NOC1 used simultaneously)
- Particularly effective when grid is not precomputed (coordinate computation dominates)

**Trade-offs**:
- Requires additional CBs (doubles memory for input and scalar buffers)
- NOC1 DRAM reads are slower than NOC0 on some architectures
- Disabled for DRAM input when precomputed grid is used

**Architecture-specific behavior**:
- Wormhole B0: Always enabled for sharded grids
- Blackhole: Enabled when channels <= 224 (avoids unpacker bottleneck)

### Optimization: Precomputed Grid

The `use_precomputed_grid` mode moves coordinate transformation and weight calculation from device to host:

**Standard Grid**: 2 elements per point (x, y normalized coordinates)
**Precomputed Grid**: 6 elements per point (h0, w0, weight_nw, weight_ne, weight_sw, weight_se)

**Benefits**:
- Eliminates floating-point math on device (floor, multiply, subtract)
- Reduces reader kernel complexity and execution time
- Amortizes computation cost for repeated inferences with same grid

### Boundary Handling: Zero Padding Mode

The implementation handles out-of-bounds sampling coordinates using zero padding:

1. **Pre-zeroing**: Input CB is completely zeroed at kernel start using `zero_out_tiles()`
2. **Selective reads**: NOC reads are skipped for invalid coordinates
3. **Weight zeroing**: Invalid corner weights set to 0.0
4. **Automatic result**: Reduction produces correct weighted average

This approach avoids conditional logic in the compute kernel and handles all boundary cases uniformly.

### Compute Kernel Reuse

The grid_sample operation reuses the generic `compute_pool_2d.cpp` kernel designed for pooling operations. Configuration is via defines:

- `REDUCE_OP = PoolType::SUM`: Use sum reduction (weights provide averaging)
- `REDUCE_DIM = ReduceDim::REDUCE_COL`: Reduce across the 4 input corners

This reuse enables:
- Shared optimization work for both operations
- Consistent tile handling and register management
- Proven correctness from extensive pooling tests

### Memory Efficiency: Sharded Mode

In sharded mode:
- Grid CB points directly to grid tensor's L1 buffer (no copy)
- Output CB points directly to output tensor's L1 buffer (no copy)
- Only input tensor requires NOC reads from DRAM

This minimizes data movement and enables efficient processing of streaming data.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the grid_sample operation and how does bilinear interpolation work in the context of tensor operations?"
   **Reason**: Understanding the fundamental operation being implemented
   **Key Findings**: Grid sample uses a grid tensor to specify sampling locations, bilinear interpolation computes weighted averages of 4 nearest neighbors, precomputed grids can be used to move coordinate computation to host.

2. **Query**: "What is TensorAccessor and how is it used to map logical tensor indices to physical memory locations in kernels?"
   **Reason**: Understanding how reader kernels access tensor data across distributed memory banks
   **Key Findings**: TensorAccessor provides get_noc_addr() to compute NOC addresses from logical page IDs, handles bank mapping for interleaved layouts.

3. **Query**: "What is the split_work_to_cores function and how does it distribute work across multiple cores?"
   **Reason**: Understanding the core distribution strategy for interleaved mode
   **Key Findings**: Divides work into two groups to handle remainder, returns core ranges and work counts per group.

4. **Query**: "What are circular buffers in tt-metal and how do cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front work?"
   **Reason**: Understanding inter-kernel communication mechanism
   **Key Findings**: CBs provide producer-consumer synchronization in L1, reserve/push for producers, wait/pop for consumers.

5. **Query**: "What does reduce_tile_math do and how does reduction work for pooling?"
   **Reason**: Understanding compute kernel's reduction operation
   **Key Findings**: Performs math-only reduction on tiles in DST registers, supports MAX, SUM operations across different dimensions.

6. **Query**: "What does unpack_tilizeA_B_block do?"
   **Reason**: Understanding how input data is prepared for compute
   **Key Findings**: Tilizes input from two CBs simultaneously, handles row-major to tiled format conversion with scalar multiplication.

7. **Query**: "What does pack_untilize_dest do?"
   **Reason**: Understanding how output is prepared for writing
   **Key Findings**: Packs data from DST registers to CB with untilization, converts tiled to row-major format.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding core architecture and programming model
   **Key Information**: Tensix core structure, kernel types (reader/compute/writer), circular buffer usage patterns.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding TensorAccessor utility
   **Key Information**: How logical indices map to physical addresses, bank distribution for interleaved tensors.

3. **Source**: `ttnn/cpp/ttnn/operations/pool/pool_utils.cpp`
   **Reason**: Understanding pool operation configuration and defines
   **Key Information**: `get_defines()` returns REDUCE_OP and REDUCE_DIM settings, scalar calculation for weighted operations.

4. **Source**: `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_utils.cpp`
   **Reason**: Understanding split reader heuristics and grid batching
   **Key Information**: Architecture-specific split reader decisions, grid batching factor calculation, stick size alignment.

5. **Source**: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp`
   **Reason**: Understanding zero_out_tiles helper function
   **Key Information**: Uses NOC reads from zero memory region to initialize CB contents to zero.

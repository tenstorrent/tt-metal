# TTNN Gather

## Overview

The TTNN Gather operation is a high-performance tensor indexing operation optimized for execution on Tenstorrent hardware. It extracts values from an input tensor based on indices provided in an index tensor along a specified dimension. The operation is designed to work efficiently with tensor data divided into tiles and leverages hardware parallelism to maximize throughput.

The gather operation is performed along a specified dimension of the input tensor, with the implementation internally transposing tensors to optimize computation on the last dimension. The operation offers two execution strategies that balance performance with scalability based on tensor size.

## Brief Functional Description

The TTNN Gather operation extracts values from an input tensor using indices specified in an index tensor along a given dimension. The operation returns a tensor with the same shape as the index tensor, where each element contains the value from the input tensor at the corresponding index position.

Both input tensors must have the same number of dimensions, and for all dimensions except the gather dimension, the index tensor size must not exceed the input tensor size.

### Arguments

- **input_tensor (Tensor)**: The source tensor from which values are gathered.
- **dim (int)**: The dimension along which values are gathered.
- **input_index_tensor (Tensor)**: A tensor containing the indices of elements to gather, with the same number of dimensions as the input tensor. Must be of type UINT32 or UINT16.
- **sparse_grad (bool, optional)**: If True, the gradient computation will be sparse. Defaults to False. **Note: Currently not supported.**
- **memory_config (MemoryConfig, optional)**: Specifies memory configuration for the output tensor. Defaults to None.
- **optional_output_tensor (Tensor, optional)**: Preallocated tensor for the gathered values. Defaults to None.
- **sub_core_grids (CoreRangeSet, optional)**: Custom core range set for operation execution. Allows specification of which cores should be used for the operation. Defaults to None (uses all available cores).

### Usage

#### TTNN

```python
import ttnn
import torch

# Create a 2D input tensor
input_tensor = torch.tensor([[10, 20, 30, 40],
                             [50, 60, 70, 80]])

# Create a 2D index tensor
index_tensor = torch.tensor([[3, 0],
                             [2, 1]])

# Convert tensors to ttnn format
input_tensor_ttnn = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
index_tensor_ttnn = ttnn.from_torch(index_tensor, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

# Perform the gather operation along dimension 1
gathered_tensor = ttnn.gather(input_tensor_ttnn, dim=1, index=index_tensor_ttnn)
# Result: gathered_tensor = [[40, 10], [70, 60]]
```

#### Metalium

```cpp
// Create input tensor
const ttnn::Tensor input_tensor = ...;
const ttnn::Tensor input_index_tensor = ...;

// Set gather dimension
const int8_t dim = 1;

// Set sparse gradient flag
const bool sparse_grad = false;

// Optional parameters
const std::optional<ttnn::MemoryConfig> memory_config = std::nullopt;
std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt;

ttnn::Tensor gathered_tensor = ttnn::gather(
    input_tensor,
    dim,
    input_index_tensor,
    sparse_grad,
    memory_config,
    optional_output_tensor
);
```

### Usage Limitations

- **Supported input tensor types**: `BFLOAT16`, `FLOAT32`, `UINT16`, `UINT32`, `INT32`
- **Supported index tensor types**: `UINT32`, `UINT16`
- **Layout**: Both tensors must be in `TILE` layout
- **Memory**: Interleaved DRAM and L1 memory supported; **sharded memory not supported**
- **Gradient**: `sparse_grad=True` is not supported in this implementation
- **Dimension constraints**: `input_index_tensor.size(d) <= input_tensor.size(d)` for all dimensions `d != dim`
- **Core grids**: When `sub_core_grids` is specified, the provided cores will be used for execution; otherwise, the operation uses all available compute cores

## Strategy Selection Overview

The TTNN Gather operation provides two execution strategies, automatically selected based on tensor size to optimize performance and resource utilization. The choice is determined by a **Wt threshold of 60** tiles in width.

| Strategy                        | Description                                                                                         | Selection Criteria                                    | Strengths                                                    | Typical Use Case                          |
|---------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------|
| **Single Row Single Core**      | Each core processes entire rows independently, handling all index tiles in that row sequentially. | `Wt_input ≤ 60` AND `Wt_index ≤ 60`                  | Simple coordination, low memory overhead, efficient for small tensors | Small to medium tensors                   |
| **Single Row Multi Core**       | Multiple cores collaborate to process tiles within each row in parallel across the index dimension. | `Wt_input > 60` OR `Wt_index > 60`                   | High parallelism, scalable to large tensors, better core utilization | Large tensors exceeding threshold         |

### Key Points:

* **Automatic Strategy Selection**: The strategy is automatically chosen based on the `WT_THRESHOLD = 60` constant. If either the input width (`Wt_input`) or index width (`Wt_index`) exceeds this threshold, the multi-core strategy is selected.

* **Single Row Single Core** is optimal for smaller tensors where the overhead of multi-core coordination outweighs the parallelism benefits. Each core independently processes complete rows.

* **Single Row Multi Core** maximizes hardware utilization for larger tensors by distributing index tiles across multiple cores, enabling parallel processing within each row.

## Implementation Architecture

### Data Flow and Tensor Preprocessing

The TTNN Gather implementation employs a sophisticated preprocessing pipeline to optimize hardware execution:

#### 1. **Tensor Transformation Pipeline**

```cpp
// Dimension normalization - transpose to make gather dimension last
const Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);

// Rank normalization - transform to 4D representation
const Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

// Padding application
const Tensor padded_tensor = ttnn::fill_implicit_tile_padding(transformed_tensor, padding_value);
```

#### 2. **Index and Input Tensor Alignment**

The implementation ensures proper tensor alignment through:

- **Index Tensor**: Padded with zeros using `ttnn::fill_implicit_tile_padding(transformed_tensor, 0)`
- **Input Tensor**: Sliced to match index tensor dimensions and padded with minimum float values
- **Shape Matching**: Input tensor is sliced to align with index tensor shape for efficient cell mapping

#### 3. **Post-Processing Reconstruction**

After gather execution, tensors are transformed back to original shape:

```cpp
// For rank ≤ 4: squeeze from 4D, then transpose if needed
output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);
if (!is_dim_last_idx) {
    output_tensor = ttnn::transpose(output_tensor, dim, -1, memory_config);
}

// For rank > 4: transpose in 4D, then reshape to original dimensions
if (!is_dim_last_idx) {
    output_tensor = ttnn::transpose(output_tensor, dim_adj, -1, memory_config);
}
output_tensor = ttnn::reshape(output_tensor, original_shape);
```

## Strategy Implementation Details

### Single Row Single Core Strategy

This strategy assigns complete rows to individual cores, providing simple and efficient processing for smaller tensors.

#### Work Distribution

```cpp
// Core assignment based on height (Ht)
const uint32_t core_h_id = core_loop * total_number_of_cores + core_id;

// Each core processes: Wt_input input tiles → Wt_index output tiles
for (uint32_t w = 0; w < Wt_index; w++) {
    // Process one index tile to produce one output tile
    process_index_tile(w, Wt_input);
}
```

#### Memory Layout

- **Input Tensor CB**: Holds entire row (`Wt_input * input_tile_size`)
- **Index Tensor CB**: Single tile buffer for sequential processing
- **Output Tensor CB**: Single tile buffer for results

#### Core Utilization Pattern

```
Tensor Shape: 2×4 tiles (Ht=2, Wt_input=4, Wt_index=2)

Core Assignment:
├── Core 0: Row 0 [T_in0, T_in1, T_in2, T_in3] → [T_idx0, T_idx1] → [T_out0, T_out1]
└── Core 1: Row 1 [T_in4, T_in5, T_in6, T_in7] → [T_idx2, T_idx3] → [T_out2, T_out3]
```

---

### Single Row Multi Core Strategy

This strategy distributes index tiles across multiple cores within each row, maximizing parallelism for larger tensors.

#### Work Distribution

```cpp
// Core assignment based on index width (Wt_index)
uint32_t current_index_tile_id = core_id;

for (uint32_t h = 0; h < Ht; h++) {
    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Each core processes specific index tiles
        process_index_tile(h * Wt_index + current_index_tile_id);
        current_index_tile_id += total_number_of_cores;
    }
}
```

#### Memory Layout

- **Input Tensor CB**: Double-buffered for continuous data flow (`2 * input_tile_size`)
- **Index Tensor CB**: Single tile buffer per core
- **Output Tensor CB**: Single tile buffer per core

#### Core Utilization Pattern

```
Tensor Shape: 2×8 tiles (Ht=2, Wt_index=8), 4 cores available

Row 0 Distribution:
├── Core 0: T_idx0, T_idx4 → T_out0, T_out4
├── Core 1: T_idx1, T_idx5 → T_out1, T_out5
├── Core 2: T_idx2, T_idx6 → T_out2, T_out6
└── Core 3: T_idx3, T_idx7 → T_out3, T_out7

Row 1: Similar distribution with offset
```

## Core Gather Algorithm

The heart of the gather operation lies in the index-to-value mapping performed within each tile. This algorithm is shared between both strategies and operates at the tile level.

### Tile-Level Index Mapping

```cpp
// Critical mapping code from gather_reader kernels
for (uint32_t count = 0; count < TILE_HW; count++) {
    // 1. Read global index from index tensor
    const uint32_t global_index = get_value_from_tile(
        input_index_tensor_l1_read_addr, count, input_index_tensor_data_format_size);

    // 2. Calculate which input tile contains this index
    const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);  // equivalent to global_index / tile_width

    // 3. Calculate position within the tile
    const uint32_t index_in_local_tile = global_index & TILE_WIDTH_MASK;  // equivalent to global_index % tile_width

    // 4. Map to tile's 2D face structure (16×16 faces within 32×32 tile)
    const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);  // / 16
    const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;             // % 16

    // 5. Calculate final local index considering tile's internal layout
    const uint16_t local_index = tile_idx * (tile_width * tile_height) +          // tile base offset
                                 which_row * (face_size * face_size) +            // row offset in tile
                                 k * face_size +                                  // depth within face
                                 which_col +                                      // column within face
                                 i * (tile_width * face_size);                   // face offset

    // 6. Read value from input tensor and write to output
    const uint32_t value = get_value_from_tile(input_tensor_l1_read_addr, local_index, input_tensor_data_format_size);
    write_value_to_tile(output_tensor_l1_read_addr, count, output_tensor_data_format_size, value);
}
```

### Tile Structure and Face Organization

The algorithm accounts for the internal organization of tiles in Tenstorrent hardware:

```
32×32 Tile Structure:
┌─────────────────┬─────────────────┐
│   Face 0,0      │   Face 0,1      │  ← Each face is 16×16
│   (16×16)       │   (16×16)       │
├─────────────────┼─────────────────┤
│   Face 1,0      │   Face 1,1      │
│   (16×16)       │   (16×16)       │
└─────────────────┴─────────────────┘

Nested Loop Structure:
for i in [0,1]:      # tile_faces (vertical)
    for j in [0,1]:  # tile_faces (horizontal)
        for k in [0,15]:  # face_size (rows within face)
            for l in [0,15]:  # face_size (cols within face)
```

### Index Calculation Optimization

The implementation uses bit manipulation for performance:

```cpp
// Optimized operations using bit shifts and masks
const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);        // Fast division by power of 2
const uint32_t index_in_local_tile = global_index & TILE_WIDTH_MASK;        // Fast modulo operation
const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size); // Fast division by 16
const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;            // Fast modulo by 16

// Equivalent non-optimized operations:
// const uint32_t tile_idx = global_index / tile_width;
// const uint32_t index_in_local_tile = global_index % tile_width;
// const uint32_t which_row = index_in_local_tile / face_size;
// const uint32_t which_col = index_in_local_tile % face_size;
```

## Kernel Architecture

### Reader-Writer Split Architecture

The implementation employs a **split-kernel architecture** where data movement and computation are optimally distributed:

#### Reader Kernel Responsibilities
- **Index Processing**: Reads index tensor tiles from DRAM to L1
- **Core Computation**: Performs the gather index-to-value mapping
- **Output Generation**: Writes gathered values to output circular buffer

#### Writer Kernel Responsibilities
- **Input Data Loading**: Reads input tensor tiles from DRAM to L1
- **Output Data Persistence**: Writes completed output tiles from L1 to DRAM

### Circular Buffer Management

```cpp
// Buffer allocation strategy
constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;    // Input data buffer
constexpr uint32_t input_index_tensor_cb_index = tt::CBIndex::c_1;  // Index data buffer
constexpr uint32_t output_tensor_cb_index = tt::CBIndex::c_2;   // Output data buffer

// Single Row Single Core: Full row buffering
CircularBufferConfig input_cb_config(Wt_input * input_tile_size, {{cb_index, data_format}});

// Single Row Multi Core: Double buffering for continuous flow
CircularBufferConfig input_cb_config(2 * input_tile_size, {{cb_index, data_format}});
```

### Multi-Core Synchronization

The multi-core strategy requires careful synchronization:

```cpp
// Core identification and work distribution
const auto start_tile_id = get_absolute_logical_y() * grid_size_x + get_absolute_logical_x();
uint32_t current_index_tile_id = start_tile_id;

// Distributed processing with stride access
for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
    process_tile(current_index_tile_id);
    current_index_tile_id += total_number_of_cores;  // Skip to next tile for this core
}
```

## Performance Characteristics

### Memory Access Patterns

```cpp
// Single Row Single Core: Batch memory access
for (uint32_t w = 0; w < Wt_input; w++) {
    noc_async_read_tile(h * Wt_input + w, input_tensor_dram, l1_addr);  // Sequential reads
}

// Single Row Multi Core: Interleaved memory access
for (uint32_t wi = 0; wi < Wt_input; wi++) {
    noc_async_read_tile(h * Wt_input + wi, input_tensor_dram, l1_addr);  // Per-tile reads
}
```

---

© Tenstorrent AI ULC 2025

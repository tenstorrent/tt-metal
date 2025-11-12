# Upsample 2D Implementation Analysis

## Overview
The existing 2D upsample operation in TTNN implements nearest-neighbor and bilinear interpolation for 4D tensors (N, H, W, C) in both row-major and tiled layouts. This document summarizes the key concepts and patterns to inform the 3D upsample implementation.

## Key Components

### 1. Program Factory Structure (`upsample_program_factory_multicore_interleaved.cpp`)

The program factory is responsible for:
- Setting up the compute grid and work distribution
- Creating circular buffers for data movement
- Instantiating reader, writer, and compute kernels
- Configuring runtime arguments for each core

#### Input/Output Handling
- **Input tensor**: 4D shape (N, H, W, C)
- **Output tensor**: 4D shape (N, H*scale_h, W*scale_w, C)
- **Layout support**: Both TILE and ROW_MAJOR layouts
- **Memory config**: DRAM interleaved (primary focus), sharded also supported

### 2. Work Distribution Pattern

#### For Row-Major Layout:
- **Work unit**: One input row (stick) of data
- **Total work units**: `N * H * W` (flattened batch, height, width dimensions)
- **Input CB pages**: 1 page per work unit
- **Data size**: `width * element_size` bytes per row

#### For Tiled Layout:
- **Work unit**: One row of tiles
- **Total work units**: Number of tile rows in input
- **Input CB pages**: Number of tiles per row
- **Data size**: Tile size (32x32 elements typically)

#### Core Assignment:
```cpp
const auto [num_cores, all_cores, core_group_1, core_group_2,
            work_per_core_group_1, work_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);
```
- Uses `split_work_to_cores` to distribute work evenly across available cores
- Two core groups handle different amounts of work for load balancing

### 3. Circular Buffer Management

#### Buffer Configuration:
- **CB Index c_0**: Input data buffer
- **CB Index c_16** (or next): Output buffer (for tiled layout)
- **Double buffering**: Applied when cores process multiple work units

#### Size Calculation:
```cpp
// Row-major
aligned_input_unit_size = round_up(input_unit_size, hal::get_dram_alignment());

// Tiled
aligned_input_unit_size = tile_size(input_cb_data_format);
```

### 4. Kernel Architecture

#### Reader Kernel (`reader_upsample_unary_stick_layout_interleaved_start_id.cpp`):
- Reads data from DRAM using TensorAccessor
- Transfers pages to circular buffer
- Simple sequential reading pattern
- Runtime args: buffer address, num_pages, start_page_id

#### Writer Kernel (`writer_upsample_interleaved.cpp`):
- Performs the actual upsampling logic
- Reads from CB and writes multiple copies to DRAM
- Key upsampling algorithm:
  ```cpp
  // For each input position (x, y) in batch b
  // Write to output positions:
  //   (scale_h * x + j, scale_w * y + k) for j in [0, scale_h), k in [0, scale_w)
  ```
- Handles both row-major (block_height=1) and tiled (block_height=TILE_HEIGHT) inputs

#### Compute Kernel (Tiled layout only):
- Uses untilize kernel to convert from tiled to row-major format
- Only needed when input is in TILE layout

### 5. Data Movement Pattern

#### Nearest-Neighbor Upsampling Logic:
1. Reader loads input data sequentially from DRAM to CB
2. Writer reads each input element from CB
3. Writer duplicates each element `scale_h × scale_w` times
4. Output positions calculated as:
   ```
   start_index = batch * (H*scale_h) * (W*scale_w) +
                 (scale_h * h) * (W*scale_w) +
                 (scale_w * w)
   ```
5. Nested loops write to all upsampled positions

### 6. TensorAccessor Pattern

Both kernels use TensorAccessor for DRAM address calculation:
```cpp
constexpr auto src_args = TensorAccessorArgs<2>();  // Compile-time offset
const auto accessor = TensorAccessor(src_args, buffer_addr, page_size);
uint64_t noc_addr = accessor.get_noc_addr(page_index);
```

### 7. Memory Layout Considerations

#### DRAM Interleaved:
- Data distributed across DRAM banks
- Page-based access pattern
- Alignment requirements (typically 16 or 32 bytes)

#### Key Observations for 3D:
- Current 2D treats batch dimension as part of flattened index
- Writer kernel calculates 4D indices from linear work unit ID
- Memory access patterns are row-major optimized

## Important Patterns and Conventions

### Runtime Arguments Structure:
```cpp
// Reader
[buffer_address, num_units, start_unit_id]

// Writer
[buffer_address, num_blocks, start_block_id]
```

### Compile-Time Arguments:
- CB indices and sizes
- Scale factors
- Tensor dimensions
- TensorAccessor args (offsets for address calculation)

### Buffer Address Updates:
- Override callback updates buffer addresses when tensors move
- Essential for operation reuse with different tensors

## Key Differences Between Layouts

| Aspect | Row-Major | Tiled |
|--------|-----------|-------|
| Work unit | Single row | Row of tiles |
| CB pages | 1 per unit | tiles_per_row |
| Compute kernel | Not needed | Untilize required |
| Writer block_height | 1 | TILE_HEIGHT |
| Output buffer | Reuses input CB | Separate CB |

## Considerations for 3D Extension

### Challenges:
1. **5D tensor handling**: (N, D, H, W, C) requires depth dimension management
2. **Work distribution**: Must account for depth in work unit calculation
3. **Index calculation**: More complex mapping from linear to 5D indices
4. **Scale factors**: Need three scale factors (depth, height, width)
5. **Memory patterns**: Depth-wise upsampling adds complexity to write patterns

### Opportunities:
1. Can reuse TensorAccessor pattern with updated indexing
2. Similar CB management approach should work
3. Core work distribution pattern remains applicable
4. Reader kernel likely needs minimal changes
5. Writer kernel needs most modification for 3D index math

## Implementation Strategy

The 3D implementation should:
1. Extend work unit concept to handle depth dimension
2. Modify writer kernel for 3D upsampling logic
3. Update index calculations for 5D tensors
4. Maintain compatibility with existing infrastructure
5. Focus on row-major layout first (simpler than tiled)

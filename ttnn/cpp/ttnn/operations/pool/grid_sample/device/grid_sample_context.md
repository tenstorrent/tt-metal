# Grid Sample Operation Context Analysis

## Overview
The grid_sample operation implements a spatial transformer that samples values from an input tensor using a grid of sampling points. The current implementation supports bilinear interpolation mode with zeros padding.

## Current Program Factory Implementation Analysis

### Work Unit Definition
- **Work unit**: A single row/stick of the grid tensor (contains multiple coordinate sets)
- **Grid batching**: Each grid row can contain K sets of (x,y) coordinates or precomputed values
- **Total work**: `total_grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1]`
- Work is distributed across cores using `split_work_to_cores()` for interleaved mode or shard-based distribution for sharded mode

### Circular Buffer Management
The program factory creates several circular buffers:

1. **Grid CB (c_0)**: Stores grid coordinates
   - Size: `grid_stick_size` (aligned to DRAM/L1 requirements)
   - For sharded: holds `grid_nsticks_per_core` rows

2. **Input CBs (c_1, c_2)**: Store input image data for 4 corner points
   - Page size: `in_ntiles_c * TILE_HW * element_size`
   - Uses BUFFERING_FACTOR=2 for double buffering
   - Split reader mode uses two CBs (c_1 and c_2) for parallel reads

3. **Scalar CBs (c_3, c_4)**: Store interpolation weights
   - Page size: `tile_size(data_format)`
   - Also uses BUFFERING_FACTOR=2
   - Split reader uses two CBs (c_3 and c_4)

4. **Output CB (c_5)**: Stores output data
   - Page size: `FACE_WIDTH * element_size`
   - For sharded: holds `output_nsticks_per_core * out_ntiles_c` pages

### Data Flow Pattern
1. **Reader kernel(s)**:
   - Reads grid coordinates from DRAM/L1
   - For each grid point:
     - Computes/reads 4 corner coordinates (NW, NE, SW, SE)
     - Reads corresponding input sticks from these 4 locations
     - Calculates/reads bilinear weights
     - Pushes data to input and scalar CBs

2. **Compute kernel**:
   - Performs weighted reduction using pool2d compute kernel
   - Reads from input CBs and scalar CBs
   - Applies weights and sums the 4 corner contributions
   - Writes results to output CB

3. **Writer kernel** (interleaved mode only):
   - Reads from output CB
   - Writes to output tensor in DRAM

### Index Calculations

#### Grid Coordinate Mapping
For standard grid (non-precomputed):
```cpp
// Grid values are in normalized [-1, 1] space
// Convert to image coordinates [0, H-1] x [0, W-1]
h_coord_image = h_coord_rel * (input_height * 0.5f) + (input_height * 0.5f - 0.5f)
w_coord_image = w_coord_rel * (input_width * 0.5f) + (input_width * 0.5f - 0.5f)
```

#### Bilinear Interpolation
```cpp
h0 = floor(h_coord_image), h1 = h0 + 1
w0 = floor(w_coord_image), w1 = w0 + 1

h_frac = h_coord_image - h0
w_frac = w_coord_image - w0

weight_nw = (1 - h_frac) * (1 - w_frac)  // North-West
weight_ne = (1 - h_frac) * w_frac        // North-East
weight_sw = h_frac * (1 - w_frac)        // South-West
weight_se = h_frac * w_frac              // South-East
```

### Memory Access Patterns
- **Grid tensor**: Sequential reads, one stick per spatial output position
- **Input tensor**: Random access pattern - 4 non-contiguous reads per grid point
- **Output tensor**: Sequential writes matching grid order

### Layout-specific Handling
1. **Interleaved mode**:
   - Data distributed across DRAM banks
   - Uses explicit reader and writer kernels
   - Runtime args track offsets for each core

2. **Height-sharded mode**:
   - Grid and output tensors sharded across L1 of cores
   - No writer kernel needed (data stays in L1)
   - Shard spec defines distribution pattern

### Core Distribution
- **Interleaved**: Uses `split_work_to_cores()` to divide grid rows evenly
- **Sharded**: Follows grid tensor's shard specification
- Supports core_group_1 and core_group_2 with different work amounts

### Compile-time vs Runtime Arguments

**Compile-time** (fixed at kernel compilation):
- CB indices
- Tensor dimensions (input_height, input_width)
- Stick sizes (aligned)
- Data formats
- Grid configuration (grid_batching_factor, use_precomputed_grid)

**Runtime** (can change between executions):
- Buffer addresses
- Number of sticks to process
- Starting offsets

## Key Differences Between Bilinear and Nearest Neighbor

### Bilinear Interpolation (Current)
- Reads 4 input values per grid point
- Computes 4 weights based on fractional distances
- Performs weighted sum: `output = w_nw*v_nw + w_ne*v_ne + w_sw*v_sw + w_se*v_se`
- Smooth interpolation between pixels

### Nearest Neighbor (To Implement)
- Reads only 1 input value per grid point
- No weight calculation needed (implicit weight = 1.0)
- Simple selection: `output = input[round(y)][round(x)]`
- Picks the closest pixel value
- Formula:
  ```cpp
  h_nearest = round(h_coord_image)  // or floor(h_coord_image + 0.5)
  w_nearest = round(w_coord_image)  // or floor(w_coord_image + 0.5)
  ```

## Implementation Considerations for Nearest Mode

1. **Simplified data flow**: Only need to read 1 input stick instead of 4
2. **No weights needed**: Can eliminate scalar CBs or use constant weight=1.0
3. **Simpler compute**: No reduction needed, direct copy with possible channel extension
4. **Better memory efficiency**: 4x fewer input reads
5. **Different CB requirements**: Smaller input CB (1 stick vs 4)
6. **Potential for optimization**: Could bypass compute kernel for simple copy cases

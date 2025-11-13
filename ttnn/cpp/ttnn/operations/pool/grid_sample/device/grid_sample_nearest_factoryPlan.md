# Grid Sample Nearest Neighbor Program Factory Implementation Plan

## Overview
This document outlines the plan for implementing a new program factory for nearest neighbor sampling mode in the grid_sample operation. The nearest neighbor mode will provide faster sampling by reading only one input value per grid point instead of four.

## Key Differences from Bilinear Mode

### 1. Data Requirements
- **Bilinear**: Reads 4 corner points, computes 4 weights, performs weighted sum
- **Nearest**: Reads 1 point, no weights needed, direct copy

### 2. Coordinate Calculation
- **Bilinear**: `h0 = floor(h), h1 = h0 + 1, w0 = floor(w), w1 = w0 + 1`
- **Nearest**: `h = round(h_coord), w = round(w_coord)`

### 3. Memory Access
- **Bilinear**: 4 non-contiguous reads per grid point
- **Nearest**: 1 read per grid point (4x reduction in memory bandwidth)

## Program Factory Design

### Circular Buffer Configuration

#### For Interleaved Mode:
1. **Grid CB (c_0)**: Same as bilinear - stores grid coordinates
2. **Input CB (c_1)**: Simplified - only needs to store 1 stick per grid point
   - Page size: `in_ntiles_c * TILE_HW * element_size`
   - Buffer pages: `BUFFERING_FACTOR` (2 for double buffering)
3. **Input CB (c_2)**: Optional second input CB for split reader mode
4. **Output CB (c_3 or c_4)**: Same as bilinear
   - Page size: `FACE_WIDTH * element_size`
   - Buffer pages: `out_ntiles_c * BUFFERING_FACTOR`

**Note**: No scalar CBs needed since we don't have weights in nearest mode

#### For Sharded Mode:
Similar to interleaved but with sharded memory specifications

### Kernel Strategy

Since nearest neighbor sampling is purely a data movement operation (no computation needed), we only need reader and writer kernels:

- **Reader**: Create `reader_grid_sample_nearest_interleaved.cpp` and `reader_grid_sample_nearest_sharded.cpp`
  - Based on existing readers but simplified to:
    1. Read grid coordinates
    2. Calculate nearest neighbor: `h = round(h_coord), w = round(w_coord)`
    3. Read single input stick at `[h, w]`
    4. Write directly to output CB (for sharded) or intermediate CB (for interleaved)
  - ToDo: Implement these new reader kernels

- **Compute**: **NOT NEEDED** - Nearest neighbor is pure data movement, no computation required

- **Writer**: Reuse existing `writer_grid_sample_interleaved.cpp` (no changes needed)
  - Only used in interleaved mode
  - Sharded mode keeps data in L1, no writer needed

### Implementation Steps

1. **Create new factory function**: `grid_sample_nearest_program_factory()`
2. **Adapt CB creation**:
   - Remove scalar CBs (no weights needed)
   - Remove compute-related CBs
   - Keep grid CB, input CB, and output CB only
3. **Configure kernel compile-time args**:
   - Remove all compute-related arguments
   - Simplify reader args for single stick reads
4. **Set runtime arguments**:
   - Only for reader and writer kernels
   - No compute kernel runtime args needed

### Proposed Factory Structure

```cpp
tt::tt_metal::operation::ProgramWithCallbacks grid_sample_nearest_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels)
```

Key differences from bilinear factory:
- No `mode` parameter (always nearest)
- Simplified CB allocation (no scalar CBs)
- Different reader kernel paths
- Simplified compile-time arguments

### Work Distribution
- Same as bilinear mode using `split_work_to_cores()` or shard specs
- Each core processes `grid_nsticks_per_core` grid rows
- Output distribution follows grid distribution

### Memory Layout Compatibility
- Input: ROW_MAJOR, DRAM interleaved or height sharded
- Grid: ROW_MAJOR, DRAM interleaved or height sharded
- Output: ROW_MAJOR, matches grid memory layout

### Performance Optimizations
1. **Memory bandwidth**: 4x reduction in input reads
2. **Compute simplification**: No multiplication/addition for weights
3. **CB size reduction**: Smaller input buffers needed
4. **Potential kernel fusion**: Reader could directly write to output CB

## Kernel Requirements

### Reader Kernel (ToDo: Implement)
**File**: `reader_grid_sample_nearest_interleaved.cpp` / `reader_grid_sample_nearest_sharded.cpp`
- Read grid coordinates
- Calculate nearest neighbor: `h = round(h_coord), w = round(w_coord)`
- Read single input stick at `[h, w]`
- For interleaved: Push to intermediate CB for writer to consume
- For sharded: Write directly to output CB (no compute/writer needed)

### Compute Kernel
**NOT REQUIRED** - Nearest neighbor sampling is a pure data movement operation with no computation.

### Writer Kernel
- Reuse existing `writer_grid_sample_interleaved.cpp`
- No modifications needed
- Only used in interleaved mode

## Testing Strategy
1. Verify coordinate rounding matches PyTorch behavior
2. Test boundary cases (coordinates at image edges)
3. Compare output with PyTorch grid_sample(mode='nearest')
4. Performance benchmarking vs bilinear mode

## Next Steps
1. Implement `grid_sample_nearest_program_factory.cpp` following this plan
2. Create/modify required kernels (marked as ToDo)
3. Update `grid_sample_op.cpp` to dispatch to nearest factory when mode="nearest"
4. Add Python bindings for nearest mode
5. Comprehensive testing and validation

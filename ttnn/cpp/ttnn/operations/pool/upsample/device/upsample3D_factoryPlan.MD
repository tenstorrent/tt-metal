# Upsample 3D Program Factory Implementation Plan

## Overview
This document outlines the implementation plan for a 3D nearest-neighbor upsample operation supporting 5D tensors (N, D, H, W, C) with row-major DRAM interleaved layout.

## Key Design Decisions

### 1. Tensor Layout and Dimensions
- **Input**: 5D tensor with shape (N, D, H, W, C)
  - N: Batch size
  - D: Depth
  - H: Height
  - W: Width
  - C: Channels
- **Output**: 5D tensor with shape (N, D*scale_d, H*scale_h, W*scale_w, C)
- **Layout**: ROW_MAJOR only (initially, tiled support can be added later)
- **Memory**: DRAM interleaved
- **Flattened view**: N×D×H×W rows, each with C elements (channel-last)

### 2. Work Distribution Strategy

#### Work Unit Definition
- **Work unit**: One stick (row) = C elements
- **Total work units**: N × D × H × W (total number of sticks in the flattened tensor)
- **Data per unit**: C × element_size bytes
- This matches the 2D pattern where each work unit is a single row of the innermost dimension

#### Work Distribution:
```cpp
work_units_to_split = N * D * H * W;  // Total number of input sticks
// Each core gets a contiguous chunk of these sticks
```

### 3. Circular Buffer Design

#### Buffer Requirements:
- **Input CB (c_0)**:
  - Size: One stick of input data (C × element_size)
  - Aligned to DRAM alignment requirements
  - Double-buffered if core processes multiple sticks

- **Output CB**:
  - For row-major: Reuse input CB (same as 2D pattern)
  - Data is read from CB and written directly to multiple DRAM locations

#### Size Calculations:
```cpp
input_unit_size = input_shape[4] * element_size;  // C * element_size (one stick)
aligned_input_unit_size = round_up(input_unit_size, dram_alignment);
work_units_to_split = input.physical_volume() / input_shape[4];  // N*D*H*W
```

### 4. Kernel Architecture

#### Reader Kernel
- Can reuse existing 2D reader (`reader_upsample_unary_stick_layout_interleaved_start_id.cpp`)
- Reads input sticks sequentially from DRAM to CB
- Runtime args: [buffer_address, num_sticks, start_stick_id]
- No changes needed - it already handles stick-based reading

#### Writer Kernel (Needs new implementation)
- **Filename**: `writer_upsample3d_interleaved.cpp`
- **Key responsibilities**:
  1. Read input stick from CB
  2. Calculate 5D position from linear stick index
  3. Write to scale_d × scale_h × scale_w output locations
  4. Handle 5D to linear index mapping for output

- **Index calculation logic**:
```cpp
// From linear stick index to 5D coordinates
// stick_index ranges from 0 to N*D*H*W-1
uint32_t curr_index = stick_index % (D * H * W);
uint32_t curr_batch = stick_index / (D * H * W);

uint32_t curr_w = curr_index % W;
uint32_t remainder = curr_index / W;
uint32_t curr_h = remainder % H;
uint32_t curr_d = remainder / H;

// Calculate output base position
// Output shape: (N, D*scale_d, H*scale_h, W*scale_w, C)
uint32_t out_d = curr_d * scale_d;
uint32_t out_h = curr_h * scale_h;
uint32_t out_w = curr_w * scale_w;

// Calculate linear index for each output position
for (uint32_t d = 0; d < scale_d; d++) {
    for (uint32_t h = 0; h < scale_h; h++) {
        for (uint32_t w = 0; w < scale_w; w++) {
            uint32_t out_stick_idx = curr_batch * (D*scale_d) * (H*scale_h) * (W*scale_w) +
                                    (out_d + d) * (H*scale_h) * (W*scale_w) +
                                    (out_h + h) * (W*scale_w) +
                                    (out_w + w);
            // Write stick to output at out_stick_idx
        }
    }
}
```

#### Compute Kernel
- Not needed for row-major layout
- Would only be required for tiled input support

### 5. Program Factory Structure

#### Function Signature:
```cpp
tt::tt_metal::operation::ProgramWithCallbacks upsample3d_multi_core_interleaved(
    const Tensor& input,
    Tensor& output,
    const uint32_t scale_factor_d,
    const uint32_t scale_factor_h,
    const uint32_t scale_factor_w);
```

#### Main Components:

1. **Validation and Setup**
   ```cpp
   // Verify input is 5D tensor
   TT_FATAL(input.get_shape().rank() == 5, "Input must be 5D tensor");
   TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only row-major layout supported");

   // Calculate dimensions
   const auto& input_shape = input.padded_shape();
   const uint32_t N = input_shape[0];
   const uint32_t D = input_shape[1];
   const uint32_t H = input_shape[2];
   const uint32_t W = input_shape[3];
   const uint32_t C = input_shape[4];

   // Work units
   const uint32_t input_unit_size = C * input.element_size();
   const uint32_t aligned_input_unit_size = round_up(input_unit_size, dram_alignment);
   const uint32_t work_units_to_split = N * D * H * W;
   ```

2. **Core Distribution**
   ```cpp
   const auto [num_cores, all_cores, core_group_1, core_group_2,
               work_per_core_group_1, work_per_core_group_2] =
       split_work_to_cores(compute_grid_size, work_units_to_split);
   ```

3. **Circular Buffer Creation**
   ```cpp
   uint32_t num_pages_in_input_cb = 1;  // One stick at a time
   if (work_per_core_group_1 > 1) {
       num_pages_in_input_cb *= 2;  // Double buffer
   }

   const auto [src0_cb_index, cb_src0] = create_cb(
       next_cb_index++, program, all_cores,
       aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);
   ```

4. **Kernel Creation**
   - Reader: Use existing reader kernel
   - Writer: Create new 3D writer kernel with proper compile-time args

5. **Runtime Arguments**
   ```cpp
   // Per core runtime args
   reader_rt_args[1] = sticks_per_core;  // Number of sticks
   reader_rt_args[2] = sticks_processed; // Start stick ID

   writer_rt_args[1] = sticks_per_core;
   writer_rt_args[2] = sticks_processed;
   ```

6. **Override Callback**
   - Update buffer addresses for tensor movement

### 6. Compile-Time Arguments Structure

#### Reader (reuse existing):
```cpp
reader_compile_time_args = {
    cb_index,           // c_0
    aligned_unit_size,  // Aligned stick size
    TensorAccessorArgs  // For DRAM addressing
}
```

#### Writer (new):
```cpp
writer_compile_time_args = {
    cb_index,           // c_0 (input/output)
    unit_size,          // Stick size in bytes (C * element_size)
    scale_d,            // Depth scale factor
    scale_h,            // Height scale factor
    scale_w,            // Width scale factor
    output_d,           // Output depth (D * scale_d)
    output_h,           // Output height (H * scale_h)
    output_w,           // Output width (W * scale_w)
    input_d,            // Input depth D
    input_h,            // Input height H
    input_w,            // Input width W
    TensorAccessorArgs  // For DRAM addressing
}
```

### 7. Key Differences from 2D Implementation

| Aspect | 2D Implementation | 3D Implementation |
|--------|------------------|-------------------|
| Input shape | (N, H, W, C) | (N, D, H, W, C) |
| Work units | N * H * W sticks | N * D * H * W sticks |
| Scale factors | 2 (H, W) | 3 (D, H, W) |
| Index calculation | 4D to linear mapping | 5D to linear mapping |
| Writer loops | 2 nested (h, w) | 3 nested (d, h, w) |
| Output positions per input | scale_h * scale_w | scale_d * scale_h * scale_w |

### 8. Memory Access Pattern

For each input stick at position (n, d, h, w):
- Input linear index: `n*(D*H*W) + d*(H*W) + h*W + w`
- Output positions: All sticks in the cube from:
  - `(n, d*scale_d, h*scale_h, w*scale_w)` to
  - `(n, d*scale_d+scale_d-1, h*scale_h+scale_h-1, w*scale_w+scale_w-1)`
- Total writes per input stick: `scale_d * scale_h * scale_w`

### 9. Implementation Notes

#### Work Distribution Example:
```cpp
// For input (2, 3, 4, 5, 6) = 2 batches, 3 depth, 4 height, 5 width, 6 channels
// Total sticks = 2 * 3 * 4 * 5 = 120
// If we have 10 cores, each core processes 12 sticks
// Core 0: sticks 0-11, Core 1: sticks 12-23, etc.
```

#### Reader Kernel Behavior:
- Reads sticks sequentially from DRAM
- Each stick is C contiguous elements
- Uses TensorAccessor with stick index for addressing

#### Writer Kernel Behavior:
- For each stick, determines its 5D position
- Calculates all output positions based on scale factors
- Writes the same stick data to multiple output locations

### 10. Testing Considerations

- Start with small tensors for validation (e.g., (1, 2, 2, 2, 4))
- Test with scale factors of 1 (identity operation)
- Test with different scale factors per dimension
- Verify correct index calculations with known patterns
- Check edge cases (single batch, single depth, etc.)
- Validate memory access patterns are contiguous where possible

### 11. Future Extensions

1. **Tiled layout support**: Add untilize kernel and separate output CB
2. **Sharded memory**: Support HEIGHT_SHARDED and BLOCK_SHARDED layouts
3. **Optimizations**:
   - Batch writes when output positions are contiguous
   - Use async NOC operations more efficiently
   - Consider vectorized writes for aligned cases
4. **Other interpolation modes**: Trilinear interpolation

## Implementation Steps

1. ✅ Analyze existing 2D implementation
2. ✅ Design 3D architecture and data flow
3. 🔄 Implement header file with function declarations
4. 🔄 Implement program factory main logic
5. 🔄 Create new 3D writer kernel
6. Integrate with build system
7. Write unit tests
8. Performance optimization

## Risk Mitigation

- **Memory bandwidth**: 3D upsampling has higher write amplification (scale_d * scale_h * scale_w)
  - Mitigation: Ensure coalesced writes where possible, optimize NOC usage

- **Index calculation complexity**: 5D indexing is error-prone
  - Mitigation: Clear documentation, thorough testing, debug assertions

- **Large tensor handling**: 5D tensors can consume significant memory
  - Mitigation: Proper work distribution, consider chunking for very large tensors

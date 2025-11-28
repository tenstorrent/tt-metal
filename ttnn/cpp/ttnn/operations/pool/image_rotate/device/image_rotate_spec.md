# Image Rotate Functional Specification

## Overview
- **Operation Name**: image_rotate
- **Category**: pool (spatial transformation)
- **Reference Operation**: grid_sample_bilinear
- **Reference Analysis**: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_analysis.md`

## Mathematical Definition

### Formula
For each output pixel at position (x_out, y_out), the corresponding input position (x_in, y_in) is computed via inverse rotation:

```
// Translate to center-relative coordinates
x_centered = x_out - center_x
y_centered = y_out - center_y

// Apply inverse rotation (negative angle to find source for each destination)
angle_rad = -angle * PI / 180.0
cos_a = cos(angle_rad)
sin_a = sin(angle_rad)

x_in = x_centered * cos_a - y_centered * sin_a + center_x
y_in = x_centered * sin_a + y_centered * cos_a + center_y

// Sample input at (x_in, y_in) using bilinear interpolation
output[n, y_out, x_out, c] = bilinear_sample(input[n, :, :, c], x_in, y_in, fill_value)
```

Where bilinear_sample computes:
```
h0 = floor(y_in)
h1 = h0 + 1
w0 = floor(x_in)
w1 = w0 + 1

h_frac = y_in - h0
w_frac = x_in - w0

weight_nw = (1 - h_frac) * (1 - w_frac)  // top-left
weight_ne = (1 - h_frac) * w_frac        // top-right
weight_sw = h_frac * (1 - w_frac)        // bottom-left
weight_se = h_frac * w_frac              // bottom-right

// For out-of-bounds corners, weight = 0 and value = fill_value
output = weight_nw * input[h0,w0] + weight_ne * input[h0,w1] +
         weight_sw * input[h1,w0] + weight_se * input[h1,w1]
```

### Semantic Description
The image_rotate operation rotates an image tensor by an arbitrary angle around a specified center point using bilinear interpolation. Positive angles rotate counter-clockwise. Areas outside the rotated image are filled with a configurable fill value (default 0.0). The output tensor has the same dimensions as the input (no expansion mode).

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input image tensor, shape (N, H, W, C) in NHWC format |
| angle | float | Yes | (-inf, +inf) | - | Rotation angle in degrees. Positive = counter-clockwise |
| center | optional<tuple<float,float>> | No | x in [0, W-1], y in [0, H-1] | ((W-1)/2, (H-1)/2) | Rotation center point (cx, cy) in pixel coordinates |
| fill | float | No | (-inf, +inf) | 0.0 | Fill value for areas outside the rotated image |
| expand | bool | No | Must be false | false | If true, return error. Only false is supported (same output dimensions) |
| output_mem_config | MemoryConfig | No | DRAM_INTERLEAVED | DRAM_INTERLEAVED | Output memory configuration |

### Input Tensor Requirements
| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be 4D (N, H, W, C) | "Input tensor must be 4D (N, H, W, C)" |
| Layout | ROW_MAJOR_LAYOUT | "Input tensor must be in ROW_MAJOR layout" |
| Dtype | bfloat16 or float32 | "Input tensor dtype must be bfloat16 or float32" |
| Storage | Must be on device (DRAM) | "Input tensor must be on device" |
| Memory | INTERLEAVED only | "Input tensor must be DRAM interleaved" |
| Channel alignment | C must be aligned to 32 bytes | "Channel dimension must be aligned to 32 bytes" |

### Output Tensor Specification
| Property | Value/Calculation |
|----------|-------------------|
| Shape | Same as input: (N, H, W, C) |
| Dtype | Same as input tensor |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | INTERLEAVED (from output_mem_config) |

## Comparison with Reference Operation

### What Can Be Reused

1. **Compute kernel**: `compute_pool_2d.cpp` - identical usage
   - Same reduction operation (SUM across 4 corners)
   - Same tilize/untilize flow
   - Same CB indexing

2. **Writer kernel**: `writer_grid_sample_interleaved.cpp` - identical usage
   - Same sequential DRAM writes
   - Same TensorAccessor pattern
   - Same output stick handling

3. **Common helper functions**:
   - `is_coordinate_valid()` - boundary checking
   - `fill_four_val()` - weight packing
   - `float_to_bfloat16()` - conversion
   - `read_four_corner_inputs()` - 4-corner NOC reads
   - `zero_out_tiles()` - CB initialization for boundary handling

4. **CB structure** (minus grid CB):
   - CB_1 (input_cb): 4 input sticks, double-buffered
   - CB_3 (scalar_cb): interpolation weights
   - CB_5 (output_cb): output sticks, double-buffered

5. **Work distribution**:
   - Same `split_work_to_cores()` pattern
   - Same core group 1 / core group 2 handling

### Key Differences
| Aspect | grid_sample | image_rotate | Implementation Impact |
|--------|-------------|--------------|----------------------|
| Grid tensor | Required input tensor | None - computed on device | No grid CB needed, no grid tensor accessor |
| Coordinate source | Read from grid tensor | Computed from angle/center | Reader computes rotation math instead of reading |
| Compile-time args | Grid tensor properties | cos_a, sin_a, center_x, center_y, fill_value | Different CT arg layout |
| Runtime args | grid_buffer_address | None needed for grid | Simpler runtime args |
| Precomputed mode | Supported | N/A | Remove precomputed grid handling |
| Batching factor | Multiple coords per grid stick | Always 1 | Simplify loop structure |
| Sharded mode | Supported | Not supported | Remove sharded code paths |
| Split reader | Supported for sharded | Not applicable | Remove split reader logic |
| Fill value | Implicit zero | Configurable | Pass fill value to reader |

## Design Decisions

### Decision 1: On-Device Grid Computation
- **Choice**: Compute rotation sampling coordinates directly in the reader kernel rather than creating a grid tensor on host
- **Rationale**:
  - Avoids allocating and transferring a grid tensor (N * H * W * 2 * sizeof(float) bytes)
  - Rotation math is simple (2 multiplies, 2 adds per coordinate after precomputing sin/cos)
  - The rotation parameters (cos_a, sin_a, center_x, center_y) are constant for all pixels
- **Alternatives Considered**:
  - Host-side grid generation + grid_sample: Higher memory bandwidth, adds host-device transfer
  - Precomputed grid tensor: Same issues, plus extra host computation
- **Tradeoffs**:
  - Pro: Lower memory usage, no extra tensor allocation
  - Con: Reader kernel does more computation (but still dominated by NOC reads)

### Decision 2: Pass Trigonometric Values as Compile-Time Args
- **Choice**: Pass cos(angle) and sin(angle) as compile-time float arguments rather than computing in kernel
- **Rationale**:
  - Angle is constant for the entire operation
  - Avoids trigonometric function calls in kernel (potentially slow on RISC-V)
  - Values can be computed once on host during program creation
- **Alternatives Considered**:
  - Pass angle and compute sin/cos in kernel: Requires math library, repeated computation
  - Runtime args: Would work but compile-time is more appropriate for constants
- **Tradeoffs**:
  - Pro: Zero runtime trig computation overhead
  - Con: Compile-time arg slot usage (but we have plenty)

### Decision 3: DRAM Interleaved Only (No Sharded Mode)
- **Choice**: Support only DRAM-interleaved input and output tensors for initial implementation
- **Rationale**:
  - Simplifies implementation significantly (no split reader, no sharded CB handling)
  - Image rotation is typically used on full images, not streaming shards
  - Can add sharded mode later if needed
- **Alternatives Considered**:
  - Full sharded support from start: Much higher complexity
  - L1-sharded input, DRAM output: Partial benefit, still complex
- **Tradeoffs**:
  - Pro: Simpler kernel, faster implementation, easier debugging
  - Con: May need to add sharded mode later for certain use cases

### Decision 4: Configurable Fill Value
- **Choice**: Accept fill value as a parameter, default to 0.0
- **Rationale**:
  - PyTorch's rotate supports fill parameter
  - Common to use mean pixel value or specific background color
  - Zero is most common default
- **Alternatives Considered**:
  - Always use zero (like grid_sample): Less flexible
  - Tensor fill value: Over-complicated for scalar fill
- **Tradeoffs**:
  - Pro: More flexible API, matches PyTorch
  - Con: Slight added complexity in boundary handling

### Decision 5: Reject expand=True at Validation
- **Choice**: Return an error if expand=True is passed
- **Rationale**:
  - Expand mode changes output dimensions, significantly more complex
  - Output size depends on rotation angle, requires different memory allocation
  - Initial implementation focuses on same-size rotation
- **Alternatives Considered**:
  - Silently ignore expand parameter: Confusing behavior
  - Implement expand mode: Significant scope increase
- **Tradeoffs**:
  - Pro: Clear error message, simpler implementation
  - Con: Not full PyTorch parity (can add later)

### Decision 6: New Reader Kernel (Not Refactor Common)
- **Choice**: Create a new reader kernel `reader_image_rotate_interleaved.cpp` rather than extending grid_sample_reader_common.hpp
- **Rationale**:
  - Rotation coordinate computation is fundamentally different from grid reading
  - No grid tensor accessor needed
  - Simpler, cleaner code without conditional compilation for both modes
  - Can still reuse helper functions via includes
- **Alternatives Considered**:
  - Extend grid_sample_reader_common.hpp with rotation mode: Would add complexity to existing code
  - Template-based approach: Over-engineered for this difference
- **Tradeoffs**:
  - Pro: Clean separation, easier to maintain
  - Con: Some code duplication (mitigated by shared helpers)

## Work Distribution

### Work Unit Definition
One work unit = one output pixel position (x_out, y_out)

For each work unit:
1. Compute source coordinates (x_in, y_in) using rotation transform
2. Calculate 4 corner pixel indices and bilinear weights
3. Read 4 input sticks from DRAM (one per corner)
4. Compute weighted sum via reduction
5. Write 1 output stick to DRAM

### Parallelization Strategy
- **Grid**: Dynamic based on total work units and available cores
- **Total work units**: N * H * W (batch_size * height * width)
- **Work per core**: Distributed using `split_work_to_cores(compute_grid_size, total_sticks)`
- **Load balancing**: Two core groups handle remainder distribution
  - Core group 1: `num_sticks_per_core_group_1` sticks each
  - Core group 2: `num_sticks_per_core_group_2` sticks each (may be 0 or 1 less)

### Output Stick Ordering
Output sticks are processed in row-major order within each batch:
```
for batch in 0..N:
    for y in 0..H:
        for x in 0..W:
            process_output_pixel(batch, y, x)
```

Each core processes a contiguous range of output stick indices.

## Data Flow

### High-Level Flow
```
Rotation Parameters (cos_a, sin_a, center_x, center_y, fill_value)
                              |
                              v
Input Tensor -----> RISCV_0 (reader/NOC0) -----> CB_input (4 sticks)
                              |                         |
                    [Compute rotation coords]           |
                    [Read 4 corners from DRAM]          |
                              |                         |
                              v                         v
                    CB_scalar (4 weights) -----> Compute Kernel
                                                        |
                                              [Tilize + weight multiply]
                                              [Column reduction (SUM)]
                                              [Untilize output]
                                                        |
                                                        v
                                                   CB_output
                                                        |
                                                        v
                                          RISCV_1 (writer/NOC1)
                                                        |
                                                        v
                                                 Output Tensor (DRAM)
```

### Kernel Data Movement
| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| reader_image_rotate_interleaved | RISCV_0 (BRISC) | NOC0 | Computes rotation coordinates, reads 4 input corners, writes weights to scalar CB |
| writer_grid_sample_interleaved | RISCV_1 (NCRISC) | NOC1 | Reads output sticks from CB, writes to DRAM |

Note: The writer kernel from grid_sample can be reused unchanged since it simply writes output sticks to DRAM.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Sizing Strategy |
|-------|------|---------|----------|----------|-----------------|
| CB_1 | cb_input | 4 corner input sticks | RISCV_0 (reader) | Compute | `4 * input_stick_nbytes * 2` (double-buffered, 4 sticks per work unit) |
| CB_3 | cb_scalar | Bilinear weights | RISCV_0 (reader) | Compute | `TILE_HW * element_size * 2` (double-buffered, 4 weights packed) |
| CB_5 | cb_output | Output sticks | Compute | RISCV_1 (writer) | `FACE_WIDTH * element_size * out_ntiles_c * 2` (double-buffered) |

**Removed from grid_sample**:
- CB_0 (grid_cb): Not needed - no grid tensor
- CB_2 (input_cb_1): Not needed - no split reader
- CB_4 (scalar_cb_1): Not needed - no split reader

### CB Sizing Formulas

**Input CB (CB_1)**:
```cpp
const uint32_t input_stick_nbytes = C * element_size;  // Aligned to 32 bytes
const uint32_t input_cb_page_size = 4 * input_stick_nbytes;  // 4 corners per work unit
const uint32_t input_cb_size = input_cb_page_size * 2;  // Double-buffered
```

**Scalar CB (CB_3)**:
```cpp
const uint32_t scalar_cb_page_size = TILE_HW * element_size;  // 1024 * 2 = 2048 bytes for bfloat16
const uint32_t scalar_cb_size = scalar_cb_page_size * 2;  // Double-buffered
```

**Output CB (CB_5)**:
```cpp
const uint32_t output_stick_nbytes = C * element_size;  // Same as input
const uint32_t out_ntiles_c = (C + FACE_WIDTH - 1) / FACE_WIDTH;  // Tiles per stick
const uint32_t output_cb_page_size = FACE_WIDTH * element_size * out_ntiles_c;
const uint32_t output_cb_size = output_cb_page_size * 2;  // Double-buffered
```

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Rotation Coordinate Computation** (per output pixel):
```cpp
// Output pixel position (computed from work unit index)
const uint32_t global_stick_idx = start_stick_idx + local_idx;
const uint32_t batch_idx = global_stick_idx / (H * W);
const uint32_t spatial_idx = global_stick_idx % (H * W);
const uint32_t y_out = spatial_idx / W;
const uint32_t x_out = spatial_idx % W;

// Rotation math (cos_a, sin_a, center_x, center_y are compile-time constants)
const float x_centered = static_cast<float>(x_out) - center_x;
const float y_centered = static_cast<float>(y_out) - center_y;
const float x_in = x_centered * cos_a - y_centered * sin_a + center_x;
const float y_in = x_centered * sin_a + y_centered * cos_a + center_y;
```

**Input Tensor Read Pattern** (per output pixel):
- Random access pattern based on computed source coordinates
- 4 potentially non-contiguous sticks read per output pixel
- Uses TensorAccessor for DRAM bank mapping
- Read order: NW corner, NE corner, SW corner, SE corner

```
Read sequence per work unit:
1. Compute (x_in, y_in) from (x_out, y_out)
2. h0 = floor(y_in), h1 = h0 + 1, w0 = floor(x_in), w1 = w0 + 1
3. Read stick[batch * H * W + h0 * W + w0] if valid, else skip
4. Read stick[batch * H * W + h0 * W + w1] if valid, else skip
5. Read stick[batch * H * W + h1 * W + w0] if valid, else skip
6. Read stick[batch * H * W + h1 * W + w1] if valid, else skip
```

### RISCV_1 ("writer" / NCRISC) Access

**Output Tensor Write Pattern**:
- Sequential writes of output sticks to DRAM
- Each stick written after compute produces one output
- Uses TensorAccessor for DRAM bank mapping

```
Write sequence:
for stick_id in range(start_stick_id, end_stick_id):
    cb_wait_front(cb_output, ntiles_c)
    dst_noc_addr = output_accessor.get_noc_addr(stick_id)
    noc_async_write(cb_read_ptr, dst_noc_addr, output_stick_size)
    noc_async_write_barrier()
    cb_pop_front(cb_output, ntiles_c)
```

### Compute Access

**CB Read Pattern**:
1. Wait for scalar CB (1 page with 4 weights)
2. Wait for input CB (1 page with 4 sticks)
3. Process all channel blocks for this output pixel
4. Pop scalar and input CBs

**CB Write Pattern**:
1. Pack result to output CB
2. Push output CB page

### Boundary Handling

**Zero Padding Approach** (same as grid_sample):
1. Input CB is pre-zeroed at kernel start using `zero_out_tiles()`
2. Out-of-bounds corner reads are skipped (no NOC read issued)
3. Corresponding weights are set to 0.0
4. Reduction naturally produces correct result with zero-weighted corners

**Fill Value Handling**:
- When all 4 corners are out of bounds, output is `fill_value`
- When some corners are valid, fill_value is implicit via zero-weight + zero-data
- For non-zero fill values: Pre-fill input CB with `fill_value` instead of zero

## Compile-Time Arguments

### Reader Kernel: `reader_image_rotate_interleaved.cpp`

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `input_cb_index` | uint32_t | CB index for input data (CB_1) |
| 1 | `scalar_cb_index` | uint32_t | CB index for interpolation weights (CB_3) |
| 2 | `input_stick_nbytes` | uint32_t | Size of one input stick in bytes |
| 3 | `input_batch` | uint32_t | Batch dimension N of input tensor |
| 4 | `input_height` | uint32_t | Height dimension H of input tensor |
| 5 | `input_width` | uint32_t | Width dimension W of input tensor |
| 6 | `cos_angle` | float | cos(-angle * PI / 180) precomputed |
| 7 | `sin_angle` | float | sin(-angle * PI / 180) precomputed |
| 8 | `center_x` | float | Rotation center x coordinate |
| 9 | `center_y` | float | Rotation center y coordinate |
| 10 | `fill_value_bf16` | uint16_t | Fill value as bfloat16 for boundary |
| 11+ | TensorAccessorArgs | - | Input tensor accessor parameters |

### Compute Kernel: `compute_pool_2d.cpp` (reused from grid_sample)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `in_ntiles_c` | uint32_t | Number of tiles in channel dimension |
| 1 | `REDUCTION_SIZE` | uint32_t | Window size for reduction = 4 (bilinear) |
| 2 | `enable_split_reader` | uint32_t | 0 (no split reader for image_rotate) |
| 3 | `total_interpolations` | uint32_t | Total output pixels to process on this core |
| 4 | `channels_per_shard` | uint32_t | Channel dimension size C |
| 5 | `in_nblocks_c` | uint32_t | Number of channel blocks |
| 6 | `MAX_ROWS_FOR_REDUCTION` | uint32_t | 16 (face height) |
| 7 | `in_cb_id_0` | uint32_t | Input CB index (CB_1) |
| 8 | `in_scalar_cb_id_0` | uint32_t | Scalar CB index (CB_3) |
| 9 | `in_cb_id_1` | uint32_t | Unused (32) - no split reader |
| 10 | `in_scalar_cb_id_1` | uint32_t | Unused (32) - no split reader |
| 11-16 | Unused | - | Reserved for other pool operations |
| 17 | `output_cb_index` | uint32_t | Output CB index (CB_5) |
| 18 | Unused | - | Reserved |
| 19 | `ONE_SCALAR_PER_CORE` | uint32_t | 0 (false) - scalars per work unit |
| 20 | `pre_tilize_cb_id` | uint32_t | Unused (32) for row-major output |
| 21 | `is_output_tiled` | uint32_t | 0 (false) for row-major output |
| 22 | `is_output_block_format` | uint32_t | 0 (false) |
| 23-30 | Unused | - | Reserved for pooling operations |

**Defines**:
- `REDUCE_OP = PoolType::SUM`
- `REDUCE_DIM = ReduceDim::REDUCE_COL`

### Writer Kernel: `writer_grid_sample_interleaved.cpp` (reused)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `output_cb_index` | uint32_t | Output CB index (CB_5) |
| 1 | `output_stick_size` | uint32_t | Size of one output stick in bytes |
| 2 | `out_ntiles_c` | uint32_t | Number of tiles per output stick |
| 3+ | TensorAccessorArgs | - | Output tensor accessor parameters |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `input_buffer_address` | uint32_t | Base address of input tensor in DRAM |
| 1 | `num_sticks` | uint32_t | Number of output sticks to process on this core |
| 2 | `start_stick_id` | uint32_t | Starting global output stick index |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `output_buffer_address` | uint32_t | Base address of output tensor in DRAM |
| 1 | `output_sticks` | uint32_t | Number of output sticks to write |
| 2 | `output_processed` | uint32_t | Starting output stick index |

## Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| angle = 0 | Output equals input (identity rotation) |
| angle = 90 | Counter-clockwise 90 degree rotation |
| angle = -90 | Clockwise 90 degree rotation |
| angle = 180 | 180 degree rotation |
| angle = 360 or -360 | Same as angle = 0 |
| angle > 360 or < -360 | Should work (angles wrap naturally via sin/cos) |
| Single pixel image (H=W=1) | Returns single pixel with same value |
| Large image (>1000 pixels per dim) | Should distribute across cores properly |
| Non-square image | Should work correctly |
| center = (0, 0) | Rotates around top-left corner |
| center outside image | Valid, rotates around external point |
| fill = non-zero | Out-of-bounds areas filled with specified value |
| N > 1 (batch) | Each image in batch rotated independently |
| expand = true | Return error at validation |

## Validation Requirements

### Input Validation (in `validate()` method)

```cpp
void ImageRotate::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors[0];

    // Tensor rank
    TT_FATAL(input.get_logical_shape().rank() == 4,
        "Input tensor must be 4D (N, H, W, C), got rank {}", input.get_logical_shape().rank());

    // Layout
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR,
        "Input tensor must be in ROW_MAJOR layout");

    // Dtype
    TT_FATAL(input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::FLOAT32,
        "Input tensor dtype must be bfloat16 or float32, got {}", input.get_dtype());

    // On device
    TT_FATAL(input.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device");

    // Interleaved memory
    TT_FATAL(!input.memory_config().is_sharded(),
        "Input tensor must be DRAM interleaved, sharded memory not supported");

    // Channel alignment
    const uint32_t C = input.get_logical_shape()[-1];
    const uint32_t element_size = input.element_size();
    TT_FATAL((C * element_size) % 32 == 0,
        "Channel dimension must be aligned to 32 bytes, got {} channels * {} bytes = {} bytes",
        C, element_size, C * element_size);

    // Expand parameter
    TT_FATAL(!expand_,
        "expand=True is not supported. Only same-size rotation (expand=False) is implemented");
}
```

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification, Validation Requirements |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Compile-Time Arguments, Runtime Arguments |
| **ttnn-kernel-dataflow** | Data Flow, Memory Access Patterns, Reader Kernel CT/RT args |
| **ttnn-kernel-compute** | Mathematical Definition (bilinear weights), Compute Kernel CT args |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

### Validation Behavior
- 3D input tensor -> error containing "must be 4D"
- 5D input tensor -> error containing "must be 4D"
- TILE_LAYOUT input -> error containing "ROW_MAJOR layout"
- uint16 dtype -> error containing "bfloat16 or float32"
- Host tensor -> error containing "must be on device"
- Sharded input -> error containing "DRAM interleaved"
- Unaligned channels -> error containing "aligned to 32 bytes"
- expand=true -> error containing "expand=True is not supported"

### Shape Behavior
- Output shape always equals input shape for all valid inputs
- Output dtype equals input dtype
- Output layout is ROW_MAJOR

### Functional Behavior

**Identity rotation (angle=0)**:
- Output should exactly match input (within floating point tolerance)

**90 degree rotation**:
- For square image: output[n, x, W-1-y, c] = input[n, y, x, c]
- Compare against `torch.rot90(input.permute(0,3,1,2), k=1).permute(0,2,3,1)`

**180 degree rotation**:
- output[n, H-1-y, W-1-x, c] = input[n, y, x, c]
- Compare against `torchvision.transforms.functional.rotate(input, 180)`

**Arbitrary angle rotation**:
- Compare against `torchvision.transforms.functional.rotate(input, angle, center=center, fill=fill, expand=False)`
- Tolerance: atol=1e-2, rtol=1e-2 (bilinear interpolation may have minor differences)

**Boundary handling**:
- Corners of rotated image should contain fill_value when original corners rotate out of bounds
- For 45 degree rotation of centered image, corners should be fill_value

**Batch processing**:
- Each image in batch should be rotated independently
- Same result as processing each image individually

**Numerical accuracy**:
- Compare output against PyTorch reference: `torchvision.transforms.functional.rotate`
- Use relaxed tolerances (atol=1e-2) due to bfloat16 precision

## Files to Create

| File | Purpose | Based On |
|------|---------|----------|
| `image_rotate/device/image_rotate_op.hpp` | Operation struct declaration | grid_sample_op.hpp |
| `image_rotate/device/image_rotate_op.cpp` | validate, compute_output_specs, create_program | grid_sample_op.cpp |
| `image_rotate/device/image_rotate_program_factory.cpp` | Program factory with CB setup, kernel creation | grid_sample_bilinear_program_factory.cpp |
| `image_rotate/device/kernels/dataflow/reader_image_rotate_interleaved.cpp` | Reader kernel with rotation math | reader_grid_sample_interleaved_start_id.cpp |
| `image_rotate/image_rotate.hpp` | TTNN API declaration | grid_sample/grid_sample.hpp |
| `image_rotate/image_rotate.cpp` | TTNN API implementation | grid_sample/grid_sample.cpp |

**Reused unchanged**:
- `pool/generic/device/kernels/compute/compute_pool_2d.cpp`
- `pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp`

## Open Questions

1. **Fill value handling for non-zero values**: The current design zeros the input CB and skips reads for out-of-bounds. For non-zero fill values, we need to either:
   - Pre-fill with fill_value instead of zero
   - Apply fill_value in compute kernel when all weights are zero

   **Recommendation**: Pre-fill with fill_value in reader kernel.

2. **Float32 grid dtype equivalent**: grid_sample supports float32 grids. For image_rotate, the source coordinates are computed as float32 internally. Should we support float32 input tensors with full precision coordinate computation?

   **Recommendation**: Support both bfloat16 and float32 input tensors. Coordinate computation is always float32 internally.

3. **Center coordinate validation**: Should we validate that center coordinates are within image bounds, or allow external center points?

   **Recommendation**: Allow any center point (including external). This matches PyTorch behavior.

## References
- Reference analysis: `/localdev/dnijemcevic/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_analysis.md`
- PyTorch reference: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.rotate.html
- DeepWiki queries:
  - "Are there any existing image transformation operations like affine transform, rotate, or perspective warp in ttnn?" -> Confirmed grid_sample is the primary spatial transformation operation
- Documentation consulted:
  - METALIUM_GUIDE.md: Core architecture and kernel types
  - grid_sample_op.hpp: Operation struct pattern
  - grid_sample_reader_common.hpp: Bilinear interpolation implementation
  - compute_pool_2d.cpp: Reduction compute kernel usage

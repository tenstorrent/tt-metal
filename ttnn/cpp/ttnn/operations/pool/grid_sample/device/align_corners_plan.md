# Plan: Add align_corners=True Support to Grid Sample Nearest Neighbor Operation

## Executive Summary

This document outlines the plan to add support for `align_corners=True` to the nearest neighbor grid sampling operation. Currently, only `align_corners=False` is supported. This enhancement will provide full compatibility with PyTorch's `torch.nn.functional.grid_sample` API.

## Background

### What is align_corners?

The `align_corners` parameter controls how normalized coordinates in the range [-1, 1] are mapped to pixel locations in the input image. This affects the coordinate transformation formula used during sampling.

#### align_corners=False (Currently Implemented)
- Extrema (-1 and 1) map to the **corner points** of corner pixels
- Mapping: `-1` → `-0.5`, `1` → `(size - 1) + 0.5`
- Formula: `unnormalized = ((coord + 1) * size - 1) / 2`
- Simplified: `unnormalized = coord * (size * 0.5) + (size * 0.5 - 0.5)`
- More resolution-agnostic (default since PyTorch 1.3.0)

#### align_corners=True (To Be Implemented)
- Extrema (-1 and 1) map to the **center points** of corner pixels
- Mapping: `-1` → `0`, `1` → `size - 1`
- Formula: `unnormalized = ((coord + 1) / 2) * (size - 1)`
- Simplified: `unnormalized = coord * ((size - 1) / 2) + ((size - 1) / 2)`
- Treats corners as pixel centers

### Example Comparison

For an image with `height=4, width=4`:

| Normalized Coord | align_corners=False | align_corners=True |
|------------------|---------------------|-------------------|
| -1.0 (x or y)    | -0.5               | 0.0               |
| 0.0              | 1.5                | 1.5               |
| 1.0              | 3.5                | 3.0               |

For nearest neighbor with rounding:
- `align_corners=False`: coord=-1.0 → -0.5 → rounds to 0
- `align_corners=True`: coord=-1.0 → 0.0 → rounds to 0

The difference is more pronounced at the boundaries and affects which pixels get sampled.

## Current Implementation Analysis

### Files Currently Involved

1. **Kernel**: `reader_writer_grid_sample_nearest_interleaved.cpp`
   - Lines 49-55: Hardcoded scaling factors for `align_corners=False`
   - Lines 101-102, 176-177: Uses these factors for coordinate transformation

2. **Program Factory**: `grid_sample_nearest_program_factory.cpp`
   - No align_corners parameter currently passed
   - Would need to pass this to kernel as compile-time arg

3. **Operation Definition**: `grid_sample_op.hpp`
   - `GridSample` struct (line 18-40): No align_corners field
   - `grid_sample_nearest_program_factory` signature (line 56-62): No align_corners parameter

4. **High-Level API**: `grid_sample.hpp`
   - `ExecuteGridSample::invoke` (line 32-39): No align_corners parameter

5. **Coordinate Reader**: `grid_sample_reader_common.hpp`
   - `GridCoordinateReaderNearest` (line 138-185): Uses height_scale/offset from caller
   - Template-based design allows compile-time specialization

### Current Formula in Code

```cpp
// Lines 49-55 in reader_writer_grid_sample_nearest_interleaved.cpp
constexpr float input_height_f = static_cast<float>(input_height);
constexpr float input_width_f = static_cast<float>(input_width);
constexpr float height_scale = input_height_f * 0.5f;              // H/2
constexpr float height_offset = height_scale - 0.5f;               // H/2 - 0.5
constexpr float width_scale = input_width_f * 0.5f;                // W/2
constexpr float width_offset = width_scale - 0.5f;                 // W/2 - 0.5
// Result: coord * (H/2) + (H/2 - 0.5) = ((coord + 1) * H - 1) / 2 ✓
```

## Implementation Strategy

### Design Decisions

1. **Compile-time vs Runtime Parameter**
   - **Decision**: Make `align_corners` a **compile-time parameter**
   - **Rationale**:
     - Allows compiler optimizations on scaling factors
     - Consistent with other kernel compile-time args (input dimensions, data types)
     - No runtime branching overhead
     - Different kernels cached for each align_corners value

2. **API Changes**
   - Add `align_corners` parameter to all layers of the stack
   - Default to `False` for backward compatibility (matches PyTorch default)
   - Make it optional in Python API with warning if not specified

3. **Kernel Modifications**
   - Use template or compile-time conditional to select formula
   - Keep both formulas in code for clarity and maintainability

## Detailed Implementation Plan

### Phase 1: Core Infrastructure Changes

#### 1.1 Update Operation Struct (`grid_sample_op.hpp`)

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_op.hpp`

**Changes**:
```cpp
struct GridSample {
    const std::string mode_;
    const std::string padding_mode_;
    const bool use_precomputed_grid_;
    const bool batch_output_channels_;
    const bool align_corners_;  // ADD THIS
    const tt::tt_metal::MemoryConfig output_mem_config_;

    // ... existing methods ...

    static constexpr auto attribute_names = std::forward_as_tuple(
        "mode", "padding_mode", "use_precomputed_grid", "batch_output_channels",
        "align_corners",  // ADD THIS
        "output_mem_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->mode_,
            this->padding_mode_,
            this->use_precomputed_grid_,
            this->batch_output_channels_,
            this->align_corners_,  // ADD THIS
            this->output_mem_config_);
    }
};
```

**Update program factory signature**:
```cpp
tt::tt_metal::operation::ProgramWithCallbacks grid_sample_nearest_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels,
    bool align_corners);  // ADD THIS
```

#### 1.2 Update Operation Implementation (`grid_sample_op.cpp`)

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_op.cpp`

**Changes in `create_program` method** (around line 246):
```cpp
if (mode_ == "nearest") {
    return grid_sample_nearest_program_factory(
        input_tensor, grid_tensor, output_tensor,
        padding_mode_, use_precomputed_grid_, batch_output_channels_,
        align_corners_);  // ADD THIS
}
```

**No validation changes needed** - align_corners is a simple boolean, no constraints.

#### 1.3 Update Program Factory (`grid_sample_nearest_program_factory.cpp`)

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_nearest_program_factory.cpp`

**Changes**:

1. **Update function signature** (line 41):
```cpp
tt::tt_metal::operation::ProgramWithCallbacks grid_sample_nearest_program_factory(
    const Tensor& input_tensor,
    const Tensor& grid_tensor,
    const Tensor& output_tensor,
    const std::string& padding_mode,
    bool use_precomputed_grid,
    bool batch_output_channels,
    bool align_corners) {  // ADD THIS
```

2. **Add align_corners to compile-time args for both readers** (lines 132-146 and 171-185):
```cpp
std::vector<uint32_t> reader0_compile_time_args = {
    work_cb_index_0,
    grid_cb_index_0,
    input_stick_size,
    grid_stick_size,
    output_stick_size,
    input_height,
    input_width,
    grid_batching_factor,
    static_cast<uint32_t>(grid_tensor.dtype()),
    grid_hw,
    use_precomputed_grid ? 1U : 0U,
    enable_split_reader ? 1U : 0U,
    0U,                                  // reader_id (0 for RISCV_0)
    align_corners ? 1U : 0U              // ADD THIS (ct_arg[13])
};
```

Similarly for `reader1_compile_time_args`.

3. **Update TensorAccessor offset** in kernel:
   - The TensorAccessor args currently start at index 13
   - After adding align_corners, they will start at index 14
   - Update this in kernel compile-time args

#### 1.4 Update Kernel (`reader_writer_grid_sample_nearest_interleaved.cpp`)

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_writer_grid_sample_nearest_interleaved.cpp`

**Changes**:

1. **Read new compile-time arg** (after line 38):
```cpp
constexpr bool align_corners = get_compile_time_arg_val(13);
```

2. **Update TensorAccessor args offset** (line 41):
```cpp
constexpr auto input_args = TensorAccessorArgs<14>();  // Changed from 13
```

3. **Replace hardcoded scaling factors** (lines 49-56):
```cpp
// Grid coordinate scaling factors
constexpr float input_height_f = static_cast<float>(input_height);
constexpr float input_width_f = static_cast<float>(input_width);

// Compute scaling factors based on align_corners
float height_scale, height_offset, width_scale, width_offset;
if constexpr (align_corners) {
    // align_corners=True: ((coord + 1) / 2) * (size - 1)
    //                   = coord * ((size - 1) / 2) + ((size - 1) / 2)
    constexpr float height_scale = (input_height_f - 1.0f) * 0.5f;
    constexpr float height_offset = height_scale;  // Same as scale
    constexpr float width_scale = (input_width_f - 1.0f) * 0.5f;
    constexpr float width_offset = width_scale;
} else {
    // align_corners=False: ((coord + 1) * size - 1) / 2
    //                    = coord * (size / 2) + (size / 2 - 0.5)
    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;
    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;
}
```

**Note**: The `constexpr if` ensures only one branch is compiled, so there's no runtime overhead.

#### 1.5 Update High-Level API (`grid_sample.hpp` and `grid_sample.cpp`)

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample.hpp`

**Changes**:
```cpp
static ttnn::Tensor invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool use_precomputed_grid = false,
    bool batch_output_channels = false,
    bool align_corners = false,  // ADD THIS with default=false
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
```

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample.cpp`

Update the operation construction to pass `align_corners` through.

#### 1.6 Update Python Bindings (`grid_sample_pybind.hpp` / `.cpp`)

**File**: Check these files and add `align_corners` parameter to Python bindings.

**Python API**:
```python
ttnn.grid_sample(
    input_tensor,
    grid,
    mode='bilinear',
    padding_mode='zeros',
    use_precomputed_grid=False,
    batch_output_channels=False,
    align_corners=False,  # ADD THIS
    memory_config=None
)
```

### Phase 2: Testing

#### 2.1 Unit Tests

**Create**: `ttnn/cpp/ttnn/operations/pool/grid_sample/test_dev/test_align_corners.py`

**Test Cases**:

1. **Basic functionality - align_corners=False** (baseline)
   - Verify existing behavior still works
   - Compare against PyTorch with `align_corners=False`

2. **Basic functionality - align_corners=True**
   - Simple test case with known coordinates
   - Compare against PyTorch with `align_corners=True`

3. **Boundary coordinates** (most critical for align_corners)
   - Test coords at exactly -1.0, 0.0, 1.0
   - Verify correct pixel sampling at boundaries
   - Both align_corners modes

4. **Various input shapes**
   - Small: 4x4
   - Medium: 32x32
   - Large: 224x224
   - Non-square: 16x32, 64x48

5. **Edge cases**
   - Single pixel input (1x1) - should behave identically for both modes
   - 2x2 input - minimal case where difference is visible
   - Coordinates slightly outside [-1, 1] range (should pad with zeros)

6. **Grid configurations**
   - Standard grid (2 elements per point)
   - Different data types (bfloat16, float32)
   - Various output grid sizes

7. **Batch processing**
   - Multiple batches (N > 1)
   - Batch output channels mode
   - Grid batching factor > 1

**Test Structure**:
```python
import pytest
import torch
import torch.nn.functional as F
import ttnn

@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("input_shape", [(1, 4, 4, 3), (1, 32, 32, 16), (2, 8, 8, 8)])
@pytest.mark.parametrize("output_size", [(4, 4), (8, 8), (16, 16)])
def test_grid_sample_nearest_align_corners(device, align_corners, input_shape, output_size):
    """Test grid_sample nearest mode with different align_corners settings."""
    N, H_in, W_in, C = input_shape
    H_out, W_out = output_size

    # Create input tensor
    input_torch = torch.randn(N, C, H_in, W_in)

    # Create normalized grid in [-1, 1]
    grid_torch = create_test_grid(N, H_out, W_out)

    # PyTorch reference
    expected = F.grid_sample(
        input_torch, grid_torch,
        mode='nearest',
        padding_mode='zeros',
        align_corners=align_corners
    )

    # Convert to TTNN format (NHWC)
    input_ttnn = ttnn.from_torch(input_torch.permute(0, 2, 3, 1), device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, device=device)

    # TTNN operation
    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn,
        mode='nearest',
        padding_mode='zeros',
        align_corners=align_corners
    )

    # Convert back and compare
    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2)

    assert torch.allclose(output_torch, expected, rtol=1e-2, atol=1e-3)

def test_boundary_coordinates_align_corners():
    """Test specific boundary coordinates where align_corners makes a difference."""
    # Input: 4x4 image with unique values at each pixel
    input_torch = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    # Grid with exact boundary coordinates
    grid_coords = torch.tensor([
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],  # Top-left, center, bottom-right
    ])
    grid_torch = grid_coords.unsqueeze(0)  # (1, 1, 3, 2)

    # Test align_corners=False
    output_false = F.grid_sample(
        input_torch, grid_torch,
        mode='nearest', padding_mode='zeros', align_corners=False
    )

    # Test align_corners=True
    output_true = F.grid_sample(
        input_torch, grid_torch,
        mode='nearest', padding_mode='zeros', align_corners=True
    )

    # Expected values:
    # align_corners=False: -1.0 maps to -0.5 (rounds to pixel 0)
    # align_corners=True:  -1.0 maps to 0.0 (pixel 0)
    # Both should sample pixel 0 at top-left

    # For coord=1.0:
    # align_corners=False: 1.0 maps to 3.5 (rounds to pixel 4, out of bounds → 0)
    # align_corners=True:  1.0 maps to 3.0 (pixel 3)

    print(f"align_corners=False output: {output_false}")
    print(f"align_corners=True output: {output_true}")

    # Verify against TTNN
    # ... (similar testing with TTNN)

def test_single_pixel_input():
    """For 1x1 input, both modes should behave identically."""
    input_torch = torch.tensor([[[[5.0]]]])  # (1, 1, 1, 1)
    grid_torch = torch.tensor([[[[0.0, 0.0]]]]))  # (1, 1, 1, 2)

    output_false = F.grid_sample(input_torch, grid_torch, mode='nearest', align_corners=False)
    output_true = F.grid_sample(input_torch, grid_torch, mode='nearest', align_corners=True)

    assert torch.allclose(output_false, output_true)
    # Both should output 5.0

def test_precomputed_grid_align_corners():
    """Test with precomputed grid for both align_corners modes."""
    # This test ensures precomputed grid respects align_corners setting
    # Precomputed grids should be generated with the correct formula
    # based on align_corners value
    pass  # TODO: Implement when precomputed grid generation is updated
```

#### 2.2 Integration Tests

**File**: Add to `ttnn/cpp/ttnn/operations/pool/grid_sample/test_dev/test_stage7_correctness.py`

Update existing tests to parameterize over `align_corners`:
```python
@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_nearest_simple(device, align_corners):
    # ... existing test logic ...
    expected_torch = F.grid_sample(
        input_torch, grid_torch,
        mode='nearest',
        padding_mode='zeros',
        align_corners=align_corners  # Parameterized
    )

    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn,
        mode='nearest',
        padding_mode='zeros',
        align_corners=align_corners
    )
    # ... comparison ...
```

#### 2.3 Sweep Tests

**Create**: Sweep test configuration for align_corners parameter

Add to sweep framework configuration:
```python
{
    "operation": "grid_sample",
    "parameters": {
        "mode": ["nearest"],
        "align_corners": [False, True],  # Test both
        "input_shapes": [...],
        "grid_shapes": [...],
        ...
    }
}
```

#### 2.4 Performance Tests

**Goal**: Verify no performance regression

1. Benchmark both `align_corners=False` and `align_corners=True`
2. Compare against baseline (current implementation)
3. Ensure compile-time specialization eliminates any runtime overhead

### Phase 3: Documentation

#### 3.1 Update Operation Documentation

**File**: Update docstrings in `grid_sample.hpp`

```cpp
/**
 * Grid sample operation for spatial sampling.
 *
 * Samples input tensor at grid locations using nearest neighbor or bilinear interpolation.
 * Grid coordinates are expected to be normalized to [-1, 1] range.
 *
 * Args:
 *   input_tensor: Input tensor of shape (N, H_in, W_in, C)
 *   grid: Sampling grid of shape (N, H_out, W_out, 2) with coordinates in [-1, 1]
 *   mode: Interpolation mode ("bilinear" or "nearest")
 *   padding_mode: How to handle out-of-bounds coordinates (currently only "zeros")
 *   use_precomputed_grid: Whether grid contains precomputed coordinates and weights
 *   batch_output_channels: Whether to extend channels instead of width dimension
 *   align_corners: If True, corner pixels are at grid extrema. If False (default),
 *                  extrema are half pixel outside corner pixels. This affects
 *                  coordinate mapping:
 *                  - False: unnorm = ((coord + 1) * size - 1) / 2
 *                  - True:  unnorm = ((coord + 1) / 2) * (size - 1)
 *   memory_config: Memory configuration for the output tensor
 *
 * Returns:
 *   Output tensor of shape (N, H_out, W_out * K, C) or (N, H_out, W_out, C * K)
 *   depending on batch_output_channels, where K is grid batching factor.
 */
```

#### 3.2 Add Examples

Create example demonstrating the difference:
```python
# Example: Demonstrate align_corners effect at boundaries
import ttnn
import torch

# Create a 4x4 input with values 0-15
input_tensor = torch.arange(16.0).reshape(1, 1, 4, 4)

# Sample at the corners
grid = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]]])

# align_corners=False: corners at (-0.5, -0.5) and (3.5, 3.5)
output_false = ttnn.grid_sample(input_tensor, grid, align_corners=False)
# Samples pixels at: (0, 0) and (out of bounds) = [0.0, 0.0]

# align_corners=True: corners at (0, 0) and (3, 3)
output_true = ttnn.grid_sample(input_tensor, grid, align_corners=True)
# Samples pixels at: (0, 0) and (3, 3) = [0.0, 15.0]
```

#### 3.3 Update IMPLEMENTATION_STATUS.md

**File**: `ttnn/cpp/ttnn/operations/pool/grid_sample/device/IMPLEMENTATION_STATUS.md`

Add entry documenting align_corners support:
```markdown
## Grid Sample Nearest Mode

### Supported Features
- [x] Interleaved memory layout
- [x] align_corners=False (default)
- [x] align_corners=True
- [x] padding_mode="zeros"
- [x] Split reader optimization (RISCV_0 and RISCV_1)
- [x] Standard grid (2 coords per point)
- [x] Precomputed grid (6 values per point)
- [x] Grid batching (multiple samples per grid row)

### Parameter Details

#### align_corners
- **False** (default): Extrema of normalized coordinates map to pixel corners
  - Formula: `pixel = ((coord + 1) * size - 1) / 2`
  - Coordinate -1 maps to pixel -0.5, coordinate 1 maps to pixel (size - 0.5)
- **True**: Extrema map to pixel centers
  - Formula: `pixel = ((coord + 1) / 2) * (size - 1)`
  - Coordinate -1 maps to pixel 0, coordinate 1 maps to pixel (size - 1)
```

### Phase 4: Validation and Quality Assurance

#### 4.1 Pre-merge Checklist

- [ ] All unit tests pass for both `align_corners=False` and `align_corners=True`
- [ ] Integration tests pass
- [ ] Sweep tests pass
- [ ] No performance regression (within 1% of baseline)
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Python API matches PyTorch signature
- [ ] Backward compatibility verified (existing code works without changes)

#### 4.2 Correctness Validation Strategy

1. **Cross-reference with PyTorch**
   - Test on identical inputs/grids
   - Verify bit-exact or near-exact match (allowing for fp precision)
   - Test suite with 100+ random test cases

2. **Boundary Testing**
   - Special attention to coordinates at -1.0, -0.99, 0.0, 0.99, 1.0
   - Verify rounding behavior matches PyTorch

3. **Edge Cases**
   - Empty grids (0 points)
   - Single pixel input (1x1)
   - Maximum size inputs supported by hardware
   - Out-of-range coordinates (slightly outside [-1, 1])

#### 4.3 Known Issues / Limitations

Document any discovered issues:
- Precision differences due to bfloat16 vs float32 arithmetic
- Any behavioral differences from PyTorch (should be none)

## Migration Guide

### For Existing Users

**No changes required!** The default behavior (`align_corners=False`) matches the current implementation.

**If you want to use align_corners=True:**
```python
# Old code (implicitly uses align_corners=False)
output = ttnn.grid_sample(input, grid, mode='nearest')

# New code (explicitly specify align_corners=True)
output = ttnn.grid_sample(input, grid, mode='nearest', align_corners=True)
```

### API Compatibility

This change is **backward compatible**:
- Default parameter value is `False` (current behavior)
- Existing calls without the parameter continue to work unchanged
- No breaking changes to any API signatures (only additions)

## Timeline and Milestones

### Phase 1: Core Implementation (Est. 2-3 days)
- [ ] Update operation struct and signatures
- [ ] Modify kernel coordinate scaling logic
- [ ] Update program factory
- [ ] Update high-level API
- [ ] Update Python bindings

### Phase 2: Testing (Est. 2-3 days)
- [ ] Write comprehensive unit tests
- [ ] Add integration tests
- [ ] Configure sweep tests
- [ ] Run performance benchmarks

### Phase 3: Documentation (Est. 1 day)
- [ ] Update docstrings
- [ ] Add examples
- [ ] Update status documents

### Phase 4: Review and Validation (Est. 1-2 days)
- [ ] Code review
- [ ] Validation against PyTorch
- [ ] Final QA

**Total Estimated Time**: 6-9 days

## Risk Analysis

### Low Risk
- Simple boolean parameter
- Compile-time specialization (no runtime branching)
- Clear mathematical formulas
- Direct PyTorch reference for validation

### Potential Issues
1. **Precomputed Grid Compatibility**
   - Risk: Precomputed grids may have been generated with align_corners=False assumption
   - Mitigation: Document that precomputed grids must match align_corners setting
   - Future: Add align_corners metadata to precomputed grid format

2. **Test Coverage**
   - Risk: Missing edge cases in testing
   - Mitigation: Comprehensive test suite with parametrized tests
   - Use property-based testing (hypothesis) for random inputs

3. **Backward Compatibility**
   - Risk: Unintentional behavior change
   - Mitigation: Default value matches current behavior
   - Extensive regression testing

## References

1. PyTorch grid_sample documentation: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
2. PyTorch source code: `grid_sampler_unnormalize` function
3. Current implementation: `reader_writer_grid_sample_nearest_interleaved.cpp`
4. Program factory pattern: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html

## Appendix: Mathematical Derivation

### Coordinate Transformation Formulas

#### Derivation for align_corners=False

Normalized coordinate space: `[-1, 1]`
Pixel space: `[-0.5, size - 0.5]` (corners at pixel edges)

Linear mapping: `pixel = a * coord + b`

Solve for a and b:
- `coord = -1` → `pixel = -0.5`: `-a + b = -0.5`
- `coord = 1` → `pixel = size - 0.5`: `a + b = size - 0.5`

Adding equations: `2b = size - 1` → `b = (size - 1) / 2 = size/2 - 0.5`
Subtracting: `2a = size` → `a = size / 2`

Result: `pixel = coord * (size/2) + (size/2 - 0.5)`
Simplify: `pixel = (coord * size + size - 1) / 2 = ((coord + 1) * size - 1) / 2` ✓

#### Derivation for align_corners=True

Normalized coordinate space: `[-1, 1]`
Pixel space: `[0, size - 1]` (corners at pixel centers)

Linear mapping: `pixel = a * coord + b`

Solve for a and b:
- `coord = -1` → `pixel = 0`: `-a + b = 0` → `b = a`
- `coord = 1` → `pixel = size - 1`: `a + b = size - 1`

Substituting `b = a`: `2a = size - 1` → `a = (size - 1) / 2`

Result: `pixel = coord * ((size - 1)/2) + (size - 1)/2`
Simplify: `pixel = ((coord + 1) / 2) * (size - 1)` ✓

### Special Cases

#### Single Pixel (size = 1)

**align_corners=False**:
- `pixel = ((coord + 1) * 1 - 1) / 2 = coord / 2`
- At `coord = 0`: `pixel = 0` ✓

**align_corners=True**:
- `pixel = ((coord + 1) / 2) * 0 = 0` (always pixel 0)
- Both modes map to same pixel for size=1

#### Two Pixels (size = 2)

**align_corners=False**:
- `coord = -1` → `pixel = -0.5` (rounds to 0)
- `coord = 1` → `pixel = 1.5` (rounds to 2, out of bounds)

**align_corners=True**:
- `coord = -1` → `pixel = 0`
- `coord = 1` → `pixel = 1`

This is where the difference becomes visible.

---

## Approval and Sign-off

Once reviewed and approved, this plan will guide the implementation of align_corners=True support for the grid_sample nearest neighbor operation.

**Implementation Owner**: [TBD]
**Reviewer**: [TBD]
**Target Completion**: [TBD]

# TDD Implementation Plan for Grid Sample Nearest Neighbor Mode

## Executive Summary
This document outlines a test-driven development plan for implementing the "nearest" sampling mode for the TTNN grid_sample operation. The plan follows a strict 7-stage progression, with each stage having failing tests that drive minimal implementation. Work ONLY with debug builds.

## Architecture Overview

### Operation Stack (8 layers)
1. **Python API** (`grid_sample_pybind.cpp`) - User interface
2. **High-Level Operation** (`grid_sample.hpp/cpp`) - ExecuteGridSample struct
3. **Device Operation** (`grid_sample_op.hpp/cpp`) - GridSample struct
4. **Program Factory** (`grid_sample_nearest_program_factory.cpp`) - Hardware program creation
5. **Common Utilities** (`grid_sample_reader_common.hpp`) - Shared coordinate logic
6. **Reader Kernels** - Data movement from DRAM/grids
7. **Compute Kernel** - Not needed for nearest (pure data movement)
8. **Writer Kernel** - Output to DRAM

### Key Differences: Bilinear vs Nearest

| Aspect | Bilinear | Nearest |
|--------|----------|---------|
| Reads per point | 4 corner pixels | 1 pixel |
| Interpolation weights | 4 weights computed | None (implicit 1.0) |
| Compute kernel | Required (weighted sum) | Not needed |
| Coordinate calculation | floor() + fractional | round() only |
| Memory bandwidth | 4x reads | 1x reads |
| Circular buffers | c_0-c_5 (including weights) | c_0, c_1, c_5 only |

## Stage 1: API Existence
**Goal**: Make the nearest mode accessible from Python

### Test: `test_dev/test_stage1_api_exists.py`
```python
import ttnn
import torch
import pytest

def test_nearest_mode_exists():
    """Test that grid_sample accepts mode='nearest'"""
    device = ttnn.open_device(device_id=0)

    # Create minimal tensors
    input_tensor = torch.ones((1, 1, 2, 2), dtype=torch.bfloat16)
    grid = torch.tensor([[[[-1, -1], [1, 1]]]], dtype=torch.bfloat16)

    input_t = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # This should not raise AttributeError but should fail with NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    assert "nearest mode not yet implemented" in str(exc_info.value).lower()

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Modify `grid_sample_pybind.cpp`**:
   - Update mode parameter validation to accept "nearest"
   - Pass mode through to ExecuteGridSample
   - Add NotImplementedError for nearest mode initially

2. **Files to modify**:
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample_pybind.cpp`

### Expected Error:
```
NotImplementedError: Nearest mode not yet implemented
```

## Stage 2: Parameter Validation
**Goal**: Validate inputs properly for nearest mode

### Test: `test_dev/test_stage2_validation.py`
```python
import ttnn
import torch
import pytest

def test_nearest_mode_validation():
    """Test parameter validation for nearest mode"""
    device = ttnn.open_device(device_id=0)

    # Test 1: Invalid tensor rank
    input_3d = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    grid = torch.ones((1, 2, 2, 2), dtype=torch.bfloat16)

    input_t = ttnn.from_torch(input_3d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.grid_sample(input_t, grid_t, mode="nearest")
    assert "must be 4D" in str(exc_info.value)

    # Test 2: Invalid grid shape
    input_4d = torch.ones((1, 1, 2, 2), dtype=torch.bfloat16)
    grid_wrong = torch.ones((1, 2, 2, 3), dtype=torch.bfloat16)  # Wrong last dim

    input_t = ttnn.from_torch(input_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid_wrong, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.grid_sample(input_t, grid_t, mode="nearest")
    assert "last dimension must be 2" in str(exc_info.value)

    # Test 3: Valid inputs should fail at device op level
    input_valid = torch.ones((1, 1, 4, 4), dtype=torch.bfloat16)
    grid_valid = torch.ones((1, 2, 2, 2), dtype=torch.bfloat16)

    input_t = ttnn.from_torch(input_valid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid_valid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.grid_sample(input_t, grid_t, mode="nearest")
    assert "device operation" in str(exc_info.value).lower()

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Update `grid_sample.cpp`** (ExecuteGridSample):
   - Add validation for tensor ranks (input must be 4D, grid must be 4D)
   - Validate grid last dimension is 2 or 6 (depending on precomputed)
   - Validate data types (BFLOAT16 for input, BFLOAT16/FLOAT32 for grid)
   - Validate memory layout (ROW_MAJOR required)
   - Pass validation, then fail at device op creation

2. **Files to modify**:
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample.cpp`

### Expected Error Progression:
```
Stage 2.1: "Input tensor must be 4D"
Stage 2.2: "Grid last dimension must be 2"
Stage 2.3: "Device operation not implemented for nearest mode"
```

## Stage 3: TTNN Registration
**Goal**: Properly register the operation with TTNN framework

### Test: `test_dev/test_stage3_registration.py`
```python
import ttnn
import torch
import pytest
import inspect

def test_nearest_mode_registration():
    """Test that nearest mode is properly registered with TTNN"""

    # Test 1: Operation has correct signature
    sig = inspect.signature(ttnn.grid_sample)
    assert 'input' in sig.parameters
    assert 'grid' in sig.parameters
    assert 'mode' in sig.parameters
    assert 'padding_mode' in sig.parameters

    # Test 2: Mode parameter accepts 'nearest'
    device = ttnn.open_device(device_id=0)

    input_t = ttnn.ones((1, 1, 4, 4), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.ones((1, 2, 2, 2), device=device, dtype=ttnn.bfloat16)

    # Should fail at device operation level, not registration
    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Error should mention program factory or device op, not registration
    error_msg = str(exc_info.value).lower()
    assert ("program" in error_msg or "device" in error_msg)
    assert "not registered" not in error_msg

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Ensure registration in `grid_sample.cpp`**:
   - Verify ttnn::register_operation is properly configured
   - Update invoke() to handle mode="nearest"
   - Forward to device operation

2. **Files to verify/modify**:
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample.cpp`
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample.hpp`

### Expected Error:
```
RuntimeError: Program factory not implemented for nearest mode
```

## Stage 4: Device Operation
**Goal**: Implement device operation validation and output shape computation

### Test: `test_dev/test_stage4_device_op.py`
```python
import ttnn
import torch
import pytest

def test_nearest_device_operation():
    """Test device operation validates and computes output shape"""
    device = ttnn.open_device(device_id=0)

    # Test 1: Validate memory config
    input_t = ttnn.ones((1, 2, 8, 8), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.ones((1, 4, 4, 2), device=device, dtype=ttnn.bfloat16)

    # Should compute output shape correctly: [1, 2, 4, 4]
    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should fail at program factory creation, not device op
    error_msg = str(exc_info.value).lower()
    assert "program factory" in error_msg or "create_program" in error_msg
    assert "validate" not in error_msg

    # Test 2: Batch size mismatch should be caught
    input_batch2 = ttnn.ones((2, 2, 8, 8), device=device, dtype=ttnn.bfloat16)
    grid_batch1 = ttnn.ones((1, 4, 4, 2), device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_batch2, grid_batch1, mode="nearest")
    assert "batch size" in str(exc_info.value).lower()

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Update `grid_sample_op.cpp`** (GridSample device operation):
   - In `validate()`: Add nearest mode support with proper validation
   - In `compute_output_specs()`: Calculate output shape [N, C, grid_H, grid_W]
   - In `create_program()`: Route to nearest program factory when mode=="nearest"
   - Initially throw "Program factory not implemented" error

2. **Files to modify**:
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_op.cpp`
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_op.hpp`

### Expected Error:
```
RuntimeError: Program factory not implemented for nearest mode
```

## Stage 5: Program Factory
**Goal**: Create program structure with circular buffers and work distribution

### Test: `test_dev/test_stage5_program_factory.py`
```python
import ttnn
import torch
import pytest

def test_nearest_program_factory():
    """Test program factory creates proper structure"""
    device = ttnn.open_device(device_id=0)

    # Small tensor that should create program
    input_t = ttnn.ones((1, 1, 32, 32), device=device, dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    grid_t = ttnn.ones((1, 16, 16, 2), device=device, dtype=ttnn.bfloat16,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should fail mentioning kernels, not program structure
    error_msg = str(exc_info.value).lower()
    assert ("kernel" in error_msg or "reader" in error_msg or
            "file not found" in error_msg)
    assert "circular buffer" not in error_msg

    ttnn.close_device(device)

def test_nearest_work_distribution():
    """Test work is properly distributed across cores"""
    device = ttnn.open_device(device_id=0)

    # Larger tensor to test multi-core distribution
    input_t = ttnn.ones((2, 4, 64, 64), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.ones((2, 32, 32, 2), device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should still fail at kernel level
    assert "kernel" in str(exc_info.value).lower()

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Complete `grid_sample_nearest_program_factory.cpp`**:
   - Set up circular buffers (c_0 for grid, c_1 for input, c_5 for output)
   - Calculate work distribution using split_work_to_cores()
   - Set compile-time arguments for kernels
   - Set runtime arguments (buffer addresses, work sizes)
   - Reference kernel files (will fail as they don't exist yet)

2. **Circular Buffer Setup**:
   ```cpp
   // Grid coordinates buffer
   uint32_t cb_grid = tt::CBIndex::c_0;

   // Input data buffer (only 1 for nearest, vs 4 for bilinear)
   uint32_t cb_input = tt::CBIndex::c_1;

   // Output buffer
   uint32_t cb_output = tt::CBIndex::c_5;
   ```

3. **Files to modify**:
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_nearest_program_factory.cpp`

### Expected Error:
```
RuntimeError: Kernel file not found: reader_grid_sample_nearest_interleaved.cpp
```

## Stage 6: Kernel Compilation
**Goal**: Create kernels that compile without errors

### Test: `test_dev/test_stage6_kernel_compile.py`
```python
import ttnn
import torch
import pytest

def test_nearest_kernels_compile():
    """Test that kernels compile without syntax errors"""
    device = ttnn.open_device(device_id=0)

    # Minimal test - just needs to compile
    input_t = ttnn.ones((1, 1, 32, 32), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.zeros((1, 8, 8, 2), device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should fail at runtime, not compilation
    error_msg = str(exc_info.value).lower()
    assert "compile" not in error_msg
    assert "syntax" not in error_msg
    # May have runtime errors or incorrect results

    ttnn.close_device(device)

def test_sharded_kernels_compile():
    """Test sharded version compiles"""
    device = ttnn.open_device(device_id=0)

    # Create sharded tensors
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreGrid(x=1, y=1),
        [32, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
        False
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.DRAM,
        shard_spec
    )

    input_t = ttnn.ones((1, 1, 32, 32), device=device, dtype=ttnn.bfloat16,
                        memory_config=mem_config)
    grid_t = ttnn.ones((1, 8, 8, 2), device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should compile but may have runtime issues
    assert "compile" not in str(exc_info.value).lower()

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Create `reader_grid_sample_nearest_interleaved.cpp`**:
   ```cpp
   // Minimal compilable kernel
   #include "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_interleaved_start_id.hpp"

   void kernel_main() {
       // Minimal implementation that compiles
       const uint32_t num_work_units = get_arg_val<uint32_t>(0);

       for (uint32_t i = 0; i < num_work_units; i++) {
           // TODO: Implement actual logic
           // For now, just push dummy data
           cb_reserve_back(tt::CBIndex::c_5, 1);
           cb_push_back(tt::CBIndex::c_5, 1);
       }
   }
   ```

2. **Create `reader_grid_sample_nearest_sharded.cpp`**:
   - Similar minimal structure for sharded version

3. **Create `writer_grid_sample_nearest_interleaved.cpp`**:
   - Minimal writer that compiles

4. **Files to create**:
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_nearest_interleaved.cpp`
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_nearest_sharded.cpp`
   - `ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_interleaved.cpp`

### Expected Behavior:
- Kernels compile successfully
- May have runtime errors or produce incorrect results

## Stage 7: Kernel Correctness
**Goal**: Implement correct nearest neighbor sampling logic

### Test: `test_dev/test_stage7_kernel_correct.py`
```python
import ttnn
import torch
import torch.nn.functional as F
import pytest

def test_nearest_identity_transform():
    """Test identity grid (no transformation)"""
    device = ttnn.open_device(device_id=0)

    # Create input
    input_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    # Identity grid - samples at exact pixel centers
    grid = torch.zeros((1, 4, 4, 2))
    for i in range(4):
        for j in range(4):
            grid[0, i, j, 1] = (2.0 * i / 3.0) - 1.0  # y
            grid[0, i, j, 0] = (2.0 * j / 3.0) - 1.0  # x

    # Convert to ttnn
    input_t = ttnn.from_torch(input_tensor.bfloat16(), device=device,
                              layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid.bfloat16(), device=device,
                            layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run operation
    output_t = ttnn.grid_sample(input_t, grid_t, mode="nearest",
                               padding_mode="zeros", align_corners=False)
    output = ttnn.to_torch(output_t)

    # Compare with PyTorch
    expected = F.grid_sample(input_tensor, grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)

    assert torch.allclose(output.float(), expected, rtol=1e-3, atol=1e-3)

    ttnn.close_device(device)

def test_nearest_downsampling():
    """Test 2x downsampling"""
    device = ttnn.open_device(device_id=0)

    # 8x8 input
    input_tensor = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)

    # 4x4 grid for 2x downsampling
    grid = torch.zeros((1, 4, 4, 2))
    for i in range(4):
        for j in range(4):
            grid[0, i, j, 1] = (2.0 * i / 3.0) - 1.0
            grid[0, i, j, 0] = (2.0 * j / 3.0) - 1.0

    input_t = ttnn.from_torch(input_tensor.bfloat16(), device=device,
                              layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid.bfloat16(), device=device,
                            layout=ttnn.ROW_MAJOR_LAYOUT)

    output_t = ttnn.grid_sample(input_t, grid_t, mode="nearest")
    output = ttnn.to_torch(output_t)

    expected = F.grid_sample(input_tensor, grid, mode="nearest",
                           align_corners=False)

    assert torch.allclose(output.float(), expected, rtol=1e-2, atol=1e-2)

    ttnn.close_device(device)

def test_nearest_edge_cases():
    """Test edge cases: out of bounds, borders"""
    device = ttnn.open_device(device_id=0)

    input_tensor = torch.ones((1, 1, 4, 4), dtype=torch.float32)

    # Grid with out-of-bounds coordinates
    grid = torch.tensor([[[[-2, -2], [2, 2]],
                          [[0, 0], [1, 1]]]], dtype=torch.float32)

    input_t = ttnn.from_torch(input_tensor.bfloat16(), device=device,
                              layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid.bfloat16(), device=device,
                            layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with zeros padding
    output_t = ttnn.grid_sample(input_t, grid_t, mode="nearest",
                               padding_mode="zeros")
    output = ttnn.to_torch(output_t)

    expected = F.grid_sample(input_tensor, grid, mode="nearest",
                           padding_mode="zeros", align_corners=False)

    assert torch.allclose(output.float(), expected, rtol=1e-2, atol=1e-2)

    ttnn.close_device(device)

def test_nearest_multichannel():
    """Test multi-channel operation"""
    device = ttnn.open_device(device_id=0)

    # Multi-channel input
    input_tensor = torch.randn((2, 3, 32, 32), dtype=torch.float32)
    grid = torch.randn((2, 16, 16, 2), dtype=torch.float32) * 0.8

    input_t = ttnn.from_torch(input_tensor.bfloat16(), device=device,
                              layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid.bfloat16(), device=device,
                            layout=ttnn.ROW_MAJOR_LAYOUT)

    output_t = ttnn.grid_sample(input_t, grid_t, mode="nearest")
    output = ttnn.to_torch(output_t)

    expected = F.grid_sample(input_tensor, grid, mode="nearest",
                           align_corners=False)

    # Calculate PCC
    output_flat = output.float().flatten()
    expected_flat = expected.flatten()

    correlation = torch.corrcoef(torch.stack([output_flat, expected_flat]))[0, 1]
    assert correlation > 0.99, f"PCC {correlation} is below threshold"

    ttnn.close_device(device)
```

### Implementation Tasks:
1. **Complete `reader_grid_sample_nearest_interleaved.cpp`**:
   - Read grid coordinates and convert to pixel indices
   - Use round() instead of floor() for nearest neighbor
   - Implement boundary handling (padding modes)
   - Read single pixel per grid point (not 4 corners)

2. **Coordinate Calculation** (in reader kernel):
   ```cpp
   // Nearest neighbor coordinate calculation
   float h_norm = grid_h_coord;  // From grid buffer
   float w_norm = grid_w_coord;  // From grid buffer

   // Convert normalized [-1, 1] to pixel coordinates
   float h_pixel = (h_norm + 1.0f) * (H_in - 1) * 0.5f;
   float w_pixel = (w_norm + 1.0f) * (W_in - 1) * 0.5f;

   // Round to nearest integer pixel
   int h_nearest = (int)round(h_pixel);
   int w_nearest = (int)round(w_pixel);

   // Apply padding mode
   if (padding_mode == "zeros") {
       if (h_nearest < 0 || h_nearest >= H_in ||
           w_nearest < 0 || w_nearest >= W_in) {
           // Write zero
       } else {
           // Read pixel at [h_nearest, w_nearest]
       }
   }
   ```

3. **Key Differences from Bilinear**:
   - No weight calculation needed
   - Single NOC read per grid point
   - No compute kernel required
   - Direct copy from input to output buffer

4. **Files to complete**:
   - Full implementation of reader kernels
   - Update program factory runtime arguments
   - Ensure proper synchronization with circular buffers

### Expected Behavior:
- All tests pass with PCC > 0.99 against PyTorch reference

## Integration and Cleanup

After Stage 7 passes:

1. **Move tests to permanent location**:
   ```bash
   # Add nearest mode tests to existing test file
   tests/ttnn/unit_tests/operations/pool/test_grid_sample.py
   ```

2. **Remove temporary test folder**:
   ```bash
   rm -rf ttnn/cpp/ttnn/operations/pool/grid_sample/test_dev/
   ```

3. **Update documentation**:
   - Add nearest mode to operation documentation
   - Update examples to show both modes

4. **Performance validation**:
   - Benchmark against bilinear mode
   - Verify 4x reduction in memory reads
   - Profile with Tracy to ensure no bottlenecks

## Build and Test Commands

### Building
```bash
# Clean build with tests
./build_metal.sh --clean --build-tests

# Incremental build during development
./build_metal.sh
```

### Running Tests for Each Stage
```bash
# Stage 1
pytest ttnn/cpp/ttnn/operations/pool/grid_sample/test_dev/test_stage1_api_exists.py -v

# Stage 2
pytest ttnn/cpp/ttnn/operations/pool/grid_sample/test_dev/test_stage2_validation.py -v

# ... continue for each stage

# Final integration tests
pytest tests/ttnn/unit_tests/operations/pool/test_grid_sample.py::test_nearest -v
```

### Debugging Support
```bash
# Enable debugging output
export TT_METAL_WATCHER=10
export TT_METAL_DPRINT_CORES=(0,0)
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'

# Run with debug output
pytest <test_file> -v -s
```

## Success Criteria

- [ ] Stage 1: API exists and throws NotImplementedError
- [ ] Stage 2: Input validation works correctly
- [ ] Stage 3: Operation is registered with TTNN
- [ ] Stage 4: Device operation validates and computes output shape
- [ ] Stage 5: Program factory creates correct structure
- [ ] Stage 6: Kernels compile without errors
- [ ] Stage 7: All correctness tests pass with PCC > 0.99
- [ ] Integration: Tests merged into main test suite
- [ ] Performance: 4x faster than bilinear for same grid size
- [ ] Documentation: Updated with nearest mode examples

## Timeline Estimate

- Stage 1-3: 2 hours (API and registration)
- Stage 4-5: 4 hours (device op and program factory)
- Stage 6-7: 8 hours (kernel implementation and debugging)
- Integration: 2 hours (cleanup and documentation)

Total: ~16 hours of focused development

## Notes

1. **Strict TDD**: Never proceed to next stage until current stage tests pass
2. **Minimal Implementation**: Only write code to make current test pass
3. **Use DeepWiki**: For architecture questions, not blind searching
4. **Test First**: Write test, see it fail, then implement
5. **Incremental Progress**: Each stage builds on previous
6. **Clear Errors**: Each stage should have meaningful error messages

This plan ensures systematic development with confidence at each stage.

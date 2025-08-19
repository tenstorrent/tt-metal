# Detailed Implementation Plan for 3D Upsample Operation in TTNN

## Overview
Implement a 3D upsample operation (`upsample3d`) in TTNN that supports nearest-neighbor interpolation for 5D tensors (N, D, H, W, C) in row-major DRAM interleaved layout.

## 1. Core Tensor Shape Requirements

**Input tensor shape**: `[N, D, H, W, C]`
- N = batch size
- D = depth dimension
- H = height dimension
- W = width dimension
- C = channels

**Output tensor shape**: `[N, D*scale_d, H*scale_h, W*scale_w, C]`
- Each dimension is scaled by its respective scale factor
- For nearest neighbor: each input voxel is replicated `scale_d × scale_h × scale_w` times

## 2. File Structure and Components to Create

### 2.1 Core Operation Files
```
ttnn/cpp/ttnn/operations/pool/upsample3d/
├── CMakeLists.txt                   # Build configuration
├── upsample3d.hpp                    # Public API interface
├── upsample3d.cpp                    # Implementation logic
├── upsample3d_pybind.cpp            # Python bindings
└── device/
    ├── upsample3d_op.hpp            # Device operation header
    ├── upsample3d_op.cpp            # Device operation implementation
    ├── upsample3d_program_factory_multicore_interleaved.cpp  # Main factory
    └── kernels/
        └── dataflow/
            ├── reader_upsample3d_interleaved.cpp    # Reader kernel
            └── writer_upsample3d_interleaved.cpp    # Writer kernel
```

### 2.2 Integration Files to Modify

**Build System Integration:**
- Update `ttnn/cpp/ttnn/operations/pool/CMakeLists.txt` to add upsample3d source files
- Update `ttnn/CMakeLists.txt` to include upsample3d pybind files

**Python Bindings Integration:**
- Update `ttnn/cpp/ttnn-pybind/__init__.cpp` to include upsample3d pybind header and call py_module function
- Create `ttnn/python/ttnn/operations/pool/upsample3d.py` wrapper for Python API

**C++ Header Integration:**
- Add upsample3d operation to appropriate ttnn header includes for C++ usage

## 3. Implementation Details

### 3.1 UpSample3D Operation Structure
```cpp
struct UpSample3D {
    const int scale_factor_d_;  // New: depth scale factor
    const int scale_factor_h_;
    const int scale_factor_w_;
    const std::string mode_;    // Only "nearest" initially
};
```

### 3.2 Memory Layout Strategy

**Work Unit Definition**:
- A "stick" is one page in memory containing `C` elements (the innermost dimension)
- For 5D tensor `[N, D, H, W, C]` in row-major layout, the tensor is stored as 2D with:
  - First dimension: `N × D × H × W` (flattened outer dimensions)
  - Second dimension: `C` (channels - one page per stick)
- Total work units = `N × D × H × W` sticks, each stick is one page distributed across memory banks
- Each work unit processes one input stick (C elements) and generates `scale_d × scale_h × scale_w` output sticks

**Memory Access Pattern**:
- Input: `[N, D, H, W, C]` → Physical volume = `N × D × H × W` pages of `C` elements each
- Output: `[N, D*scale_d, H*scale_h, W*scale_w, C]` → Each input page generates `scale_d × scale_h × scale_w` output pages
- Pages are distributed across memory banks in round-robin fashion

### 3.3 Kernel Implementation Strategy

**Reader Kernel**:
- Reads one input page at a time: `C` elements (one stick)
- Takes linear page index as input and reads corresponding page from interleaved DRAM
- No coordinate conversion needed - works directly with flattened page indices

**Writer Kernel**:
- For each input page, writes `scale_d × scale_h × scale_w` output pages
- Each output page contains the same `C` channel values as the input page
- Handles coordinate mapping from input `(n, d, h, w)` to all corresponding output positions:
  ```
  input_page[n×D×H×W + d×H×W + h×W + w] →
  output_page[n×D'×H'×W' + d'×H'×W' + h'×W' + w']
  where: d' ∈ [d×scale_d, (d+1)×scale_d)
         h' ∈ [h×scale_h, (h+1)×scale_h)
         w' ∈ [w×scale_w, (w+1)×scale_w)
  ```

### 3.4 Core Algorithm (Nearest Neighbor)

For input page at coordinate `(n, d, h, w)` containing `C` elements:
1. Calculate output page coordinates:
   - `d_start = d * scale_d`, `d_end = (d+1) * scale_d`
   - `h_start = h * scale_h`, `h_end = (h+1) * scale_h`
   - `w_start = w * scale_w`, `w_end = (w+1) * scale_w`

2. Replicate input page to all corresponding output pages:
   ```
   input_page_idx = n×D×H×W + d×H×W + h×W + w

   for d' in [d_start, d_end):
     for h' in [h_start, h_end):
       for w' in [w_start, w_end):
         output_page_idx = n×D'×H'×W' + d'×H'×W' + h'×W' + w'
         copy_page(input_page[input_page_idx], output_page[output_page_idx])
   ```

## 4. STRICT Test-Driven Development Implementation Strategy

### CRITICAL REQUIREMENTS:
- **BUILD COMMAND:** `./build_metal.sh`
- **PYTHON ENV:** `source python_env/bin/activate`
- **ALL INTERMEDIATE TESTS:** Must be in `test_progress_upsample3d/` folder (will be deleted later)
- **FINAL TEST:** Must be in official `tests/` folder of repo
- **SET TIMEOUTS:** All tests must have 20 second timeouts to quickly catch device hangs
- **NO SKIPPING STEPS:** Each step must be completed in order, no jumping ahead
- **EVERY TEST IN PYTHON FILES:** All tests must be written as Python files that can be executed

### Test Directory Structure:
```
test_progress_upsample3d/              # Temporary test folder (will be deleted)
├── step1_test_api_exists.py
├── step2_test_parameter_validation.py
├── step3_test_operation_registration.py
├── step4_test_device_operation.py
├── step5_test_program_factory.py
├── step6_test_kernel_compilation.py
├── step7_test_minimal_working.py
├── step8_test_full_implementation.py
└── utils.py                           # Shared test utilities

tests/ttnn/unit_tests/operations/      # Final test location
└── test_upsample3d.py                 # Final PCC comparison test
```

### Step 1: Python API Stub + Test
**Test File:** `test_progress_upsample3d/step1_test_api_exists.py`
**Implementation:**
- Create basic Python binding that just exists and is callable
- Return "not implemented" error for now

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(30)  # 30 second timeout
def test_upsample3d_api_exists():
    """Test that ttnn.upsample3d exists and is callable"""
    assert hasattr(ttnn, 'upsample3d'), "ttnn.upsample3d does not exist"

    # Try to call it and expect NotImplementedError
    input_tensor = torch.randn(1, 2, 2, 2, 4, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_tensor, device=None, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(NotImplementedError):
        ttnn.upsample3d(input_tensor, (2, 2, 2))

if __name__ == "__main__":
    test_upsample3d_api_exists()
    print("Step 1: API exists test PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step1_test_api_exists.py`
4. Test MUST pass before proceeding to Step 2

### Step 2: Parameter Validation Stub + Test
**Test File:** `test_progress_upsample3d/step2_test_parameter_validation.py`
**Implementation:**
- Add parameter validation in Python binding
- Parse scale factors, validate tensor dimensionality
- Still return "not implemented" but with proper validation

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(30)
def test_upsample3d_parameter_validation():
    """Test parameter validation for upsample3d"""

    # Test invalid tensor dimensions (should fail)
    invalid_tensor_4d = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    invalid_tensor_4d = ttnn.from_torch(invalid_tensor_4d, device=None, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises((ValueError, RuntimeError)):  # Accept either error type
        ttnn.upsample3d(invalid_tensor_4d, (2, 2, 2))

    # Test invalid scale factors (should fail)
    valid_tensor = torch.randn(1, 2, 2, 2, 4, dtype=torch.bfloat16)
    valid_tensor = ttnn.from_torch(valid_tensor, device=None, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises((ValueError, RuntimeError)):
        ttnn.upsample3d(valid_tensor, (0, 2, 2))  # Zero scale factor

    with pytest.raises((ValueError, RuntimeError)):
        ttnn.upsample3d(valid_tensor, (-1, 2, 2))  # Negative scale factor

    # Test valid parameters (should get NotImplementedError)
    with pytest.raises(NotImplementedError):
        ttnn.upsample3d(valid_tensor, (2, 2, 2))

if __name__ == "__main__":
    test_upsample3d_parameter_validation()
    print("Step 2: Parameter validation test PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step2_test_parameter_validation.py`
4. Test MUST pass before proceeding to Step 3

### Step 3: C++ Operation Registration + Test
**Test File:** `test_progress_upsample3d/step3_test_operation_registration.py`
**Implementation:**
- Create minimal `UpSample3D` struct with scale factors only
- Register operation in TTNN (stub invoke method)
- Return dummy tensor with correct output shape

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(20)  # Short timeout to catch hangs
def test_upsample3d_operation_registered():
    """Test that UpSample3D operation is registered in TTNN"""
    device = ttnn.open_device(device_id=0)
    try:
        input_tensor = torch.randn(1, 2, 2, 2, 4, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Should not raise NotImplementedError anymore, should return tensor
        result = ttnn.upsample3d(input_tensor, (2, 2, 2))
        assert result is not None

    finally:
        ttnn.close_device(device)

@pytest.mark.timeout(20)
def test_upsample3d_output_shape_computation():
    """Test output shape computation without actual upsampling logic"""
    device = ttnn.open_device(device_id=0)
    try:
        input_shape = [1, 2, 3, 4, 8]  # N, D, H, W, C
        scale_factors = (2, 3, 4)      # scale_d, scale_h, scale_w
        expected_shape = [1, 4, 9, 16, 8]  # N, D*2, H*3, W*4, C

        input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        result = ttnn.upsample3d(input_tensor, scale_factors)
        result_torch = ttnn.to_torch(result)

        assert list(result_torch.shape) == expected_shape, f"Expected shape {expected_shape}, got {list(result_torch.shape)}"

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_upsample3d_operation_registered()
    test_upsample3d_output_shape_computation()
    print("Step 3: Operation registration tests PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step3_test_operation_registration.py`
4. Both tests MUST pass before proceeding to Step 4

### Step 4: Device Operation + Coordinate Logic + Test
**Test File:** `test_progress_upsample3d/step4_test_device_operation.py`
**Implementation:**
- Add `validate()` method to `UpSample3D`
- Check 5D tensor, positive scale factors, "nearest" mode
- Implement coordinate mapping functions (no actual kernel execution)
- Add buffer size calculations
- Stub `create_program()` method
- **ADD DEBUG PRINTS:** Add temporary debug prints in device operation to show validation steps (DELETE THESE LATER)

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(20)
def test_upsample3d_input_validation():
    """Test input validation in UpSample3D device operation"""
    device = ttnn.open_device(device_id=0)
    try:
        # Test unsupported mode (only "nearest" should be supported initially)
        input_tensor = torch.randn(1, 2, 2, 2, 4, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # This should raise error for unsupported mode (if we add mode parameter)
        # For now, just test that valid case works
        result = ttnn.upsample3d(input_tensor, (2, 2, 2))
        assert result is not None

    finally:
        ttnn.close_device(device)

@pytest.mark.timeout(20)
def test_upsample3d_various_scale_factors():
    """Test that device operation can handle various scale factor combinations"""
    device = ttnn.open_device(device_id=0)
    try:
        # Test different scale factor combinations to exercise validation logic
        test_cases = [
            ([1, 1, 2, 2, 4], (1, 1, 1)),    # No scaling
            ([1, 2, 1, 3, 4], (2, 3, 1)),    # Mixed scale factors
            ([1, 1, 1, 1, 8], (3, 2, 4)),    # Large scale factors
        ]

        for input_shape, scale_factors in test_cases:
            input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
            input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

            # This should work without crashing and return correct output shape
            result = ttnn.upsample3d(input_tensor, scale_factors)
            result_torch = ttnn.to_torch(result)

            expected_shape = [
                input_shape[0],
                input_shape[1] * scale_factors[0],
                input_shape[2] * scale_factors[1],
                input_shape[3] * scale_factors[2],
                input_shape[4]
            ]

            assert list(result_torch.shape) == expected_shape, f"Failed for {input_shape} with {scale_factors}"

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_upsample3d_input_validation()
    test_upsample3d_various_scale_factors()
    print("Step 4: Device operation tests PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step4_test_device_operation.py`
4. All tests MUST pass before proceeding to Step 5

### Step 5: Minimal Program Factory + Test
**Test File:** `test_progress_upsample3d/step5_test_program_factory.py`
**Implementation:**
- Create program factory that creates empty program
- Add dummy reader/writer kernel files that compile but do nothing
- Calculate work distribution correctly
- **ADD DEBUG PRINTS:** Add temporary debug prints in program factory to show work distribution (DELETE THESE LATER)

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(20)  # Short timeout to quickly catch hangs
def test_upsample3d_program_creation():
    """Test that program factory creates programs without crashing"""
    device = ttnn.open_device(device_id=0)
    try:
        # Small test case to verify program creation
        input_tensor = torch.randn(1, 1, 2, 2, 4, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # This should create and run a program (even if it does nothing useful yet)
        result = ttnn.upsample3d(input_tensor, (2, 2, 2))

        # Just verify we got a result tensor with correct shape
        result_torch = ttnn.to_torch(result)
        expected_shape = [1, 2, 4, 4, 4]
        assert list(result_torch.shape) == expected_shape

    finally:
        ttnn.close_device(device)

@pytest.mark.timeout(20)
def test_upsample3d_edge_cases():
    """Test edge cases that exercise work distribution and program creation"""
    device = ttnn.open_device(device_id=0)
    try:
        # Test edge cases that will exercise the work distribution logic
        edge_cases = [
            ([1, 1, 1, 1, 4], (2, 2, 2)),    # Very small tensor - few work units
            ([1, 4, 4, 4, 4], (1, 1, 1)),    # Large tensor, no scaling
            ([2, 1, 1, 8, 16], (1, 2, 1)),   # Wide tensor
            ([1, 8, 1, 1, 4], (2, 1, 2)),    # Tall tensor
        ]

        for input_shape, scale_factors in edge_cases:
            input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
            input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

            # This exercises the work distribution logic in program factory
            result = ttnn.upsample3d(input_tensor, scale_factors)
            result_torch = ttnn.to_torch(result)

            expected_shape = [
                input_shape[0],
                input_shape[1] * scale_factors[0],
                input_shape[2] * scale_factors[1],
                input_shape[3] * scale_factors[2],
                input_shape[4]
            ]

            assert list(result_torch.shape) == expected_shape, f"Edge case failed for {input_shape}"
            print(f"✓ Edge case passed: {input_shape} -> {expected_shape}")

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_upsample3d_program_creation()
    test_upsample3d_edge_cases()
    print("Step 5: Program factory tests PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step5_test_program_factory.py`
4. All tests MUST pass before proceeding to Step 6

### Step 6: Kernel Compilation + Test
**Test File:** `test_progress_upsample3d/step6_test_kernel_compilation.py`
**Implementation:**
- Create actual reader/writer kernel files with real logic
- Implement circular buffer setup
- Add runtime argument setup

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(20)  # Short timeout to quickly catch hangs
def test_upsample3d_kernels_compile():
    """Test that kernels compile and programs can be executed"""
    device = ttnn.open_device(device_id=0)
    try:
        # Very small test case to minimize chance of hanging
        input_tensor = torch.randn(1, 1, 1, 2, 4, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # This should now execute actual kernels (even if logic is minimal)
        result = ttnn.upsample3d(input_tensor, (2, 2, 2))

        # Verify we can convert result back to torch without crashes
        result_torch = ttnn.to_torch(result)
        expected_shape = [1, 2, 2, 4, 4]
        assert list(result_torch.shape) == expected_shape

        print(f"Kernel compilation test passed, result shape: {result_torch.shape}")

    finally:
        ttnn.close_device(device)

@pytest.mark.timeout(20)
def test_upsample3d_different_tensor_sizes():
    """Test kernel execution with different tensor sizes"""
    device = ttnn.open_device(device_id=0)
    try:
        test_shapes = [
            ([1, 1, 1, 1, 4], (2, 2, 2)),
            ([1, 1, 2, 1, 4], (1, 2, 3)),
            ([1, 2, 1, 1, 8], (2, 1, 2)),
        ]

        for input_shape, scale_factors in test_shapes:
            input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
            input_tensor = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

            result = ttnn.upsample3d(input_tensor, scale_factors)
            result_torch = ttnn.to_torch(result)

            expected_shape = [
                input_shape[0],
                input_shape[1] * scale_factors[0],
                input_shape[2] * scale_factors[1],
                input_shape[3] * scale_factors[2],
                input_shape[4]
            ]

            assert list(result_torch.shape) == expected_shape, f"Failed for {input_shape} with scales {scale_factors}"

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_upsample3d_kernels_compile()
    test_upsample3d_different_tensor_sizes()
    print("Step 6: Kernel compilation tests PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step6_test_kernel_compilation.py`
4. All tests MUST pass before proceeding to Step 7

### Step 7: Minimal Working Implementation + Test
**Test File:** `test_progress_upsample3d/step7_test_minimal_working.py`
**Implementation:**
- Implement actual upsampling logic in kernels
- Start with very simple case (small tensors, scale factor 2)

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(20)
def test_upsample3d_small_tensor():
    """Test basic upsampling functionality with small tensors"""
    device = ttnn.open_device(device_id=0)
    try:
        # Create small input tensor with known values
        input_data = torch.arange(1, 17, dtype=torch.bfloat16).reshape(1, 2, 2, 2, 2)
        input_tensor = ttnn.from_torch(input_data, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        scale_factors = (2, 2, 2)

        # Get TTNN result
        ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)
        ttnn_result_torch = ttnn.to_torch(ttnn_result)

        # Get PyTorch reference result
        # Convert NDHWC to NCDHW for PyTorch
        input_pytorch = input_data.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        pytorch_result = F.interpolate(input_pytorch, scale_factor=scale_factors, mode='nearest')
        pytorch_result = pytorch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

        # For now, just check that shapes match (content verification in step 8)
        assert ttnn_result_torch.shape == pytorch_result.shape, f"Shape mismatch: TTNN {ttnn_result_torch.shape} vs PyTorch {pytorch_result.shape}"

        print(f"Shape test passed: {ttnn_result_torch.shape}")

    finally:
        ttnn.close_device(device)

@pytest.mark.timeout(20)
def test_upsample3d_simple_scale_factor_2():
    """Test specifically with scale factor 2 in all dimensions"""
    device = ttnn.open_device(device_id=0)
    try:
        # Simple 1x1x2x2x4 tensor
        input_data = torch.randn(1, 1, 2, 2, 4, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_data, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        result = ttnn.upsample3d(input_tensor, (2, 2, 2))
        result_torch = ttnn.to_torch(result)

        expected_shape = [1, 2, 4, 4, 4]
        assert list(result_torch.shape) == expected_shape

        # Basic sanity check: result should not be all zeros
        assert torch.any(result_torch != 0), "Result tensor is all zeros"

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_upsample3d_small_tensor()
    test_upsample3d_simple_scale_factor_2()
    print("Step 7: Minimal working implementation tests PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step7_test_minimal_working.py`
4. All tests MUST pass before proceeding to Step 8

### Step 8: Full Implementation + Test
**Test File:** `test_progress_upsample3d/step8_test_full_implementation.py`
**Implementation:**
- Handle all scale factor combinations
- Optimize for larger tensors
- Add edge case handling

**Required Test Content:**
```python
import pytest
import ttnn
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(__file__))

@pytest.mark.timeout(20)  # Short timeout to quickly catch hangs
def test_upsample3d_various_scale_factors():
    """Test upsample3d with various scale factor combinations"""
    device = ttnn.open_device(device_id=0)
    try:
        test_cases = [
            ([1, 2, 3, 4, 8], (1, 1, 1)),    # No scaling
            ([1, 2, 3, 4, 8], (2, 1, 3)),    # Mixed scale factors
            ([1, 3, 2, 2, 4], (3, 4, 2)),    # Different combinations
            ([2, 1, 4, 6, 16], (1, 2, 1)),   # Batch size > 1
        ]

        for input_shape, scale_factors in test_cases:
            input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
            input_ttnn = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

            # TTNN result
            ttnn_result = ttnn.upsample3d(input_ttnn, scale_factors)
            ttnn_result_torch = ttnn.to_torch(ttnn_result)

            # PyTorch reference
            input_pytorch = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
            pytorch_result = F.interpolate(input_pytorch, scale_factor=scale_factors, mode='nearest')
            pytorch_result = pytorch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

            # Shape verification
            assert ttnn_result_torch.shape == pytorch_result.shape, f"Shape mismatch for {input_shape} with scales {scale_factors}"

            print(f"✓ Passed shape test for input {input_shape} with scales {scale_factors}")

    finally:
        ttnn.close_device(device)

@pytest.mark.timeout(20)
def test_upsample3d_coordinate_correctness():
    """Test that each output position contains the correct input value"""
    device = ttnn.open_device(device_id=0)
    try:
        # Create input with unique values to verify coordinate mapping
        input_shape = [1, 2, 2, 3, 4]  # Small but with unique pattern
        scale_factors = (2, 3, 2)

        # Create tensor where each position has a unique value based on coordinates
        input_data = torch.zeros(input_shape, dtype=torch.bfloat16)
        for n in range(input_shape[0]):
            for d in range(input_shape[1]):
                for h in range(input_shape[2]):
                    for w in range(input_shape[3]):
                        # Unique value based on coordinates
                        value = n*1000 + d*100 + h*10 + w
                        input_data[n, d, h, w, :] = value

        input_ttnn = ttnn.from_torch(input_data, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        result = ttnn.upsample3d(input_ttnn, scale_factors)
        result_torch = ttnn.to_torch(result)

        # Verify coordinate correctness using nearest neighbor logic
        output_shape = result_torch.shape
        for n in range(output_shape[0]):
            for d_out in range(output_shape[1]):
                for h_out in range(output_shape[2]):
                    for w_out in range(output_shape[3]):
                        # Calculate source coordinates for nearest neighbor
                        d_in = d_out // scale_factors[0]
                        h_in = h_out // scale_factors[1]
                        w_in = w_out // scale_factors[2]

                        expected_value = n*1000 + d_in*100 + h_in*10 + w_in
                        actual_values = result_torch[n, d_out, h_out, w_out, :]

                        # All channel values should match expected
                        for c in range(actual_values.shape[0]):
                            assert abs(actual_values[c] - expected_value) < 1e-3, \
                                f"Coordinate error at output ({n},{d_out},{h_out},{w_out},{c}): expected {expected_value}, got {actual_values[c]}"

        print("✓ Coordinate correctness verification passed")

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_upsample3d_various_scale_factors()
    test_upsample3d_coordinate_correctness()
    print("Step 8: Full implementation tests PASSED")
```

**Build & Test Protocol:**
1. `./build_metal.sh`
2. `source python_env/bin/activate`
3. `python test_progress_upsample3d/step8_test_full_implementation.py`
4. All tests MUST pass before creating final test

### FINAL TEST (Official Repository Test)
**Test File:** `tests/ttnn/unit_tests/operations/test_upsample3d.py`
**Required Content:**
```python
import pytest
import torch
import torch.nn.functional as F
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

@pytest.mark.timeout(20)
@pytest.mark.parametrize("input_shape", [
    [1, 2, 3, 4, 8],
    [1, 1, 4, 4, 16],
    [2, 3, 2, 2, 8],
])
@pytest.mark.parametrize("scale_factors", [
    (2, 2, 2),
    (1, 3, 2),
    (3, 1, 4),
    (2, 2, 1),
])
def test_upsample3d_pcc(device, input_shape, scale_factors):
    """Official PCC test for upsample3d against PyTorch reference"""

    # Create input tensor
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # TTNN result
    ttnn_result = ttnn.upsample3d(input_ttnn, scale_factors)
    ttnn_result_torch = ttnn.to_torch(ttnn_result)

    # PyTorch reference
    input_pytorch = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
    pytorch_result = F.interpolate(input_pytorch, scale_factor=scale_factors, mode='nearest')
    pytorch_result = pytorch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    # PCC comparison
    pcc_passed, pcc_message = assert_with_pcc(pytorch_result, ttnn_result_torch, pcc=0.99999)
    assert pcc_passed, f"PCC test failed: {pcc_message}"
```

### MANDATORY TESTING PROTOCOL FOR IMPLEMENTATION:
1. **CREATE TEST DIRECTORY FIRST:** `mkdir -p test_progress_upsample3d/`
2. **NO STEP SKIPPING:** Each step test file must be created and must pass before proceeding
3. **BUILD BETWEEN STEPS:** Always run `./build_metal.sh` between steps
4. **ACTIVATE ENV:** Always run `source python_env/bin/activate` before testing
5. **TIMEOUT ENFORCEMENT:** All tests have 20 second timeouts to quickly catch hangs
6. **STEP COMPLETION VERIFICATION:** Each step must print "PASSED" message before moving to next step
7. **FINAL TEST CREATION:** Only create official test after all 8 steps pass

## 5. Key Technical Considerations

**Memory Requirements**:
- Output tensor size = `N × D×scale_d × H×scale_h × W×scale_w × C`
- Can be significantly larger than input (cubic growth with scale factors)

**Coordinate Mapping**:
- Page-based indexing: input page = `n×D×H×W + d×H×W + h×W + w`
- Output page = `n×D'×H'×W' + d'×H'×W' + h'×W' + w'` where `D'=D×scale_d`, `H'=H×scale_h`, `W'=W×scale_w`
- Each input page maps to `scale_d × scale_h × scale_w` output pages

**Work Distribution**:
- Distribute `N × D × H × W` work units (pages) across available cores
- Each core processes multiple input pages and generates corresponding output pages

**Validation Requirements**:
- Verify input tensor is 5D with expected shape
- Ensure scale factors are positive integers
- Validate only "nearest" mode initially

This implementation leverages the existing 2D upsample infrastructure while extending it to handle the additional depth dimension and 3D coordinate transformations required for 3D upsampling.

## 6. Call Stack Flow and Testing Points for Upsample3D

### Complete Call Flow
```
Python Test
    ↓
ttnn.upsample3d(input_tensor, (scale_d, scale_h, scale_w))  # Python API call
    ↓
upsample3d_pybind.cpp::bind_upsample3d()                     # Python binding
    ↓
ttnn::upsample3d (registered operation)                     # C++ operation registration
    ↓
upsample3d.cpp::ExecuteUpSample3D::invoke()                  # C++ implementation
    ↓
tt::tt_metal::operation::run(UpSample3D{...})               # TT-Metal operation runner
    ↓
upsample3d_op.cpp::UpSample3D::create_program()             # Device operation
    ↓
upsample3d_program_factory_multicore_interleaved()          # Program factory
    ↓
┌─────────────────────────────────────────────┐
│ Kernel Creation & Configuration             │
├─────────────────────────────────────────────┤
│ reader_upsample3d_interleaved.cpp           │  # Reader kernel
│ writer_upsample3d_interleaved.cpp           │  # Writer kernel
│ Work distribution across cores              │
│ Circular buffer setup                       │
│ Runtime argument configuration              │
└─────────────────────────────────────────────┘
    ↓
Device Execution (Hardware)
```

### Individual Testing Points

**1. Python API Layer (test_upsample3d.py)**
```python
# Test: Python function exists and accepts correct parameters
def test_upsample3d_api_exists():
    assert hasattr(ttnn, 'upsample3d')

# Test: Parameter validation at Python level
def test_upsample3d_parameter_validation():
    # Test invalid scale factors, wrong tensor dimensions, etc.
```

**2. Python Binding Layer (upsample3d_pybind.cpp)**
```cpp
// Test: C++ binding compilation and parameter forwarding
def test_upsample3d_pybind_compilation():
    # Verify that pybind module compiles and links correctly

def test_upsample3d_argument_parsing():
    # Test that Python arguments are correctly parsed to C++ types
```

**3. Operation Registration (upsample3d.hpp)**
```cpp
// Test: Operation is properly registered in TTNN
def test_upsample3d_operation_registered():
    # Verify ttnn::upsample3d exists and is callable
```

**4. C++ Implementation Layer (upsample3d.cpp)**
```cpp
// Test: ExecuteUpSample3D::invoke logic
def test_upsample3d_invoke():
    # Test scale factor parsing from variant<int, Array3D>
    # Test memory config handling
    # Test parameter validation

// Test: Shape computation without execution
def test_upsample3d_output_shape_computation():
    input_shape = [2, 4, 8, 16, 32]  # N, D, H, W, C
    scale_factors = (2, 3, 4)        # scale_d, scale_h, scale_w
    expected_output = [2, 8, 24, 64, 32]
    # Verify compute_output_specs() returns correct shape
```

**5. Device Operation Layer (upsample3d_op.cpp)**
```cpp
// Test: UpSample3D struct and validation
def test_upsample3d_struct_creation():
    op = UpSample3D{2, 3, 4, "nearest"}
    # Test struct creation and member access

def test_upsample3d_input_validation():
    # Test validate() method with various invalid inputs:
    # - Non-5D tensors
    # - Invalid modes
    # - Zero/negative scale factors
```

**6. Program Factory Layer (upsample3d_program_factory_multicore_interleaved.cpp)**
```cpp
// Test: Program creation without execution
def test_upsample3d_program_creation():
    # Create program object, verify kernels are created
    # Test work splitting logic
    # Test circular buffer sizing

def test_upsample3d_work_distribution():
    # Test various input sizes and core counts
    # Verify work is evenly distributed
    # Test edge cases (more cores than work units)

def test_upsample3d_coordinate_mapping():
    input_coord = (1, 2, 3, 4)  # n=1, d=2, h=3, w=4
    expected_output_coords = [...]  # All corresponding output coordinates
    # Test mapping function without actual kernel execution
```

**7. Kernel Compilation Layer**
```cpp
// Test: Kernel compilation and linking
def test_upsample3d_kernels_compile():
    # Verify reader and writer kernels compile successfully
    # Test with different compile-time arguments

def test_upsample3d_kernel_argument_setup():
    # Test runtime argument calculation
    # Verify buffer addresses, work counts, start indices
```

**8. Memory and Buffer Management**
```cpp
// Test: Buffer allocation and page calculations
def test_upsample3d_buffer_sizing():
    # Test input/output buffer size calculations
    # Verify page count computations

def test_upsample3d_circular_buffer_config():
    # Test CB sizing for different input shapes
    # Verify alignment requirements
```

**9. End-to-End Functional Tests (PyTorch Reference Validation)**
```python
def test_upsample3d_small_tensor():
    # Create 5D input tensor [N, D, H, W, C]
    input_tensor = create_5d_tensor([1, 2, 2, 2, 4])
    scale_factors = (2, 2, 2)  # scale_d, scale_h, scale_w

    # TTNN result
    ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)

    # PyTorch reference (need to convert NDHWC -> NCDHW for PyTorch)
    torch_input = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
    torch_result = F.interpolate(torch_input, scale_factor=scale_factors, mode='nearest')
    torch_result = torch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    # Compare results
    assert torch.equal(ttnn_result, torch_result)

def test_upsample3d_coordinate_correctness():
    # Create input with unique values at each (d,h,w) position
    input_tensor = create_unique_valued_5d_tensor([1, 2, 3, 4, 8])
    scale_factors = (2, 3, 2)

    ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)

    # Verify each output position contains correct input value using nearest neighbor logic
    for n in range(1):
        for d_out in range(4):  # 2*2
            for h_out in range(9):  # 3*3
                for w_out in range(8):  # 4*2
                    # Calculate source coordinates for nearest neighbor
                    d_in = d_out // 2
                    h_in = h_out // 3
                    w_in = w_out // 2

                    expected_value = input_tensor[n, d_in, h_in, w_in, :]
                    actual_value = ttnn_result[n, d_out, h_out, w_out, :]
                    assert torch.equal(expected_value, actual_value)

def test_upsample3d_various_scale_factors():
    test_cases = [
        ([1, 2, 3, 4, 8], (1, 1, 1)),    # No scaling
        ([1, 2, 3, 4, 8], (2, 1, 3)),    # Mixed scale factors
        ([1, 3, 2, 2, 4], (3, 4, 2)),    # Different combinations
        ([2, 1, 4, 6, 16], (1, 2, 1)),   # Batch size > 1
    ]

    for input_shape, scale_factors in test_cases:
        input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

        # TTNN result
        ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)

        # PyTorch reference
        torch_input = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        torch_result = F.interpolate(torch_input, scale_factor=scale_factors, mode='nearest')
        torch_result = torch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

        # Validate results match
        pcc_passed, pcc_message = assert_with_pcc(torch_result, ttnn_result, pcc=0.99999)
        assert torch.equal(ttnn_result, torch_result), f"Failed for shape {input_shape}, scales {scale_factors}"
```

This layered testing approach allows verification of each component independently, making debugging much easier and ensuring each layer works correctly before moving to the next level.

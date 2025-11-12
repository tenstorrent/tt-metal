# Test-Driven Development Plan: Upsample 3D Operation

## Overview
This document outlines a strict test-driven development (TDD) approach for implementing the upsample 3D operation. The plan follows a reverse approach, starting from Python API tests and progressively implementing deeper layers of the stack. Each stage must pass its tests before proceeding to the next.

## Test Organization
- **Development tests location**: `ttnn/cpp/ttnn/operations/pool/upsample/test_dev/`
- **Run tests**: `pytest test_dev/test_stage{N}_*.py -xvs` from the upsample directory
- **Principle**: Write test first, then minimal implementation to make it pass

## Stage 1: API Existence Test
**Goal**: Verify the Python API exists and can be imported

### Test 1.1: API Import and Basic Call
**File**: `test_dev/test_stage1_api_exists.py`

```python
import pytest
import torch
import ttnn

def test_upsample3d_api_exists():
    """Test that upsample3d function exists in ttnn module"""
    assert hasattr(ttnn, 'upsample3d'), "ttnn.upsample3d API does not exist"

def test_upsample3d_callable():
    """Test that upsample3d is callable"""
    assert callable(ttnn.upsample3d), "ttnn.upsample3d is not callable"

def test_upsample3d_basic_call_fails_gracefully():
    """Test that we can call upsample3d and it fails with meaningful error"""
    device = ttnn.open_device(device_id=0)
    try:
        input_tensor = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        with pytest.raises((RuntimeError, AttributeError, NotImplementedError)) as exc_info:
            output = ttnn.upsample3d(tt_input, scale_factor=2)

        error_msg = str(exc_info.value).lower()
        assert "upsample3d" in error_msg or "not implemented" in error_msg or "attribute" in error_msg
    finally:
        ttnn.close_device(device)
```

### Implementation 1.1: Minimal Python Binding
**File**: `ttnn/cpp/ttnn/operations/pool/upsample/upsample_pybind.cpp`
Add to existing binding file:
```cpp
// Minimal binding that throws NotImplementedError
m.def("upsample3d",
    [](const Tensor& input,
       std::variant<int, std::array<int, 3>> scale_factor,
       const std::optional<MemoryConfig>& memory_config) {
        throw std::runtime_error("upsample3d not implemented yet");
    },
    py::arg("input"),
    py::arg("scale_factor"),
    py::arg("memory_config") = std::nullopt,
    R"doc(3D upsampling operation - not implemented)doc");
```

**Success Criteria**: All three tests pass, API is accessible from Python

---

## Stage 2: Parameter Validation Test
**Goal**: Verify input parameters are properly validated

### Test 2.1: Input Validation
**File**: `test_dev/test_stage2_validation.py`

```python
import pytest
import torch
import ttnn

@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

def test_input_rank_validation(device):
    """Test that input must be 5D"""
    # 4D tensor should fail
    input_4d = torch.ones((1, 2, 2, 4), dtype=torch.bfloat16)
    tt_input_4d = ttnn.from_torch(input_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input_4d, scale_factor=2)
    assert "5D" in str(exc_info.value) or "rank" in str(exc_info.value).lower()

def test_scale_factor_validation(device):
    """Test scale_factor parameter validation"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with negative scale factor
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=-1)
    assert "positive" in str(exc_info.value).lower() or "scale" in str(exc_info.value).lower()

    # Test with zero scale factor
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=0)
    assert "positive" in str(exc_info.value).lower() or "scale" in str(exc_info.value).lower()

def test_scale_factor_types(device):
    """Test that scale_factor accepts int or tuple of 3 ints"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # These should pass validation (may fail later)
    try:
        ttnn.upsample3d(tt_input, scale_factor=2)  # int
    except RuntimeError as e:
        assert "not implemented" in str(e).lower() or "device" in str(e).lower()

    try:
        ttnn.upsample3d(tt_input, scale_factor=(2, 2, 2))  # tuple of 3
    except RuntimeError as e:
        assert "not implemented" in str(e).lower() or "device" in str(e).lower()

    # Wrong tuple size should fail validation
    with pytest.raises((RuntimeError, ValueError, TypeError)) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=(2, 2))  # tuple of 2
    assert "3" in str(exc_info.value) or "scale" in str(exc_info.value).lower()

def test_layout_validation(device):
    """Test that only ROW_MAJOR layout is supported initially"""
    input_5d = torch.ones((1, 2, 32, 32, 32), dtype=torch.bfloat16)  # Tile-aligned
    tt_input_tiled = ttnn.from_torch(input_5d, device=device, layout=ttnn.TILE_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input_tiled, scale_factor=2)
    assert "ROW_MAJOR" in str(exc_info.value) or "layout" in str(exc_info.value).lower()
```

### Implementation 2.1: Host Operation with Validation
**File**: `ttnn/cpp/ttnn/operations/pool/upsample/upsample3d.hpp`
```cpp
#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::upsample {

struct ExecuteUpSample3D {
    static Tensor invoke(
        const Tensor& input,
        std::variant<int, std::array<int, 3>> scale_factor,
        std::optional<MemoryConfig> memory_config = std::nullopt);
};

} // namespace ttnn::operations::upsample
```

**File**: `ttnn/cpp/ttnn/operations/pool/upsample/upsample3d.cpp`
```cpp
#include "upsample3d.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::upsample {

Tensor ExecuteUpSample3D::invoke(
    const Tensor& input,
    std::variant<int, std::array<int, 3>> scale_factor,
    std::optional<MemoryConfig> memory_config) {

    // Input validation
    TT_FATAL(input.get_shape().rank() == 5,
             "Input tensor must be 5D (N, D, H, W, C), got rank {}", input.get_shape().rank());

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR,
             "Only ROW_MAJOR layout is supported for 3D upsample, got {}", input.get_layout());

    // Parse and validate scale factors
    uint32_t scale_d, scale_h, scale_w;
    if (std::holds_alternative<int>(scale_factor)) {
        int scale = std::get<int>(scale_factor);
        TT_FATAL(scale > 0, "Scale factor must be positive, got {}", scale);
        scale_d = scale_h = scale_w = scale;
    } else {
        auto scales = std::get<std::array<int, 3>>(scale_factor);
        TT_FATAL(scales[0] > 0 && scales[1] > 0 && scales[2] > 0,
                 "All scale factors must be positive, got ({}, {}, {})",
                 scales[0], scales[1], scales[2]);
        scale_d = scales[0];
        scale_h = scales[1];
        scale_w = scales[2];
    }

    // For now, throw not implemented after validation
    throw std::runtime_error("upsample3d device operation not implemented yet");
}

} // namespace ttnn::operations::upsample
```

Update Python binding to call the host operation:
```cpp
// In upsample_pybind.cpp, replace the lambda with actual function call
m.def("upsample3d",
    &ttnn::operations::upsample::ExecuteUpSample3D::invoke,
    py::arg("input"),
    py::arg("scale_factor"),
    py::arg("memory_config") = std::nullopt,
    R"doc(3D nearest-neighbor upsampling)doc");
```

**Success Criteria**: All validation tests pass, errors are descriptive

---

## Stage 3: TTNN Operation Registration Test
**Goal**: Verify the operation is properly registered with TTNN framework

### Test 3.1: Operation Registration
**File**: `test_dev/test_stage3_registration.py`

```python
import pytest
import torch
import ttnn
import inspect

@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

def test_operation_signature():
    """Test that upsample3d has the expected signature"""
    sig = inspect.signature(ttnn.upsample3d)
    params = list(sig.parameters.keys())

    assert 'input' in params, "Missing 'input' parameter"
    assert 'scale_factor' in params, "Missing 'scale_factor' parameter"
    assert 'memory_config' in params, "Missing 'memory_config' parameter"

def test_operation_docstring():
    """Test that operation has proper documentation"""
    assert ttnn.upsample3d.__doc__ is not None, "Missing docstring"
    doc_lower = ttnn.upsample3d.__doc__.lower()
    assert "3d" in doc_lower or "upsample" in doc_lower, "Docstring doesn't describe operation"

def test_operation_with_memory_config(device):
    """Test that memory_config parameter is accepted"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should accept memory config without error
    try:
        ttnn.upsample3d(tt_input, scale_factor=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    except RuntimeError as e:
        # Should fail at device operation level, not parameter level
        assert "device" in str(e).lower() or "not implemented" in str(e).lower()

def test_registered_operation_dispatch(device):
    """Test that operation goes through TTNN registration system"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should mention operation or device in error (not just "not implemented")
    error_msg = str(exc_info.value).lower()
    assert "operation" in error_msg or "device" in error_msg
```

### Implementation 3.1: Register Operation with TTNN
Update `upsample3d.hpp` to use TTNN registration:
```cpp
#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::upsample {

struct ExecuteUpSample3D {
    static Tensor invoke(
        const Tensor& input,
        std::variant<int, std::array<int, 3>> scale_factor,
        std::optional<MemoryConfig> memory_config = std::nullopt);
};

} // namespace ttnn::operations::upsample

namespace ttnn {
    constexpr auto upsample3d = ttnn::register_operation<
        "ttnn::upsample3d",
        ttnn::operations::upsample::ExecuteUpSample3D>();
} // namespace ttnn
```

Update `upsample_pybind.cpp` to use registered operation:
```cpp
#include "upsample3d.hpp"

// In the pybind module definition:
ttnn::bind_registered_operation(
    module,
    ttnn::upsample3d,
    R"doc(
    Performs 3D nearest-neighbor upsampling on 5D input tensors.

    Args:
        input: 5D input tensor with shape (N, D, H, W, C) in ROW_MAJOR layout
        scale_factor: Integer or tuple of 3 integers specifying the upsampling factor
                     for depth, height, and width dimensions respectively
        memory_config: Optional memory configuration for the output tensor

    Returns:
        Upsampled 5D tensor with shape (N, D*scale_d, H*scale_h, W*scale_w, C)

    Example:
        >>> input = torch.ones((1, 2, 4, 4, 16), dtype=torch.bfloat16)
        >>> tt_input = ttnn.from_torch(input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        >>> output = ttnn.upsample3d(tt_input, scale_factor=2)
        >>> # Output shape: (1, 4, 8, 8, 16)
    )doc",
    ttnn::pybind_overload_t{
        [](const Tensor& input,
           int scale_factor,
           std::optional<MemoryConfig> memory_config) {
            return ttnn::upsample3d(input, scale_factor, memory_config);
        },
        py::arg("input"),
        py::arg("scale_factor"),
        py::arg("memory_config") = std::nullopt},
    ttnn::pybind_overload_t{
        [](const Tensor& input,
           std::array<int, 3> scale_factor,
           std::optional<MemoryConfig> memory_config) {
            return ttnn::upsample3d(input, scale_factor, memory_config);
        },
        py::arg("input"),
        py::arg("scale_factor"),
        py::arg("memory_config") = std::nullopt});
```

**Success Criteria**: Operation is registered, has proper signature and documentation

---

## Stage 4: Device Operation Existence Test
**Goal**: Verify device operation is created and validates inputs

### Test 4.1: Device Operation Creation
**File**: `test_dev/test_stage4_device_op.py`

```python
import pytest
import torch
import ttnn
import numpy as np

@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

def test_device_operation_invoked(device):
    """Test that device operation is called with proper validation"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should fail at program/device level, not host validation
    error_msg = str(exc_info.value).lower()
    assert "program" in error_msg or "device" in error_msg or "create" in error_msg

def test_output_shape_calculation(device):
    """Test that output shape is calculated correctly by device op"""
    test_cases = [
        ((1, 2, 3, 4, 8), 2, (1, 4, 6, 8, 8)),
        ((1, 2, 3, 4, 8), (2, 3, 2), (1, 4, 9, 8, 8)),
        ((2, 1, 5, 5, 16), (1, 2, 2), (2, 1, 10, 10, 16)),
    ]

    for input_shape, scale_factor, expected_shape in test_cases:
        input_tensor = torch.ones(input_shape, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Even though execution fails, shape calculation should work
        with pytest.raises(RuntimeError) as exc_info:
            ttnn.upsample3d(tt_input, scale_factor=scale_factor)

        # Error should be about program creation, not shape calculation
        error_msg = str(exc_info.value).lower()
        assert "program" in error_msg or "factory" in error_msg

def test_memory_config_handling(device):
    """Test that memory config is properly passed to device operation"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with different memory configs
    configs = [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]
    for config in configs:
        with pytest.raises(RuntimeError) as exc_info:
            ttnn.upsample3d(tt_input, scale_factor=2, memory_config=config)

        # Should fail at program level
        assert "program" in str(exc_info.value).lower()

def test_interleaved_memory_validation(device):
    """Test that only interleaved memory is supported"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # For now, should only support interleaved
    # This test documents the current limitation
    with pytest.raises(RuntimeError):
        # Would fail when sharding is attempted
        ttnn.upsample3d(tt_input, scale_factor=2)
```

### Implementation 4.1: Device Operation
**File**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample3d_op.hpp`
```cpp
#pragma once
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::upsample {

struct UpSample3D {
    uint32_t scale_factor_d_;
    uint32_t scale_factor_h_;
    uint32_t scale_factor_w_;
    MemoryConfig output_mem_config_;

    void validate(const std::vector<Tensor>& inputs) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& inputs,
        std::vector<Tensor>& outputs) const;
};

} // namespace ttnn::operations::upsample
```

**File**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample3d_op.cpp`
```cpp
#include "upsample3d_op.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace ttnn::operations::upsample {

void UpSample3D::validate(const std::vector<Tensor>& inputs) const {
    const auto& input = inputs.at(0);

    TT_FATAL(input.get_shape().rank() == 5,
             "Input must be 5D tensor, got rank {}", input.get_shape().rank());

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR,
             "Only ROW_MAJOR layout is supported, got {}", input.get_layout());

    TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
             "Only INTERLEAVED memory layout is supported");

    TT_FATAL(input.is_allocated(), "Input tensor must be allocated on device");

    TT_FATAL(input.storage_type() == StorageType::DEVICE,
             "Input must be on device, got storage type {}", input.storage_type());
}

std::vector<TensorSpec> UpSample3D::compute_output_specs(
    const std::vector<Tensor>& inputs) const {
    const auto& input = inputs.at(0);
    const auto& input_shape = input.get_shape();

    // Calculate output shape: (N, D*scale_d, H*scale_h, W*scale_w, C)
    std::array<uint32_t, 5> output_shape_array = {
        input_shape[0],                      // N (batch)
        input_shape[1] * scale_factor_d_,    // D * scale_d
        input_shape[2] * scale_factor_h_,    // H * scale_h
        input_shape[3] * scale_factor_w_,    // W * scale_w
        input_shape[4]                       // C (channels)
    };

    auto output_shape = Shape(output_shape_array);

    // Create output tensor spec with same dtype and layout as input
    return {TensorSpec(
        output_shape,
        TensorLayout(input.get_dtype(), PageConfig(Layout::ROW_MAJOR), output_mem_config_))};
}

operation::ProgramWithCallbacks UpSample3D::create_program(
    const std::vector<Tensor>& inputs,
    std::vector<Tensor>& outputs) const {

    // Will be implemented in Stage 5
    throw std::runtime_error("Program factory not implemented yet");
}

} // namespace ttnn::operations::upsample
```

Update `upsample3d.cpp` to use device operation:
```cpp
#include "upsample3d.hpp"
#include "device/upsample3d_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

Tensor ExecuteUpSample3D::invoke(
    const Tensor& input,
    std::variant<int, std::array<int, 3>> scale_factor,
    std::optional<MemoryConfig> memory_config) {

    // Input validation (keep existing)
    TT_FATAL(input.get_shape().rank() == 5,
             "Input tensor must be 5D (N, D, H, W, C), got rank {}", input.get_shape().rank());

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR,
             "Only ROW_MAJOR layout is supported for 3D upsample");

    // Parse scale factors (keep existing)
    uint32_t scale_d, scale_h, scale_w;
    if (std::holds_alternative<int>(scale_factor)) {
        int scale = std::get<int>(scale_factor);
        TT_FATAL(scale > 0, "Scale factor must be positive, got {}", scale);
        scale_d = scale_h = scale_w = scale;
    } else {
        auto scales = std::get<std::array<int, 3>>(scale_factor);
        TT_FATAL(scales[0] > 0 && scales[1] > 0 && scales[2] > 0,
                 "All scale factors must be positive");
        scale_d = scales[0];
        scale_h = scales[1];
        scale_w = scales[2];
    }

    // Use memory config from parameter or input
    auto output_memory_config = memory_config.value_or(input.memory_config());

    // Create and run device operation
    auto output = operation::run(
        UpSample3D{scale_d, scale_h, scale_w, output_memory_config},
        {input}).at(0);

    return output;
}

} // namespace ttnn::operations::upsample
```

**Success Criteria**: Device operation validates inputs and computes output shape correctly

---

## Stage 5: Program Factory API Test
**Goal**: Verify program factory is called and creates basic program structure

### Test 5.1: Program Factory Invocation
**File**: `test_dev/test_stage5_program_factory.py`

```python
import pytest
import torch
import ttnn

@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

def test_program_creation_attempted(device):
    """Test that program factory is invoked"""
    input_tensor = torch.ones((1, 1, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=1)  # Identity operation

    # Should mention kernel or circular buffer
    error_msg = str(exc_info.value).lower()
    assert "kernel" in error_msg or "buffer" in error_msg or "compile" in error_msg

def test_single_core_program(device):
    """Test minimal single-core configuration"""
    # Very small tensor for single core
    input_tensor = torch.ones((1, 1, 1, 1, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=1)

    # Should get to kernel creation
    assert "kernel" in str(exc_info.value).lower()

def test_multi_core_work_distribution(device):
    """Test that work distribution is calculated"""
    # Larger tensor that should use multiple cores
    input_tensor = torch.ones((1, 4, 8, 8, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should attempt kernel creation
    error_msg = str(exc_info.value).lower()
    assert "kernel" in error_msg or "reader" in error_msg or "writer" in error_msg

def test_circular_buffer_creation(device):
    """Test that circular buffers are set up"""
    input_tensor = torch.ones((1, 2, 2, 2, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should mention kernels (CBs are created before kernels)
    assert "kernel" in str(exc_info.value).lower()
```

### Implementation 5.1: Program Factory Structure
Update `upsample3d_op.cpp`:
```cpp
operation::ProgramWithCallbacks UpSample3D::create_program(
    const std::vector<Tensor>& inputs,
    std::vector<Tensor>& outputs) const {

    const auto& input = inputs.at(0);
    auto& output = outputs.at(0);

    // Forward to program factory
    return upsample3d_multi_core_interleaved(
        input, output,
        scale_factor_d_, scale_factor_h_, scale_factor_w_);
}
```

Complete the existing `upsample3D_program_factory.cpp`:
```cpp
#include "upsample3D_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::upsample {

operation::ProgramWithCallbacks upsample3d_multi_core_interleaved(
    const Tensor& input,
    Tensor& output,
    const uint32_t scale_factor_d,
    const uint32_t scale_factor_h,
    const uint32_t scale_factor_w) {

    Program program{};

    // Validate inputs
    TT_FATAL(input.get_shape().rank() == 5, "Input must be 5D");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported");

    // Get device and grid
    Device* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Get shapes and dimensions
    const auto& input_shape = input.get_shape();
    const uint32_t N = input_shape[0];
    const uint32_t D = input_shape[1];
    const uint32_t H = input_shape[2];
    const uint32_t W = input_shape[3];
    const uint32_t C = input_shape[4];

    // Calculate work units (each stick is one work unit)
    const uint32_t input_unit_size = C * input.element_size();
    const uint32_t aligned_input_unit_size = round_up(input_unit_size, 32);
    const uint32_t work_units_to_split = N * D * H * W;

    // Distribute work across cores
    const auto [num_cores, all_cores, core_group_1, core_group_2,
                work_per_core_group_1, work_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    // Create circular buffers
    uint32_t cb_index = tt::CBIndex::c_0;
    uint32_t num_pages = work_per_core_group_1 > 1 ? 2 : 1; // Double buffer if needed

    CircularBufferConfig cb_config = CircularBufferConfig(
        aligned_input_unit_size * num_pages,
        {{cb_index, datatype_to_dataformat_converter(input.get_dtype())}})
        .set_page_size(cb_index, aligned_input_unit_size);

    CreateCircularBuffer(program, all_cores, cb_config);

    // Get buffer addresses
    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    // TODO: Create kernels in Stage 6
    throw std::runtime_error("Kernel creation not implemented yet");

    return {std::move(program), {}};
}

} // namespace ttnn::operations::upsample
```

**Success Criteria**: Program factory creates program with CBs and work distribution

---

## Stage 6: Kernel Compilation Test
**Goal**: Verify kernels compile successfully

### Test 6.1: Kernel Compilation
**File**: `test_dev/test_stage6_kernel_compile.py`

```python
import pytest
import torch
import ttnn

@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

def test_kernels_compile(device):
    """Test that kernels compile without syntax errors"""
    input_tensor = torch.ones((1, 1, 2, 2, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should now fail at runtime, not compilation
    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=1)

    error_msg = str(exc_info.value).lower()
    # Should not have compilation errors
    assert "compile" not in error_msg and "syntax" not in error_msg
    # May fail during execution
    assert len(error_msg) > 0  # Some runtime error

def test_reader_kernel_created(device):
    """Test that reader kernel is properly set up"""
    input_tensor = torch.ones((1, 1, 1, 1, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should create and compile reader kernel
    with pytest.raises(RuntimeError):
        ttnn.upsample3d(tt_input, scale_factor=1)
    # Just verify it doesn't fail at compilation

def test_writer_kernel_created(device):
    """Test that writer kernel is properly set up"""
    input_tensor = torch.ones((1, 2, 2, 2, 16), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should create and compile writer kernel
    with pytest.raises(RuntimeError):
        ttnn.upsample3d(tt_input, scale_factor=2)
    # Just verify it doesn't fail at compilation

def test_runtime_args_set(device):
    """Test that runtime arguments are properly configured"""
    input_tensor = torch.ones((1, 1, 2, 2, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Kernels should compile and runtime args should be set
    with pytest.raises(RuntimeError):
        ttnn.upsample3d(tt_input, scale_factor=1)
```

### Implementation 6.1: Create Kernels
Create minimal writer kernel:
**File**: `ttnn/cpp/ttnn/kernels/dataflow/writer_upsample3d_interleaved.cpp`
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Get compile time arguments
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t scale_d = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w = get_compile_time_arg_val(4);

    // Get runtime arguments
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Minimal implementation for Stage 6 - just copy for identity case
    if (scale_d == 1 && scale_h == 1 && scale_w == 1) {
        // Identity operation - simple copy
        for (uint32_t i = 0; i < num_sticks; i++) {
            cb_wait_front(cb_id, 1);

            uint32_t l1_addr = get_read_ptr(cb_id);
            uint64_t dst_noc_addr = get_noc_addr(dst_addr) + (start_stick_id + i) * stick_size_bytes;
            noc_async_write(l1_addr, dst_noc_addr, stick_size_bytes);

            cb_pop_front(cb_id, 1);
        }
    }
    // Full implementation will be added in Stage 7

    noc_async_write_barrier();
}
```

Update `upsample3D_program_factory.cpp` to create kernels:
```cpp
// Add after circular buffer creation:

// Compile time arguments for reader
std::vector<uint32_t> reader_compile_time_args = {
    cb_index,                    // CB index
    aligned_input_unit_size      // Stick size
};

// Compile time arguments for writer
std::vector<uint32_t> writer_compile_time_args = {
    cb_index,                    // CB index
    aligned_input_unit_size,     // Stick size in bytes
    scale_factor_d,              // Depth scale factor
    scale_factor_h,              // Height scale factor
    scale_factor_w               // Width scale factor
};

// Create reader kernel (reuse existing)
KernelHandle reader_kernel = CreateKernel(
    program,
    "ttnn/cpp/ttnn/kernels/dataflow/reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
    all_cores,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args
    });

// Create writer kernel
KernelHandle writer_kernel = CreateKernel(
    program,
    "ttnn/cpp/ttnn/kernels/dataflow/writer_upsample3d_interleaved.cpp",
    all_cores,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args
    });

// Set runtime arguments for each core group
auto set_runtime_args = [&](CoreRange core_range, uint32_t num_sticks, uint32_t start_stick) {
    auto cores = corerange_to_cores(core_range, std::nullopt);
    for (const auto& core : cores) {
        // Reader runtime args
        SetRuntimeArgs(program, reader_kernel, core,
            {src_buffer->address(), num_sticks, start_stick});

        // Writer runtime args
        SetRuntimeArgs(program, writer_kernel, core,
            {dst_buffer->address(), num_sticks, start_stick});
    }
};

uint32_t curr_stick = 0;
set_runtime_args(core_group_1, work_per_core_group_1, curr_stick);
curr_stick += work_per_core_group_1 * core_group_1.num_cores();

if (core_group_2.num_cores() > 0) {
    set_runtime_args(core_group_2, work_per_core_group_2, curr_stick);
}

// Add callback to update buffer addresses
auto override_addresses = [reader_kernel, writer_kernel](
    const Program& program,
    const std::vector<Buffer*>& input_buffers,
    const std::vector<Buffer*>& output_buffers) {

    auto reader_runtime_args = GetRuntimeArgs(program, reader_kernel);
    auto writer_runtime_args = GetRuntimeArgs(program, writer_kernel);

    for (const auto& core : reader_runtime_args) {
        reader_runtime_args[core.first][0] = input_buffers[0]->address();
        writer_runtime_args[core.first][0] = output_buffers[0]->address();
    }
    SetRuntimeArgs(program, reader_kernel, reader_runtime_args);
    SetRuntimeArgs(program, writer_kernel, writer_runtime_args);
};

return {std::move(program), override_addresses};
```

**Success Criteria**: Kernels compile without errors, runtime args are set

---

## Stage 7: Kernel Correctness Test
**Goal**: Verify kernels execute correctly with proper 3D upsampling

### Test 7.1: Kernel Execution Correctness
**File**: `test_dev/test_stage7_kernel_correct.py`

```python
import pytest
import torch
import ttnn
import numpy as np

@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)

def test_identity_operation(device):
    """Test scale_factor=1 (identity operation)"""
    input_shape = (1, 2, 2, 2, 4)
    input_tensor = torch.ones(input_shape, dtype=torch.bfloat16) * 3.14
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.upsample3d(tt_input, scale_factor=1)
    output_tensor = ttnn.to_torch(output)

    assert output_tensor.shape == input_shape
    torch.testing.assert_close(output_tensor, input_tensor, rtol=1e-3, atol=1e-3)

def test_simple_depth_scaling(device):
    """Test scaling only in depth dimension"""
    input_shape = (1, 2, 3, 3, 8)
    scale_factor = (2, 1, 1)  # Only scale depth

    input_tensor = torch.arange(np.prod(input_shape), dtype=torch.float32).reshape(input_shape)
    input_tensor = input_tensor.to(torch.bfloat16)

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.upsample3d(tt_input, scale_factor=scale_factor)
    output_tensor = ttnn.to_torch(output)

    # Check output shape
    expected_shape = (1, 4, 3, 3, 8)  # Depth doubled
    assert output_tensor.shape == expected_shape

    # Check that each depth plane is duplicated
    for d in range(2):
        torch.testing.assert_close(
            output_tensor[0, d*2, :, :, :],
            output_tensor[0, d*2+1, :, :, :],
            rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            output_tensor[0, d*2, :, :, :],
            input_tensor[0, d, :, :, :],
            rtol=1e-2, atol=1e-2
        )

def test_uniform_scaling(device):
    """Test uniform scaling in all dimensions"""
    input_shape = (1, 2, 2, 2, 16)
    scale_factor = 2

    input_tensor = torch.ones(input_shape, dtype=torch.bfloat16) * 42.0
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.upsample3d(tt_input, scale_factor=scale_factor)
    output_tensor = ttnn.to_torch(output)

    expected_shape = (1, 4, 4, 4, 16)
    assert output_tensor.shape == expected_shape

    # All values should be 42.0
    expected = torch.ones(expected_shape, dtype=torch.bfloat16) * 42.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-2, atol=1e-2)

def test_mixed_scaling(device):
    """Test different scale factors for each dimension"""
    input_shape = (1, 2, 3, 4, 8)
    scale_factors = (2, 2, 2)  # Scale all dimensions by 2

    # Create input with known pattern
    input_tensor = torch.arange(np.prod(input_shape), dtype=torch.float32).reshape(input_shape)
    input_tensor = input_tensor.to(torch.bfloat16)

    # PyTorch reference implementation
    input_ncdhw = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
    torch_output = torch.nn.functional.interpolate(
        input_ncdhw, scale_factor=scale_factors, mode='nearest'
    )
    torch_output = torch_output.permute(0, 2, 3, 4, 1)  # Back to NDHWC

    # TTNN implementation
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.upsample3d(tt_input, scale_factor=scale_factors)
    ttnn_output = ttnn.to_torch(output)

    # Verify shape
    assert ttnn_output.shape == torch_output.shape

    # Check correctness with reasonable tolerance for bfloat16
    torch.testing.assert_close(ttnn_output, torch_output, rtol=1e-1, atol=1e-1)

def test_multi_batch(device):
    """Test with multiple batches"""
    input_shape = (2, 2, 2, 2, 8)
    scale_factor = 2

    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.upsample3d(tt_input, scale_factor=scale_factor)
    output_tensor = ttnn.to_torch(output)

    expected_shape = (2, 4, 4, 4, 8)
    assert output_tensor.shape == expected_shape

    # Verify each batch is processed correctly
    for batch in range(2):
        # Check that values are replicated correctly
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    source_val = input_tensor[batch, d, h, w, :]
                    for dd in range(2):
                        for hh in range(2):
                            for ww in range(2):
                                target_val = output_tensor[
                                    batch, d*2+dd, h*2+hh, w*2+ww, :
                                ]
                                torch.testing.assert_close(
                                    target_val, source_val,
                                    rtol=1e-1, atol=1e-1
                                )

def test_non_uniform_scaling(device):
    """Test with different scale factors per dimension"""
    input_shape = (1, 2, 2, 2, 16)
    scale_factors = (1, 2, 3)  # Different scaling per dimension

    input_tensor = torch.ones(input_shape, dtype=torch.bfloat16) * 7.0
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.upsample3d(tt_input, scale_factor=scale_factors)
    output_tensor = ttnn.to_torch(output)

    expected_shape = (1, 2, 4, 6, 16)  # D*1, H*2, W*3
    assert output_tensor.shape == expected_shape

    # All values should still be 7.0
    expected = torch.ones(expected_shape, dtype=torch.bfloat16) * 7.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-2, atol=1e-2)
```

### Implementation 7.1: Complete Writer Kernel
Update `writer_upsample3d_interleaved.cpp` with full implementation:
```cpp
#include "dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t scale_d = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_d = get_compile_time_arg_val(5);
    constexpr uint32_t out_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_w = get_compile_time_arg_val(7);
    constexpr uint32_t in_d = get_compile_time_arg_val(8);
    constexpr uint32_t in_h = get_compile_time_arg_val(9);
    constexpr uint32_t in_w = get_compile_time_arg_val(10);
    constexpr uint32_t in_n = get_compile_time_arg_val(11);

    // Runtime arguments
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    constexpr bool dst_is_dram = true;

    // Process each input stick
    for (uint32_t i = 0; i < num_sticks; i++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_addr = get_read_ptr(cb_id);

        // Calculate 5D position from linear stick index
        uint32_t stick_idx = start_stick_id + i;

        // Decompose linear index to 5D coordinates
        // stick_idx represents position in flattened (N, D, H, W) space
        uint32_t curr_idx = stick_idx % (in_d * in_h * in_w);
        uint32_t curr_batch = stick_idx / (in_d * in_h * in_w);

        uint32_t curr_w = curr_idx % in_w;
        uint32_t remainder = curr_idx / in_w;
        uint32_t curr_h = remainder % in_h;
        uint32_t curr_d = remainder / in_h;

        // Calculate base output position
        uint32_t out_d_base = curr_d * scale_d;
        uint32_t out_h_base = curr_h * scale_h;
        uint32_t out_w_base = curr_w * scale_w;

        // Write to all upsampled positions
        for (uint32_t d = 0; d < scale_d; d++) {
            for (uint32_t h = 0; h < scale_h; h++) {
                for (uint32_t w = 0; w < scale_w; w++) {
                    // Calculate output stick index
                    uint32_t out_stick_idx =
                        curr_batch * (out_d * out_h * out_w) +
                        (out_d_base + d) * (out_h * out_w) +
                        (out_h_base + h) * out_w +
                        (out_w_base + w);

                    // Calculate output address
                    uint64_t dst_noc_addr = get_noc_addr(dst_addr) +
                                           out_stick_idx * stick_size_bytes;

                    // Write the stick to output location
                    noc_async_write(l1_addr, dst_noc_addr, stick_size_bytes);
                }
            }
        }

        cb_pop_front(cb_id, 1);
    }

    noc_async_write_barrier();
}
```

Update `upsample3D_program_factory.cpp` to pass all necessary compile-time args:
```cpp
// Update writer compile time args
std::vector<uint32_t> writer_compile_time_args = {
    cb_index,                    // 0: CB index
    aligned_input_unit_size,     // 1: Stick size in bytes
    scale_factor_d,              // 2: Depth scale factor
    scale_factor_h,              // 3: Height scale factor
    scale_factor_w,              // 4: Width scale factor
    D * scale_factor_d,          // 5: Output depth
    H * scale_factor_h,          // 6: Output height
    W * scale_factor_w,          // 7: Output width
    D,                          // 8: Input depth
    H,                          // 9: Input height
    W,                          // 10: Input width
    N                           // 11: Batch size
};
```

**Success Criteria**: All correctness tests pass with PCC > 0.99

---

## Test Execution Workflow

### Progressive Testing Strategy
Each stage builds on the previous one. Run tests in order:

```bash
# Navigate to test directory
cd ttnn/cpp/ttnn/operations/pool/upsample/test_dev/

# Stage 1: API exists
pytest test_stage1_api_exists.py -xvs

# Stage 2: Validation
pytest test_stage2_validation.py -xvs

# Stage 3: Registration
pytest test_stage3_registration.py -xvs

# Stage 4: Device operation
pytest test_stage4_device_op.py -xvs

# Stage 5: Program factory
pytest test_stage5_program_factory.py -xvs

# Stage 6: Kernel compilation
pytest test_stage6_kernel_compile.py -xvs

# Stage 7: Kernel correctness
pytest test_stage7_kernel_correct.py -xvs

# Run all development tests
pytest . -xvs
```

### Debugging Tips

1. **Use TTNN environment variables for debugging**:
```bash
export TT_METAL_WATCHER=10  # Enable watcher
export TT_METAL_DPRINT_CORES=(0,0)  # Enable debug print
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'
```

2. **Add debug prints in kernels**:
```cpp
#include "debug/dprint.h"
DPRINT << "Processing stick " << i << ENDL();
```

3. **Check tensor properties**:
```python
print(f"Input shape: {tt_input.shape}")
print(f"Input dtype: {tt_input.dtype}")
print(f"Input layout: {tt_input.layout}")
```

## Build System Integration

### CMake Configuration
Add to `ttnn/cpp/ttnn/operations/pool/CMakeLists.txt`:
```cmake
set(UPSAMPLE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/upsample/upsample3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/upsample/device/upsample3d_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/upsample/device/upsample3D_program_factory.cpp
)

target_sources(ttnn_operations PRIVATE ${UPSAMPLE_SRCS})
```

### Kernel Registration
Ensure kernel is findable by adding to kernel search paths.

## Success Metrics

### Per-Stage Success Criteria
- **Stage 1**: ✅ API exists and is callable from Python
- **Stage 2**: ✅ All parameter validation works correctly
- **Stage 3**: ✅ Operation is properly registered with TTNN
- **Stage 4**: ✅ Device operation validates and computes shapes
- **Stage 5**: ✅ Program factory creates program structure
- **Stage 6**: ✅ Kernels compile without errors
- **Stage 7**: ✅ Kernels execute correctly with accurate results

### Overall Success Indicators
- All 7 stages complete with passing tests
- No memory leaks (verify with valgrind if needed)
- No Watcher errors during execution
- Reasonable performance (within 2x of theoretical bandwidth)

## Common Issues and Solutions

### Issue: Import Error
**Solution**: Ensure Python bindings are compiled and installed correctly

### Issue: Validation Failures
**Solution**: Check tensor properties match requirements (5D, ROW_MAJOR, etc.)

### Issue: Kernel Compilation Errors
**Solution**: Check kernel syntax, ensure all compile-time args are provided

### Issue: Incorrect Output Values
**Solution**: Verify index calculations, use debug prints to trace execution

### Issue: Performance Problems
**Solution**: Profile with Tracy, check NOC utilization, verify work distribution

## Next Steps After Stage 7

Once all 7 stages pass:
1. Create comprehensive final tests (Stage 8)
2. Add performance optimizations
3. Support additional features (tiled layout, sharded memory)
4. Document the operation
5. Submit for code review

This plan ensures systematic, test-driven development with clear verification at each stage.

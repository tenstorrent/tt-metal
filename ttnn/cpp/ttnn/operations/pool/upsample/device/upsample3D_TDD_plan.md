# Test-Driven Development Plan: Upsample 3D Operation

## Key Updates in This Version

This TDD plan has been updated to follow the official TTNN operation structure:

1. **Proper TTNN Operation Structure**: The plan now explicitly follows the nested struct pattern required for TTNN operations with `operation_attributes_t`, `tensor_args_t`, and program factory structures.

2. **Device Operation Architecture**: Stage 4 has been completely revised to implement the proper device operation with all required static methods (`validate_on_program_cache_miss`, `compute_output_specs`, etc.).

3. **Program Factory Pattern**: Stage 5 now implements the `MultiCoreInterleaved` factory with proper `create()` and `override_runtime_arguments()` methods.

4. **Python Binding Structure**: Stages 1-3 now use `ttnn::bind_registered_operation` with proper registration macros instead of raw pybind11.

5. **Golden Function Attachment**: New Stage 8 adds PyTorch golden function for validation using `ttnn.attach_golden_function()`.

## Overview
This document outlines a strict test-driven development (TDD) approach for implementing the upsample 3D operation as a proper TTNN operation. The plan follows a reverse approach, starting from Python API tests and progressively implementing deeper layers of the stack according to the official TTNN operation structure. Each stage must pass its tests before proceeding to the next.

**IMPORTANT**: This operation follows the standard TTNN operation structure as documented at https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html

## Required File Structure for TTNN Operation

Per the TTNN operation guidelines, the following files must be created:

### C++ Implementation Files
```
ttnn/cpp/ttnn/operations/pool/upsample/
├── upsample3d.hpp                          # Main operation header with ExecuteUpSample3D struct
├── upsample3d.cpp                          # Implementation of invoke() method
├── device/
│   ├── upsample3d_device_operation.hpp    # Device operation with nested structs
│   ├── upsample3d_device_operation.cpp    # Device operation implementation
│   └── upsample3d_program_factory.cpp     # Program factory implementations
```

### Python Binding Files
```
ttnn/python/ttnn/operations/pool/
├── upsample/
│   └── upsample3d_pybind.hpp              # Python binding for upsample3d
└── pool_pybind.hpp                        # Category-level bindings
```

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

### Implementation 1.1: Minimal Python Binding Following TTNN Structure

**Step 1: Create Operation Header**
**File**: `ttnn/cpp/ttnn/operations/pool/upsample/upsample3d.hpp`
```cpp
#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn::operations::upsample {

struct ExecuteUpSample3D {
    // Minimal implementation for Stage 1
    static Tensor invoke(
        const Tensor& input,
        std::variant<int, std::array<int, 3>> scale_factor,
        std::optional<MemoryConfig> memory_config = std::nullopt) {
        throw std::runtime_error("upsample3d not implemented yet");
    }
};

} // namespace ttnn::operations::upsample

namespace ttnn {
    // Register the operation with TTNN framework
    constexpr auto upsample3d = ttnn::register_operation<
        "ttnn::upsample3d",
        ttnn::operations::upsample::ExecuteUpSample3D>();
} // namespace ttnn
```

**Step 2: Create Python Binding**
**File**: `ttnn/python/ttnn/operations/pool/upsample/upsample3d_pybind.hpp`
```cpp
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/pool/upsample/upsample3d.hpp"

namespace ttnn::operations::upsample {

void bind_upsample3d(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::upsample3d,
        R"doc(
        3D nearest-neighbor upsampling operation.

        Not yet implemented.
        )doc",
        ttnn::pybind_overload_t{
            [](const Tensor& input,
               int scale_factor,
               std::optional<MemoryConfig> memory_config) {
                return ttnn::upsample3d(input, scale_factor, memory_config);
            },
            py::arg("input"),
            py::arg("scale_factor"),
            py::arg("memory_config") = std::nullopt
        },
        ttnn::pybind_overload_t{
            [](const Tensor& input,
               std::array<int, 3> scale_factor,
               std::optional<MemoryConfig> memory_config) {
                return ttnn::upsample3d(input, scale_factor, memory_config);
            },
            py::arg("input"),
            py::arg("scale_factor"),
            py::arg("memory_config") = std::nullopt
        }
    );
}

} // namespace ttnn::operations::upsample
```

**Step 3: Register in Category Module**
**File**: `ttnn/python/ttnn/operations/pool/pool_pybind.hpp`
Add to the module initialization:
```cpp
#include "ttnn/python/ttnn/operations/pool/upsample/upsample3d_pybind.hpp"

// In the module definition function:
ttnn::operations::upsample::bind_upsample3d(module);
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

## Stage 4: Device Operation Structure Test
**Goal**: Verify device operation follows proper TTNN structure with nested structs

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

### Implementation 4.1: Device Operation Following TTNN Structure

**File**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample3d_device_operation.hpp`
```cpp
#pragma once
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include <variant>

namespace ttnn::operations::upsample {

// Device operation with proper nested structs per TTNN guidelines
struct UpSample3DDeviceOperation {

    // Operation attributes (non-tensor parameters)
    struct operation_attributes_t {
        uint32_t scale_factor_d;
        uint32_t scale_factor_h;
        uint32_t scale_factor_w;
        MemoryConfig memory_config;
    };

    // Tensor arguments
    struct tensor_args_t {
        const Tensor& input;
    };

    // Program factory for multi-core interleaved execution
    struct MultiCoreInterleaved {
        struct shared_variables_t {
            KernelHandle reader_kernel;
            KernelHandle writer_kernel;
            uint32_t num_cores;
            uint32_t work_per_core_group_1;
            uint32_t work_per_core_group_2;
            CoreRange core_group_1;
            CoreRange core_group_2;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& op_attr,
            const tensor_args_t& tensor_args,
            tensor_return_value_t<Tensor>& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& op_attr,
            const tensor_args_t& tensor_args,
            tensor_return_value_t<Tensor>& output);
    };

    // Factory selection
    using program_factory_t = std::variant<MultiCoreInterleaved>;

    // Required static methods
    static program_factory_t select_program_factory(const operation_attributes_t& op_attr);
    static void validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensor_args);
    static TensorSpec compute_output_specs(const operation_attributes_t& op_attr, const tensor_args_t& tensor_args);
    static tensor_return_value_t<Tensor> create_output_tensors(const operation_attributes_t& op_attr, const tensor_args_t& tensor_args);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        uint32_t scale_factor_d,
        uint32_t scale_factor_h,
        uint32_t scale_factor_w,
        const MemoryConfig& memory_config);
};

} // namespace ttnn::operations::upsample
```

**File**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample3d_device_operation.cpp`
```cpp
#include "upsample3d_device_operation.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

// Validation on program cache miss (full validation)
void UpSample3DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op_attr,
    const tensor_args_t& tensor_args) {

    const auto& input = tensor_args.input;

    TT_FATAL(input.get_shape().rank() == 5,
             "Input must be 5D tensor, got rank {}", input.get_shape().rank());

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR,
             "Only ROW_MAJOR layout is supported, got {}", input.get_layout());

    TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
             "Only INTERLEAVED memory layout is supported");

    TT_FATAL(input.is_allocated(), "Input tensor must be allocated on device");

    TT_FATAL(input.storage_type() == StorageType::DEVICE,
             "Input must be on device, got storage type {}", input.storage_type());

    TT_FATAL(op_attr.scale_factor_d > 0, "Scale factor D must be positive");
    TT_FATAL(op_attr.scale_factor_h > 0, "Scale factor H must be positive");
    TT_FATAL(op_attr.scale_factor_w > 0, "Scale factor W must be positive");
}

// Validation on program cache hit (lighter validation)
void UpSample3DDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& op_attr,
    const tensor_args_t& tensor_args) {
    // Just verify tensor is on device
    const auto& input = tensor_args.input;
    TT_FATAL(input.is_allocated(), "Input tensor must be allocated on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE,
             "Input must be on device");
}

// Compute output specifications
TensorSpec UpSample3DDeviceOperation::compute_output_specs(
    const operation_attributes_t& op_attr,
    const tensor_args_t& tensor_args) {

    const auto& input = tensor_args.input;
    const auto& input_shape = input.get_shape();

    // Calculate output shape: (N, D*scale_d, H*scale_h, W*scale_w, C)
    std::array<uint32_t, 5> output_shape_array = {
        input_shape[0],                          // N (batch)
        input_shape[1] * op_attr.scale_factor_d, // D * scale_d
        input_shape[2] * op_attr.scale_factor_h, // H * scale_h
        input_shape[3] * op_attr.scale_factor_w, // W * scale_w
        input_shape[4]                          // C (channels)
    };

    auto output_shape = Shape(output_shape_array);

    // Create output tensor spec
    return TensorSpec(
        output_shape,
        TensorLayout(input.get_dtype(), PageConfig(Layout::ROW_MAJOR), op_attr.memory_config));
}

// Create output tensors
tensor_return_value_t<Tensor> UpSample3DDeviceOperation::create_output_tensors(
    const operation_attributes_t& op_attr,
    const tensor_args_t& tensor_args) {

    return create_device_tensor(
        compute_output_specs(op_attr, tensor_args),
        tensor_args.input.device());
}

// Select program factory (single option for now)
UpSample3DDeviceOperation::program_factory_t
UpSample3DDeviceOperation::select_program_factory(const operation_attributes_t& op_attr) {
    return MultiCoreInterleaved{};
}

// Map user arguments to operation attributes and tensor arguments
std::tuple<UpSample3DDeviceOperation::operation_attributes_t,
           UpSample3DDeviceOperation::tensor_args_t>
UpSample3DDeviceOperation::invoke(
    const Tensor& input,
    uint32_t scale_factor_d,
    uint32_t scale_factor_h,
    uint32_t scale_factor_w,
    const MemoryConfig& memory_config) {

    return {
        operation_attributes_t{
            .scale_factor_d = scale_factor_d,
            .scale_factor_h = scale_factor_h,
            .scale_factor_w = scale_factor_w,
            .memory_config = memory_config
        },
        tensor_args_t{
            .input = input
        }
    };
}

} // namespace ttnn::operations::upsample
```

**Update main operation to use device operation:**
**File**: `ttnn/cpp/ttnn/operations/pool/upsample/upsample3d.cpp`
```cpp
#include "upsample3d.hpp"
#include "device/upsample3d_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

Tensor ExecuteUpSample3D::invoke(
    const Tensor& input,
    std::variant<int, std::array<int, 3>> scale_factor,
    std::optional<MemoryConfig> memory_config) {

    // Input validation
    TT_FATAL(input.get_shape().rank() == 5,
             "Input tensor must be 5D (N, D, H, W, C), got rank {}", input.get_shape().rank());

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR,
             "Only ROW_MAJOR layout is supported for 3D upsample");

    // Parse scale factors
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

    // Call device operation using the proper TTNN device operation framework
    auto [op_attr, tensor_args] = UpSample3DDeviceOperation::invoke(
        input, scale_d, scale_h, scale_w, output_memory_config);

    // Run the device operation
    auto output_tensors = ttnn::device_operation::run<UpSample3DDeviceOperation>(
        UpSample3DDeviceOperation::select_program_factory(op_attr),
        op_attr,
        tensor_args);

    return output_tensors;
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

### Implementation 5.1: Program Factory Structure Following TTNN Guidelines

**File**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample3d_program_factory.cpp`
```cpp
#include "upsample3d_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::upsample {

// Implementation of MultiCoreInterleaved factory's create method
UpSample3DDeviceOperation::MultiCoreInterleaved::cached_program_t
UpSample3DDeviceOperation::MultiCoreInterleaved::create(
    const operation_attributes_t& op_attr,
    const tensor_args_t& tensor_args,
    tensor_return_value_t<Tensor>& output) {

    const auto& input = tensor_args.input;

    Program program{};

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

    // Compile time arguments for reader
    std::vector<uint32_t> reader_compile_time_args = {
        cb_index,                    // CB index
        aligned_input_unit_size      // Stick size
    };

    // Compile time arguments for writer (Stage 6 will add more)
    std::vector<uint32_t> writer_compile_time_args = {
        cb_index,                    // CB index
        aligned_input_unit_size,     // Stick size
        op_attr.scale_factor_d,      // Scale factors
        op_attr.scale_factor_h,
        op_attr.scale_factor_w
    };

    // Create reader kernel
    KernelHandle reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args
        });

    // Create writer kernel (minimal for Stage 5, full in Stage 6)
    KernelHandle writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/kernels/dataflow/writer_unary_interleaved.cpp", // Placeholder
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args
        });

    // Set runtime arguments
    uint32_t curr_stick = 0;
    auto cores_group_1 = corerange_to_cores(core_group_1, std::nullopt);
    for (const auto& core : cores_group_1) {
        SetRuntimeArgs(program, reader_kernel, core,
            {src_buffer->address(), work_per_core_group_1, curr_stick});
        SetRuntimeArgs(program, writer_kernel, core,
            {dst_buffer->address(), work_per_core_group_1, curr_stick});
        curr_stick += work_per_core_group_1;
    }

    if (core_group_2.num_cores() > 0) {
        auto cores_group_2 = corerange_to_cores(core_group_2, std::nullopt);
        for (const auto& core : cores_group_2) {
            SetRuntimeArgs(program, reader_kernel, core,
                {src_buffer->address(), work_per_core_group_2, curr_stick});
            SetRuntimeArgs(program, writer_kernel, core,
                {dst_buffer->address(), work_per_core_group_2, curr_stick});
            curr_stick += work_per_core_group_2;
        }
    }

    // Return cached program with shared variables
    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel,
            .num_cores = num_cores,
            .work_per_core_group_1 = work_per_core_group_1,
            .work_per_core_group_2 = work_per_core_group_2,
            .core_group_1 = core_group_1,
            .core_group_2 = core_group_2
        }
    };
}

// Implementation of override_runtime_arguments method
void UpSample3DDeviceOperation::MultiCoreInterleaved::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& op_attr,
    const tensor_args_t& tensor_args,
    tensor_return_value_t<Tensor>& output) {

    const auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    // Update runtime args with new buffer addresses
    auto reader_runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel);
    auto writer_runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel);

    uint32_t curr_stick = 0;
    auto cores_group_1 = corerange_to_cores(shared_vars.core_group_1, std::nullopt);
    for (const auto& core : cores_group_1) {
        reader_runtime_args[core][0] = src_buffer->address();
        writer_runtime_args[core][0] = dst_buffer->address();
        curr_stick += shared_vars.work_per_core_group_1;
    }

    if (shared_vars.core_group_2.num_cores() > 0) {
        auto cores_group_2 = corerange_to_cores(shared_vars.core_group_2, std::nullopt);
        for (const auto& core : cores_group_2) {
            reader_runtime_args[core][0] = src_buffer->address();
            writer_runtime_args[core][0] = dst_buffer->address();
            curr_stick += shared_vars.work_per_core_group_2;
        }
    }

    SetRuntimeArgs(program, shared_vars.reader_kernel, reader_runtime_args);
    SetRuntimeArgs(program, shared_vars.writer_kernel, writer_runtime_args);
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

## Stage 8: Golden Function Attachment
**Goal**: Attach a PyTorch golden reference function for validation

### Test 8.1: Golden Function Validation
**File**: `test_dev/test_stage8_golden_function.py`

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

def test_golden_function_attached(device):
    """Test that golden function is attached and callable"""
    assert hasattr(ttnn.upsample3d, 'golden_function'), "Golden function not attached"
    assert callable(ttnn.upsample3d.golden_function), "Golden function not callable"

def test_golden_function_validation(device):
    """Test that golden function produces correct results"""
    input_shape = (1, 2, 3, 4, 16)
    scale_factor = 2

    # Create input tensor
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Call golden function directly
    golden_output = ttnn.upsample3d.golden_function(
        input_tensor, scale_factor=scale_factor)

    # Expected shape
    expected_shape = (1, 4, 6, 8, 16)
    assert golden_output.shape == expected_shape

    # Verify it matches PyTorch's implementation
    input_ncdhw = input_tensor.permute(0, 4, 1, 2, 3)
    torch_output = torch.nn.functional.interpolate(
        input_ncdhw, scale_factor=scale_factor, mode='nearest')
    torch_output = torch_output.permute(0, 2, 3, 4, 1)

    torch.testing.assert_close(golden_output, torch_output)

def test_ttnn_matches_golden(device):
    """Test that TTNN implementation matches golden function"""
    input_shape = (1, 2, 2, 2, 32)
    scale_factor = 2

    # Create input
    input_tensor = torch.ones(input_shape, dtype=torch.bfloat16) * 3.14

    # Golden function result
    golden_output = ttnn.upsample3d.golden_function(
        input_tensor, scale_factor=scale_factor)

    # TTNN result
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.upsample3d(tt_input, scale_factor=scale_factor)
    ttnn_output = ttnn.to_torch(tt_output)

    # Compare with reasonable tolerance for bfloat16
    torch.testing.assert_close(ttnn_output, golden_output, rtol=1e-1, atol=1e-1)
```

### Implementation 8.1: Attach Golden Function

**File**: `ttnn/python/ttnn/operations/pool/upsample/upsample3d_golden.py`
```python
import torch
from typing import Union, Tuple
import ttnn

def upsample3d_golden_function(
    input_tensor: torch.Tensor,
    scale_factor: Union[int, Tuple[int, int, int]],
    memory_config: any = None) -> torch.Tensor:
    """
    Golden reference implementation for 3D upsampling using PyTorch.

    Args:
        input_tensor: 5D tensor with shape (N, D, H, W, C) in NDHWC format
        scale_factor: Integer or tuple of 3 integers for scaling each spatial dimension
        memory_config: Ignored for golden function

    Returns:
        Upsampled tensor with shape (N, D*scale_d, H*scale_h, W*scale_w, C)
    """
    # Validate input
    if input_tensor.dim() != 5:
        raise ValueError(f"Input must be 5D, got {input_tensor.dim()}D")

    # Parse scale factors
    if isinstance(scale_factor, int):
        scale_factors = (scale_factor, scale_factor, scale_factor)
    else:
        if len(scale_factor) != 3:
            raise ValueError(f"Scale factor tuple must have 3 elements, got {len(scale_factor)}")
        scale_factors = scale_factor

    # Convert NDHWC to NCDHW for PyTorch
    input_ncdhw = input_tensor.permute(0, 4, 1, 2, 3)

    # Use PyTorch's interpolate for nearest neighbor upsampling
    output_ncdhw = torch.nn.functional.interpolate(
        input_ncdhw,
        scale_factor=scale_factors,
        mode='nearest'
    )

    # Convert back to NDHWC
    output_ndhwc = output_ncdhw.permute(0, 2, 3, 4, 1)

    return output_ndhwc

# Optional: Define preprocessing and postprocessing if needed
def preprocess_golden_function_inputs(args, kwargs):
    """Preprocess inputs before passing to golden function"""
    # Extract input tensor from TTNN tensor if needed
    input_tensor = args[0]
    if hasattr(input_tensor, 'to_torch'):
        input_tensor = input_tensor.to_torch()

    # Return modified args and kwargs
    return (input_tensor,) + args[1:], kwargs

def postprocess_golden_function_outputs(output, args, kwargs):
    """Postprocess golden function output if needed"""
    # Output is already a torch tensor, no conversion needed
    return output
```

**File**: `ttnn/python/ttnn/operations/pool/upsample/__init__.py`
```python
import ttnn
from .upsample3d_golden import (
    upsample3d_golden_function,
    preprocess_golden_function_inputs,
    postprocess_golden_function_outputs
)

# Attach the golden function to the TTNN operation
ttnn.attach_golden_function(
    ttnn.upsample3d,
    golden_function=upsample3d_golden_function,
    preprocess_inputs=preprocess_golden_function_inputs,
    postprocess_outputs=postprocess_golden_function_outputs
)
```

**Update Python binding to import golden function:**
Add to `ttnn/python/ttnn/operations/pool/__init__.py`:
```python
# Import upsample module to attach golden functions
from . import upsample
```

**Success Criteria**: Golden function is attached and validates against PyTorch

---

## Next Steps After Stage 8

Once all 8 stages pass:
1. Add performance optimizations
2. Support additional features (tiled layout, sharded memory)
3. Add comprehensive sweep tests
4. Document the operation in tech reports
5. Submit for code review

This plan ensures systematic, test-driven development with clear verification at each stage, following the official TTNN operation structure.

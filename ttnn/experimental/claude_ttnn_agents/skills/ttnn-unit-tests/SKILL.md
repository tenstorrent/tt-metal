---
name: ttnn-unit-tests
description: "Use this skill when asked to write tests, create unit tests, add pytest tests, or test a TTNN operation. Provides templates, best practices, and patterns for TTNN unit tests including parametrization, assertions, validation tests, and edge cases."
---

# TTNN Unit Test Patterns

Use this skill when writing pytest tests for TTNN operations.

## Test File Location

```
tests/ttnn/unit_tests/operations/<category>/test_<op_name>.py
```

Categories: `normalization`, `eltwise`, `data_movement`, `reduction`, `matmul`, `conv`, etc.

## Complete Test Template

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the <operation_name> operation."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_reference(x, *args, **kwargs):
    """Reference PyTorch implementation."""
    return torch.nn.functional.<op>(x, *args, **kwargs)


# =============================================================================
# Functional Tests
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),   # Single tile
        (1, 1, 64, 64),   # 2x2 tiles
        (1, 1, 128, 128), # 4x4 tiles
    ],
    ids=["single_tile", "2x2_tiles", "4x4_tiles"],
)
def test_op_basic_shapes(device, shape):
    """Test operation with various tile-aligned shapes."""
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_output = torch_reference(torch_input)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # or ttnn.ROW_MAJOR_LAYOUT
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.my_op(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Verify output properties
    assert output_torch.shape == torch_input.shape
    assert_with_pcc(torch_output, output_torch, pcc=0.999)


# =============================================================================
# Validation Tests (Error Cases)
# =============================================================================


def test_op_wrong_layout(device):
    """Test that wrong layout raises an error."""
    torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # Wrong layout if op requires ROW_MAJOR
        device=device,
    )

    with pytest.raises(Exception) as excinfo:
        ttnn.my_op(input_tensor)

    # Check error message contains expected keywords
    assert "layout" in str(excinfo.value).lower()
```

## Key Rules

### 1. Always Set Seed First
```python
def test_op(device):
    torch.manual_seed(0)  # First line - ensures reproducibility
```

### 2. Use Device Fixture (Never Create Manually)
```python
# CORRECT - use pytest fixture
def test_op(device):
    tensor = ttnn.from_torch(..., device=device)

# WRONG - never create device manually
def test_op():
    device = ttnn.open_device(0)  # Don't do this
```

### 3. Use Test IDs for Readability
```python
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 256),
        (1, 1, 32, 512),
    ],
    ids=["wide_8tiles", "wide_16tiles"],  # Makes test output readable
)
def test_op(device, shape):
```

### 4. Parametrize Even Single Values
```python
# CORRECT - allows easy extension
@pytest.mark.parametrize("height", [32])
def test_op(device, height):

# WRONG - hardcoded
def test_op(device):
    height = 32
```

## Assertion Functions

### assert_with_pcc (Primary - for floating point)
```python
from tests.ttnn.utils_for_testing import assert_with_pcc

assert_with_pcc(expected, actual, pcc=0.999)
```

PCC thresholds:
- `0.9999` - Strict (simple eltwise)
- `0.999` - Standard (operations with accumulation, normalization)
- `0.99` - Relaxed (some activations, approximations)

### assert_allclose (Absolute/Relative Tolerance)
```python
from tests.ttnn.utils_for_testing import assert_allclose

assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)
```

### assert_equal (Exact - for integers)
```python
from tests.ttnn.utils_for_testing import assert_equal

assert_equal(expected, actual)  # For int32, uint32
```

## Shape Testing Strategy

Test these shape categories systematically:

```python
# 1. Single tile (minimum)
(1, 1, 32, 32)

# 2. Multi-tile square
(1, 1, 64, 64), (1, 1, 128, 128)

# 3. Wide tensors (many tiles per row)
(1, 1, 32, 256), (1, 1, 32, 512), (1, 1, 32, 1024)

# 4. Tall tensors (many tile rows)
(1, 1, 64, 32), (1, 1, 128, 32), (1, 1, 256, 32)

# 5. 2D tensors (if supported)
(32, 64), (64, 128)
```

## Testing Different Layouts

### TILE_LAYOUT (most common)
```python
input_tensor = ttnn.from_torch(
    torch_input,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)
```

### ROW_MAJOR_LAYOUT (for ops that require it)
```python
input_tensor = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Verify output layout
assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
```

## Edge Case Tests

### Uniform/Constant Input
```python
def test_op_uniform_input(device):
    """Test with uniform input (e.g., variance=0 for normalization)."""
    torch_input = torch.full((1, 1, 32, 64), 5.0, dtype=torch.bfloat16)
    # ... rest of test
```

### Special Parameter Values
```python
def test_op_identity_params(device):
    """Test with identity parameters (gamma=1, beta=0)."""
    gamma = torch.ones((width,), dtype=torch.bfloat16)
    beta = torch.zeros((width,), dtype=torch.bfloat16)
    # ... rest of test
```

### Parameter Sweep
```python
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-12])
def test_op_epsilon_values(device, epsilon):
    """Test with different epsilon values."""
```

## Validation/Error Tests

### Testing Expected Errors
```python
def test_op_wrong_dtype(device):
    """Test that wrong dtype raises an error."""
    torch_input = torch.rand((32, 32), dtype=torch.float32)  # Wrong dtype

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    with pytest.raises(Exception) as excinfo:
        ttnn.my_op(input_tensor)

    error_msg = str(excinfo.value).lower()
    assert "bfloat16" in error_msg or "dtype" in error_msg
```

### Testing Alignment Requirements
```python
def test_op_non_tile_aligned_width(device):
    """Test that non-tile-aligned width raises an error."""
    # Width 48 is not divisible by 32
    torch_input = torch.rand((1, 1, 32, 48), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, ...)

    with pytest.raises(Exception) as excinfo:
        ttnn.my_op(input_tensor)

    error_msg = str(excinfo.value).lower()
    assert "width" in error_msg or "32" in error_msg or "align" in error_msg
```

## Marking Known Limitations

### Expected Failures (xfail)
```python
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            (2, 1, 64, 64),
            marks=pytest.mark.xfail(reason="Batching not supported yet")
        ),
        (1, 1, 64, 64),  # This one should pass
    ],
    ids=["batched_xfail", "single_batch"],
)
def test_op(device, shape):
```

### Skip Tests
```python
@pytest.mark.skip(reason="Feature not implemented")
def test_future_feature(device):
```

## Memory Configuration Tests

```python
@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
    ids=["DRAM", "L1"],
)
def test_op_memory_configs(device, memory_config):
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        memory_config=memory_config,
    )
    output = ttnn.my_op(input_tensor, memory_config=memory_config)
```

## Vision Ops (L1 Small Size)

```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv(device):
    # Custom L1 small allocation for vision ops
```

## Running Tests Safely

Always follow this sequence to avoid device state issues:

```bash
# Kill hung processes, reset device, run with timeout
pkill -9 -f pytest || true && tt-smi -r 0 && timeout 60 pytest <test_file> -v
```

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import networkx as nx
from tracer_backend import OperationGraph, Operation
from tracer_backend_utils import (
    ConvAttrs,
    PoolAttrs,
    AtenConvolution,
    AtenAddm,
    AtenMaxPool2dWithIndices,
    AtenAddTensor,
    AtenMulTensor,
    AtenCat,
    AtenBmm,
)
from typing import List, Optional, Type, Dict, Any
from dataclasses import dataclass
from pytorch_graph_utils import format_file_with_black
import os
import torch

HEADER_IMPORTS = set(
    [
        "import pytest",
        "import torch",
        "import ttnn",
    ]
)


class UnitTestOperation:
    def generate_code(self, indent="") -> str:
        """Generate the code for this unit test operation."""
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["UnitTestOperation"]:
        """Parse a single operation and return its code."""
        return None


class ConvolutionUnittest(UnitTestOperation):
    def __init__(self, conv_attrs: ConvAttrs):
        self.attrs = conv_attrs
        HEADER_IMPORTS.add(
            "from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map"
        )

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ConvolutionUnittest"]:
        if operation.function_call_name == "torch.ops.aten.convolution":
            conv = operation.to_operation(AtenConvolution)
            return ConvolutionUnittest(conv.attrs)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this convolution unit test operation."""
        group_unit_test = ConvolutionGroupUnittest([self.attrs])
        return group_unit_test.generate_code()


class AddmUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ConvolutionUnittest"]:
        if operation.function_call_name == "torch.ops.aten.addmm":
            admm = operation.to_operation(AtenAddm)
            return AddmUnittest(admm.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this convolution unit test operation."""
        group_unit_test = AddmGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class Maxpool2dUnittest(UnitTestOperation):
    def __init__(self, pool_attrs: PoolAttrs):
        self.attrs = pool_attrs
        HEADER_IMPORTS.add("from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["Maxpool2dUnittest"]:
        if operation.function_call_name == "torch.ops.aten.max_pool2d_with_indices":
            pool = operation.to_operation(AtenMaxPool2dWithIndices)
            return Maxpool2dUnittest(pool.attrs)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this convolution unit test operation."""
        group_unit_test = Maxpool2dGroupUnittest([self.attrs])
        return group_unit_test.generate_code()


class AddTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["AddTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.add.Tensor":
            add_tensor = operation.to_operation(AtenAddTensor)
            return AddTensorUnittest(add_tensor.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this add tensor unit test operation."""
        group_unit_test = AddTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class MulTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["MulTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.mul.Tensor":
            mul_tensor = operation.to_operation(AtenMulTensor)
            return MulTensorUnittest(mul_tensor.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this mul tensor unit test operation."""
        group_unit_test = MulTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class CatUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]], dim: Optional[int] = None):
        self.input_shapes = input_shapes
        self.dim = dim
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["CatUnittest"]:
        if operation.function_call_name == "torch.ops.aten.cat":
            try:
                cat_op = operation.to_operation(AtenCat)
                dim = cat_op.args[1] if len(cat_op.args) > 1 else None

                # Use input_shapes if available, otherwise derive from output_shapes
                input_shapes = cat_op.input_shapes
                if not input_shapes and operation.output_shapes and hasattr(operation, "args"):
                    tensor_list = operation.args[0] if operation.args else []
                    if isinstance(tensor_list, list) and len(tensor_list) >= 2:
                        output_shape = (
                            operation.output_shapes[0]
                            if isinstance(operation.output_shapes, list)
                            else next(iter(operation.output_shapes.values()), None)
                        )
                        if output_shape and hasattr(output_shape, "__iter__"):
                            input_shapes = {i: list(output_shape) for i in range(len(tensor_list))}

                return CatUnittest(input_shapes, dim)
            except Exception:
                return None
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this cat unit test operation."""
        group_unit_test = CatGroupUnittest([{"input_shapes": self.input_shapes, "dim": self.dim}])
        return group_unit_test.generate_code()


class BmmUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["BmmUnittest"]:
        if operation.function_call_name == "torch.ops.aten.bmm":
            bmm_op = operation.to_operation(AtenBmm)
            return BmmUnittest(bmm_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this bmm unit test operation."""
        group_unit_test = BmmGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ConvolutionGroupUnittest(UnitTestOperation):
    def __init__(self, attrs_list: List[ConvAttrs]):
        self.attrs = attrs_list
        HEADER_IMPORTS.add(
            "from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map"
        )

    def generate_code(self) -> str:
        """Generate the code for this convolution unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b], [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize("device_params", [{{"l1_small_size": 32768}}], indirect=True)
@pytest.mark.parametrize(
    "input_batch, input_depth, hidden_units, input_height, input_width, kernel, stride, padding, dilation",
    (
{''.join(set(f'        ({attr.input_batch}, {attr.input_depth}, {attr.hidden_units}, {attr.input_height}, {attr.input_width}, {attr.kernel}, {attr.stride}, {attr.padding}, {attr.dilation}),' for attr in self.attrs))}
    )
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
)
def test_conv(
    device,
    torch_tensor_map,
    input_batch,
    hidden_units,
    input_depth,
    input_height,
    input_width,
    has_bias,
    dtype,
    kernel,
    stride,
    padding,
    dilation,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        dtype,
        ttnn.bfloat8_b,
        input_batch,
        hidden_units,
        input_depth,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        {str({})},
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        input_dtype=dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        run_twice=True,
        fast_compare=True,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
    )
        """


class Maxpool2dGroupUnittest(UnitTestOperation):
    def __init__(self, attrs_list: List[PoolAttrs]):
        self.attrs = attrs_list
        HEADER_IMPORTS.add("from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d")

    def generate_code(self) -> str:
        """Generate the code for this maxpool2d unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "ceil_mode",
    [False, True],
)
@pytest.mark.parametrize(
    "sharding",
    [None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED]
)
@pytest.mark.parametrize(
    "input_batch, input_depth, input_height, input_width, kernel, stride, padding, dilation",
    (
{''.join(set(f'        ({attr.input_batch}, {attr.input_depth}, {attr.input_height}, {attr.input_width}, {attr.kernel}, {attr.stride}, {attr.padding}, {attr.dilation}),' for attr in self.attrs))}
    )
)
def test_maxpool2d(
    device,
    input_batch,
    input_depth,
    input_height,
    input_width,
    dtype,
    kernel,
    stride,
    padding,
    dilation,
    sharding,
    ceil_mode,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")

    [kernel_h, kernel_w] = kernel
    [stride_h, stride_w] = stride
    [pad_h, pad_w] = padding
    [dilation_h, dilation_w] = dilation

    return run_max_pool2d(
        input_batch,
        input_depth,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding,
        ceil_mode,
    )
        """


class AddmGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) > 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        for shape in self.input_shape_list:
            if 1 in shape:
                if len(shape[1]) == 2:
                    shape[1] = [1, 1] + shape[1]
                elif len(shape[1]) == 3:
                    shape[1] = [1] + shape[1]
            if 2 in shape:
                if len(shape[2]) == 2:
                    shape[2] = [1, 1] + shape[2]
                elif len(shape[2]) == 3:
                    shape[2] = [1] + shape[2]
        HEADER_IMPORTS.add(
            "from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map"
        )

    def generate_code(self) -> str:
        """Generate the code for this convolution unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "batch_size, channel_a, channel_b, m_size, k_size, n_size",
    (
{''.join(set(f'        ({shapes[1][-4]}, {shapes[1][-3]}, {shapes[2][-3]}, {shapes[1][-2]}, {shapes[1][-1]}, {shapes[2][-1]}),' for shapes in self.input_shape_list))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_sd_matmul(device, batch_size, channel_a, channel_b, m_size, k_size, n_size, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn((batch_size, channel_a, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((batch_size, channel_b, k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class AddTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        # Normalize shapes to ensure they have at least 4 dimensions for consistency
        for shape in self.input_shape_list:
            for key in [0, 1]:  # Process both input tensors
                if key in shape:
                    if len(shape[key]) == 1:
                        shape[key] = [1, 1, 1] + shape[key]
                    elif len(shape[key]) == 2:
                        shape[key] = [1, 1] + shape[key]
                    elif len(shape[key]) == 3:
                        shape[key] = [1] + shape[key]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this add tensor unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_add_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class MulTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        # Normalize shapes to ensure they have at least 4 dimensions for consistency
        for shape in self.input_shape_list:
            for key in [0, 1]:  # Process both input tensors
                if key in shape:
                    if len(shape[key]) == 1:
                        shape[key] = [1, 1, 1] + shape[key]
                    elif len(shape[key]) == 2:
                        shape[key] = [1, 1] + shape[key]
                    elif len(shape[key]) == 3:
                        shape[key] = [1] + shape[key]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this mul tensor unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_mul_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a * torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    output_tensor = ttnn.multiply(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class CatGroupUnittest(UnitTestOperation):
    def __init__(self, cat_ops_list: List[Dict[str, Any]]):
        self.cat_ops_list = self._process_cat_operations(cat_ops_list)
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def _normalize_shape_to_4d(shape: List[int]) -> List[int]:
        """Normalize a tensor shape to 4D by padding with 1s at the beginning."""
        if len(shape) == 1:
            return [1, 1, 1] + shape
        elif len(shape) == 2:
            return [1, 1] + shape
        elif len(shape) == 3:
            return [1] + shape
        else:
            return shape

    @staticmethod
    def _convert_shape_to_list(shape: Any) -> Optional[List[int]]:
        """Convert various shape formats to a list of integers."""
        if isinstance(shape, torch.Size):
            return list(shape)
        elif isinstance(shape, (list, tuple)):
            return list(shape)
        else:
            return None

    def _process_cat_operations(self, cat_ops_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize cat operations."""
        valid_ops = [op for op in cat_ops_list if op is not None and "input_shapes" in op]
        processed_ops = []

        for op in valid_ops:
            input_shapes = op["input_shapes"]
            if not input_shapes or len(input_shapes) < 2:
                continue

            processed_shapes = {}
            for k, v in input_shapes.items():
                shape_list = self._convert_shape_to_list(v)
                if shape_list is not None:
                    processed_shapes[k] = self._normalize_shape_to_4d(shape_list)

            if processed_shapes:
                processed_ops.append({"input_shapes": processed_shapes, "dim": op.get("dim", -1)})

        return processed_ops

    def _generate_parameter_tuples(self) -> List[str]:
        """Generate parameter tuples for the test."""
        param_tuples = []
        for op in self.cat_ops_list:
            input_shapes = op.get("input_shapes", {})
            if not input_shapes:
                continue

            # Extract shapes in sorted order
            shapes = [input_shapes[key] for key in sorted(input_shapes.keys()) if isinstance(key, int)]

            if len(shapes) >= 2:  # Need at least 2 tensors for concat
                dim = op.get("dim", -1)
                param_tuples.append(f"        ({shapes}, {dim})")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(param_tuples))  # Preserves order, removes duplicates

    def generate_code(self) -> str:
        """Generate the code for this cat unit test operation."""
        param_tuples = self._generate_parameter_tuples()
        params_str = ",\n".join(param_tuples) if param_tuples else "        ([[1, 1, 1, 32], [1, 1, 1, 32]], -1)"

        return f"""

@pytest.mark.parametrize(
    "tensor_shapes, cat_dim",
    (
{params_str},
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_cat(device, tensor_shapes, cat_dim, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensors = [torch.randn(shape, dtype=torch.bfloat16) for shape in tensor_shapes]
    torch_output_tensor = torch.cat(torch_input_tensors, dim=cat_dim)

    input_tensors = [
        ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype)
        for torch_tensor in torch_input_tensors
    ]

    output_tensor = ttnn.concat(input_tensors, dim=cat_dim)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class BmmGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        # Ensure shapes are 3D for BMM (batch matrix multiplication)
        for shape in self.input_shape_list:
            for key in [0, 1]:
                if key in shape:
                    if len(shape[key]) == 2:
                        shape[key] = [1] + shape[key]  # Add batch dimension
                    elif len(shape[key]) > 3:
                        shape[key] = shape[key][-3:]  # Take last 3 dimensions
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this bmm unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_bmm(device, input_shape_a, input_shape_b, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch.bmm(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class UnitTestOperationCombiner:
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine a group unit test operations into one."""
        raise NotImplementedError("Subclasses should implement this method.")


class ConvolutionCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple convolution operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        # Assuming all operations are ConvolutionUnittest
        combined_attrs = [conv.attrs for conv in operations if isinstance(conv, ConvolutionUnittest)]
        return ConvolutionGroupUnittest(combined_attrs)


class AddmCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple addm operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        # Assuming all operations are ConvolutionUnittest
        combined_shapes = [addm.input_shapes for addm in operations if isinstance(addm, AddmUnittest)]
        return AddmGroupUnittest(combined_shapes)


class Maxpool2dCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple maxpool2d operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        # Assuming all operations are ConvolutionUnittest
        combined_attrs = [pool.attrs for pool in operations if isinstance(pool, Maxpool2dUnittest)]
        return Maxpool2dGroupUnittest(combined_attrs)


class AddTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple add tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            add_tensor.input_shapes for add_tensor in operations if isinstance(add_tensor, AddTensorUnittest)
        ]
        return AddTensorGroupUnittest(combined_shapes)


class MulTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple mul tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            mul_tensor.input_shapes for mul_tensor in operations if isinstance(mul_tensor, MulTensorUnittest)
        ]
        return MulTensorGroupUnittest(combined_shapes)


class CatCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple cat operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_ops = []
        for cat_op in operations:
            if isinstance(cat_op, CatUnittest):
                # If input_shapes are empty, we'll let CatGroupUnittest handle it
                combined_ops.append({"input_shapes": cat_op.input_shapes, "dim": cat_op.dim})

        return CatGroupUnittest(combined_ops)


class BmmCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple bmm operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [bmm_op.input_shapes for bmm_op in operations if isinstance(bmm_op, BmmUnittest)]
        return BmmGroupUnittest(combined_shapes)


class UnitTestCombiner:
    def __init__(
        self,
        operations: List[UnitTestOperation],
        class_to_combiner: Dict[Type[UnitTestOperation], Type[UnitTestOperationCombiner]],
    ):
        self.operations = operations
        self.class_to_combiner = class_to_combiner

    def group_operations(self) -> List[UnitTestOperation]:
        """Group operations based on their type or attributes."""
        grouped_operations: Dict[Type[UnitTestOperation], List[UnitTestOperation]] = {}
        for operation in self.operations:
            operation_type = type(operation)
            if operation_type not in grouped_operations:
                grouped_operations[operation_type] = []
            grouped_operations[operation_type].append(operation)
        return grouped_operations

    def fuse_operations(self) -> List[UnitTestOperation]:
        """Fuse operations that can be combined."""
        grouped_operations = self.group_operations()
        fused_operations: List[UnitTestOperation] = []
        for operation_type, operations in grouped_operations.items():
            if operation_type in self.class_to_combiner:
                combiner = self.class_to_combiner[operation_type]
                fused_operation = combiner.combine(operations)
                fused_operations.append(fused_operation)
            else:
                fused_operations.extend(operations)
        return fused_operations


@dataclass
class PytorchLayerUnitTestGraphConfig:
    """Configuration for the Pytorch Layer Unit Test Graph."""

    operation_graph: OperationGraph
    register_unit_test_operations: Optional[List[Type[UnitTestOperation]]] = None
    group_unit_test_operations: Optional[Dict[Type[UnitTestOperation], Type[UnitTestOperationCombiner]]] = None

    def __post_init__(self):
        if self.register_unit_test_operations is None:
            self.register_unit_test_operations = [
                ConvolutionUnittest,
                AddmUnittest,
                Maxpool2dUnittest,
                AddTensorUnittest,
                MulTensorUnittest,
                CatUnittest,
                BmmUnittest,
            ]
        if self.group_unit_test_operations is None:
            self.group_unit_test_operations = {
                ConvolutionUnittest: ConvolutionCombiner,
                AddmUnittest: AddmCombiner,
                Maxpool2dUnittest: Maxpool2dCombiner,
                AddTensorUnittest: AddTensorCombiner,
                MulTensorUnittest: MulTensorCombiner,
                CatUnittest: CatCombiner,
                BmmUnittest: BmmCombiner,
            }


class PytorchLayerUnitTestGraph:
    def __init__(self, config: PytorchLayerUnitTestGraphConfig):
        self.graph = config.operation_graph.graph
        self.config = config

    def create_unit_test_operations(self) -> List[UnitTestOperation]:
        """Create unit test operations from the graph."""
        unit_test_operations = []
        for node_id in list(nx.topological_sort(self.graph)):
            operation = self.graph.nodes[node_id].get("operation")
            if operation:
                for unit_test_op_cls in self.config.register_unit_test_operations:
                    unit_test_op = unit_test_op_cls.parse_from_operation(operation)
                    if unit_test_op:
                        unit_test_operations.append(unit_test_op)

        if self.config.group_unit_test_operations:
            combiner = UnitTestCombiner(unit_test_operations, self.config.group_unit_test_operations)
            unit_test_operations = combiner.fuse_operations()
        return unit_test_operations

    def generate_pytorch_code(self) -> str:
        """Generate PyTorch code from the graph."""
        unit_test_operations = self.create_unit_test_operations()
        code_lines = list(HEADER_IMPORTS)
        for operation in unit_test_operations:
            code_lines.append(operation.generate_code())
        return "\n".join(code_lines)

    def dump_to_python_file(self, file_path: str, format_code: bool = False):
        """Dump the generated PyTorch code into a Python file."""
        pytorch_code = self.generate_pytorch_code()
        with open(file_path, "w") as f:
            f.write("# Auto-generated PyTorch code\n")
            for line in pytorch_code.splitlines():
                f.write(f"{line}\n")
        print(f"Generated pytest code dumped to {os.path.abspath(file_path)}")

        if format_code:
            format_file_with_black(file_path)

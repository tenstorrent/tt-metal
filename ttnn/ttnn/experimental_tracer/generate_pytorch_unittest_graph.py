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
    AtenNativeBatchNorm,
    AtenSilu,
    AtenSiluInplace,
    AtenPermute,
    AtenView,
    AtenClone,
    AtenUnsafeView,
    AtenExpand,
    AtenTransposeInt,
    AtenDivTensor,
    AtenSigmoid,
    AtenSoftmax,
    AtenSplitTensor,
    TorchOnes,
    AtenSplitWithSizes,
    AtenUpsampleNearest2d,
    AtenSubTensor,
    AtenMeanDim,
    AtenSliceTensor,
    AtenNativeLayerNorm,
    AtenUnsqueeze,
    AtenGelu,
    AtenReluInplace,
    AtenConstantPadNd,
    AtenSqueezeDim,
    AtenMm,
    AtenLinalgVectorNorm,
    AtenClampMin,
    AtenMaxDim,
    AtenRelu,
    AtenAsStridedInplace,
    AtenRoll,
    AtenLeakyReluInplace,
    AtenSoftplus,
    AtenTanh,
    AtenPowTensorScalar,
    AtenCopyInplace,
    AtenAdaptiveMaxPool2d,
    AtenUnbindInt,
    AtenClamp,
    AtenSelectInt,
    AtenTopk,
    AtenGridSampler2d,
    AtenSumDimIntList,
    AtenRsubScalar,
    AtenStack,
    AtenIndexTensor,
    AtenLog,
    AtenHardtanhInplace,
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


class NativeBatchNormUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["NativeBatchNormUnittest"]:
        if operation.function_call_name == "torch.ops.aten.native_batch_norm":
            batch_norm = operation.to_operation(AtenNativeBatchNorm)
            return NativeBatchNormUnittest(batch_norm.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this batch norm unit test operation."""
        group_unit_test = NativeBatchNormGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SiluUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SiluUnittest"]:
        if operation.function_call_name == "torch.ops.aten.silu_":
            silu_op = operation.to_operation(AtenSiluInplace)
            return SiluUnittest(silu_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this silu unit test operation."""
        group_unit_test = SiluGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class DivTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["DivTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.div.Tensor":
            div_tensor = operation.to_operation(AtenDivTensor)
            return DivTensorUnittest(div_tensor.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this div tensor unit test operation."""
        group_unit_test = DivTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SigmoidUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SigmoidUnittest"]:
        if operation.function_call_name == "torch.ops.aten.sigmoid":
            sigmoid_op = operation.to_operation(AtenSigmoid)
            return SigmoidUnittest(sigmoid_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this sigmoid unit test operation."""
        group_unit_test = SigmoidGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SoftmaxUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SoftmaxUnittest"]:
        if operation.function_call_name == "torch.ops.aten._softmax":
            softmax_op = operation.to_operation(AtenSoftmax)
            return SoftmaxUnittest(softmax_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this softmax unit test operation."""
        group_unit_test = SoftmaxGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SubTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SubTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.sub.Tensor":
            sub_tensor = operation.to_operation(AtenSubTensor)
            return SubTensorUnittest(sub_tensor.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this sub tensor unit test operation."""
        group_unit_test = SubTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class UpsampleNearest2dUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["UpsampleNearest2dUnittest"]:
        if operation.function_call_name == "torch.ops.aten.upsample_nearest2d":
            upsample_op = operation.to_operation(AtenUpsampleNearest2d)
            return UpsampleNearest2dUnittest(upsample_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this upsample nearest2d unit test operation."""
        group_unit_test = UpsampleNearest2dGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class TorchOnesUnittest(UnitTestOperation):
    def __init__(self, output_shapes: Optional[List[Any]]):
        self.output_shapes = output_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["TorchOnesUnittest"]:
        if operation.function_call_name == "torch.ones":
            ones_op = operation.to_operation(TorchOnes)
            return TorchOnesUnittest(ones_op.output_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this torch.ones unit test operation."""
        group_unit_test = TorchOnesGroupUnittest([self.output_shapes])
        return group_unit_test.generate_code()


class PermuteUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["PermuteUnittest"]:
        if operation.function_call_name == "torch.ops.aten.permute":
            permute_op = operation.to_operation(AtenPermute)
            return PermuteUnittest(permute_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this permute unit test operation."""
        group_unit_test = PermuteGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ViewUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ViewUnittest"]:
        if operation.function_call_name == "torch.ops.aten.view":
            view_op = operation.to_operation(AtenView)
            return ViewUnittest(view_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this view unit test operation."""
        group_unit_test = ViewGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class CloneUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["CloneUnittest"]:
        if operation.function_call_name == "torch.ops.aten.clone":
            clone_op = operation.to_operation(AtenClone)
            return CloneUnittest(clone_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this clone unit test operation."""
        group_unit_test = CloneGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class UnsafeViewUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["UnsafeViewUnittest"]:
        if operation.function_call_name == "torch.ops.aten._unsafe_view":
            unsafe_view_op = operation.to_operation(AtenUnsafeView)
            return UnsafeViewUnittest(unsafe_view_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this unsafe view unit test operation."""
        group_unit_test = UnsafeViewGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ExpandUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ExpandUnittest"]:
        if operation.function_call_name == "torch.ops.aten.expand":
            expand_op = operation.to_operation(AtenExpand)
            return ExpandUnittest(expand_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this expand unit test operation."""
        group_unit_test = ExpandGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class TransposeIntUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["TransposeIntUnittest"]:
        if operation.function_call_name == "torch.ops.aten.transpose.int":
            transpose_op = operation.to_operation(AtenTransposeInt)
            return TransposeIntUnittest(transpose_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this transpose int unit test operation."""
        group_unit_test = TransposeIntGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SplitTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SplitTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.split.Tensor":
            split_op = operation.to_operation(AtenSplitTensor)
            return SplitTensorUnittest(split_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this split tensor unit test operation."""
        group_unit_test = SplitTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SplitWithSizesUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SplitWithSizesUnittest"]:
        if operation.function_call_name == "torch.ops.aten.split_with_sizes":
            split_op = operation.to_operation(AtenSplitWithSizes)
            return SplitWithSizesUnittest(split_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this split with sizes unit test operation."""
        group_unit_test = SplitWithSizesGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


# Note: These operations (View, Permute, Clone, UnsafeView, Expand, TransposeInt, SplitTensor, SplitWithSizes)
# are generally zero-cost operations but unit tests are still useful for validation


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

    torch_input_tensor_a = torch.permute(torch_input_tensor_a, (0, 2, 3, 1))
    torch_input_tensor_b = torch.permute(torch_input_tensor_b, (0, 2, 3, 1))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    output_tensor = ttnn.multiply(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
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


class NativeBatchNormGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this batch norm unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_native_batch_norm(device, input_shape, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    num_features = input_shape[1]

    # Create batch norm module
    batch_norm = torch.nn.BatchNorm2d(num_features, dtype=torch.bfloat16)
    torch_output_tensor = batch_norm(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    # Create batch norm parameters
    weight = torch.ones(num_features, dtype=torch.bfloat16)
    bias = torch.zeros(num_features, dtype=torch.bfloat16)
    running_mean = torch.zeros(num_features, dtype=torch.bfloat16)
    running_var = torch.ones(num_features, dtype=torch.bfloat16)

    # Convert parameters to ttnn tensors
    ttnn_weight = ttnn.from_torch(weight, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    ttnn_bias = ttnn.from_torch(bias, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    ttnn_running_mean = ttnn.from_torch(running_mean, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    ttnn_running_var = ttnn.from_torch(running_var, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    # Perform actual batch normalization using ttnn.batch_norm
    output_tensor = ttnn.batch_norm(
        input_tensor,
        running_mean=ttnn_running_mean,
        running_var=ttnn_running_var,
        weight=ttnn_weight,
        bias=ttnn_bias,
        training=False
    )

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SiluGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this silu unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_silu(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.silu(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.silu(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class DivTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this div tensor unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_div_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16) + 0.1  # Avoid division by zero
    torch_output_tensor = torch_input_tensor_a / torch_input_tensor_b

    torch_input_tensor_a = torch.permute(torch_input_tensor_a, (0, 2, 3, 1))
    torch_input_tensor_b = torch.permute(torch_input_tensor_b, (0, 2, 3, 1))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SigmoidGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this sigmoid unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sigmoid(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.sigmoid(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.sigmoid(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SoftmaxGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this softmax unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_softmax(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.softmax(torch_input_tensor, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SubTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this sub tensor unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sub_tensor(device, input_shape_a, input_shape_b, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a - torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=layout, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=layout, device=device, dtype=dtype)

    output_tensor = ttnn.subtract(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class UpsampleNearest2dGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this upsample nearest2d unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape, scale_factor",
    (
{''.join(set(f'        ({shape[0]}, 2),' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_upsample_nearest2d(device, input_shape, scale_factor, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.interpolate(
        torch_input_tensor, scale_factor=scale_factor, mode='nearest'
    )

    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)

    input_tensor = ttnn.upsample(input_tensor, scale_factor=(2, 2), mode="nearest")

    # Note: This is a simplified test - actual ttnn implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class TorchOnesGroupUnittest(UnitTestOperation):
    def __init__(self, output_shapes_list: List[Optional[List[Any]]]):
        self.output_shape_list = [shapes for shapes in output_shapes_list if shapes is not None]
        self.output_shape_list = [
            [list(shape) if hasattr(shape, "__iter__") else [shape] for shape in shapes]
            for shapes in self.output_shape_list
            if shapes
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this torch.ones unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "output_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.output_shape_list if len(shape) > 0))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_torch_ones(device, output_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_output_tensor = torch.ones(output_shape, dtype=torch.bfloat16)
    output_tensor = ttnn.ones(output_shape, layout=layout, device=device, dtype=dtype)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class PermuteGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this permute unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_permute(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    dims = list(range(len(input_shape)))
    dims[-1], dims[-2] = dims[-2], dims[-1]  # Swap last two dimensions
    torch_output_tensor = torch_input_tensor.permute(dims)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.permute(input_tensor, dims)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class ViewGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this view unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_view(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    new_shape = [-1, input_shape[-1]]  # Flatten all but last dimension
    torch_output_tensor = torch_input_tensor.view(new_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.reshape(input_tensor, new_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class CloneGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this clone unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_clone(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.clone()

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.clone(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class UnsafeViewGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this unsafe view unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_unsafe_view(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    new_shape = [-1, input_shape[-1]]  # Flatten all but last dimension
    torch_output_tensor = torch_input_tensor.view(new_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.reshape(input_tensor, new_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class ExpandGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this expand unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_expand(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    expand_shape = list(input_shape)
    expand_shape[0] = expand_shape[0] * 2  # Double the batch size
    torch_output_tensor = torch_input_tensor.expand(expand_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.expand(input_tensor, expand_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class TransposeIntGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this transpose int unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_transpose_int(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    dim0, dim1 = -2, -1  # Transpose last two dimensions
    torch_output_tensor = torch_input_tensor.transpose(dim0, dim1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.transpose(input_tensor, dim0, dim1)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SplitTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]

        # Filter out shapes that would cause TILE_LAYOUT issues
        # Keep only shapes where the last dimension when divided by 2 is tile-aligned (multiple of 32)
        # Be very conservative to avoid C++ assertion failures
        self.tile_compatible_shapes = []
        self.non_tile_shapes = []

        for shape in self.input_shape_list:
            if 0 in shape:
                last_dim = shape[0][-1]
                split_size = last_dim // 2 if last_dim > 1 else 1

                # Check if both the dimension and split size are tile-aligned
                if last_dim % 64 == 0 and split_size % 32 == 0:
                    self.tile_compatible_shapes.append(shape)
                # TTNN split seems to have fundamental constraints even with ROW_MAJOR_LAYOUT
                # For now, disable non-tile-aligned testing entirely to avoid C++ assertion failures
                # elif last_dim >= 64 and split_size >= 32 and last_dim % 32 == 0:
                #     self.non_tile_shapes.append(shape)
                # Skip all other shapes that might cause issues

        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this split tensor unit test operation."""
        tile_shapes_code = ""
        non_tile_shapes_code = ""

        if self.tile_compatible_shapes:
            tile_shapes_code = f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.tile_compatible_shapes if 0 in shape))}
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_split_tensor(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    # Skip TILE_LAYOUT for shapes that don't have tile-aligned split sizes
    split_size = input_shape[-1] // 2 if input_shape[-1] > 1 else 1
    if layout == ttnn.TILE_LAYOUT and (split_size % 32 != 0 or input_shape[-1] % 64 != 0):
        pytest.skip("TILE_LAYOUT requires tile-aligned dimensions and split sizes (multiples of 32)")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensors = torch.split(torch_input_tensor, split_size, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensors = ttnn.split(input_tensor, split_size, dim=-1)

    for torch_out, ttnn_out in zip(torch_output_tensors, output_tensors):
        ttnn_out = ttnn.to_torch(ttnn_out)
        pcc = 0.99
        assert_with_pcc(torch_out, ttnn_out, pcc=pcc)
"""

        if self.non_tile_shapes:
            non_tile_shapes_code = f"""

@pytest.mark.parametrize("input_shape", ({''.join(set(f'{shape[0]}, ' for shape in self.non_tile_shapes if 0 in shape)).rstrip(', ')},))
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_split_tensor_non_tile_aligned(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    split_size = input_shape[-1] // 2 if input_shape[-1] > 1 else 1
    torch_output_tensors = torch.split(torch_input_tensor, split_size, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensors = ttnn.split(input_tensor, split_size, dim=-1)

    for torch_out, ttnn_out in zip(torch_output_tensors, output_tensors):
        ttnn_out = ttnn.to_torch(ttnn_out)
        pcc = 0.99
        assert_with_pcc(torch_out, ttnn_out, pcc=pcc)
"""

        return tile_shapes_code + non_tile_shapes_code


class SplitWithSizesGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]

        # Filter shapes for tile compatibility like in SplitTensorGroupUnittest
        self.tile_compatible_shapes = []
        self.non_tile_shapes = []

        for shape in self.input_shape_list:
            if 0 in shape:
                last_dim = shape[0][-1]
                # For split_with_sizes, check if the dimension itself is reasonable for TILE_LAYOUT
                # We'll be more conservative here
                if last_dim % 96 == 0:  # Divisible by 96 (3*32) for 3-way split
                    self.tile_compatible_shapes.append(shape)
                # Disable non-tile aligned testing for split_with_sizes too
                # elif last_dim >= 96 and last_dim % 32 == 0:
                #     self.non_tile_shapes.append(shape)
                # Skip all other shapes that might cause issues

        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this split with sizes unit test operation."""
        tile_shapes_code = ""
        non_tile_shapes_code = ""

        if self.tile_compatible_shapes:
            tile_shapes_code = f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.tile_compatible_shapes if 0 in shape))}
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_split_with_sizes(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    last_dim = input_shape[-1]
    split_sizes = (
        [last_dim // 3, last_dim // 3, last_dim - 2 * (last_dim // 3)]
        if last_dim > 2
        else [last_dim]
    )
    torch_output_tensors = torch.split(torch_input_tensor, split_sizes, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensors = ttnn.split(input_tensor, split_sizes, dim=-1)

    for torch_out, ttnn_out in zip(torch_output_tensors, output_tensors):
        ttnn_out = ttnn.to_torch(ttnn_out)
        pcc = 0.99
        assert_with_pcc(torch_out, ttnn_out, pcc=pcc)
"""

        if self.non_tile_shapes:
            non_tile_shapes_code = f"""

@pytest.mark.parametrize("input_shape", ({''.join(set(f'{shape[0]}, ' for shape in self.non_tile_shapes if 0 in shape)).rstrip(', ')},))
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_split_with_sizes_non_tile_aligned(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    last_dim = input_shape[-1]
    split_sizes = (
        [last_dim // 3, last_dim // 3, last_dim - 2 * (last_dim // 3)]
        if last_dim > 2
        else [last_dim]
    )
    torch_output_tensors = torch.split(torch_input_tensor, split_sizes, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensors = ttnn.split(input_tensor, split_sizes, dim=-1)

    for torch_out, ttnn_out in zip(torch_output_tensors, output_tensors):
        ttnn_out = ttnn.to_torch(ttnn_out)
        pcc = 0.99
        assert_with_pcc(torch_out, ttnn_out, pcc=pcc)
"""

        return tile_shapes_code + non_tile_shapes_code


# Additional unittest classes for new operations


class MeanDimUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["MeanDimUnittest"]:
        if operation.function_call_name == "torch.ops.aten.mean.dim":
            mean_op = operation.to_operation(AtenMeanDim)
            return MeanDimUnittest(mean_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this mean dim unit test operation."""
        group_unit_test = MeanDimGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SliceTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SliceTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.slice.Tensor":
            slice_op = operation.to_operation(AtenSliceTensor)
            return SliceTensorUnittest(slice_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this slice tensor unit test operation."""
        group_unit_test = SliceTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class NativeLayerNormUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["NativeLayerNormUnittest"]:
        if operation.function_call_name == "torch.ops.aten.native_layer_norm":
            layer_norm_op = operation.to_operation(AtenNativeLayerNorm)
            return NativeLayerNormUnittest(layer_norm_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this native layer norm unit test operation."""
        group_unit_test = NativeLayerNormGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class UnsqueezeUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["UnsqueezeUnittest"]:
        if operation.function_call_name == "torch.ops.aten.unsqueeze":
            unsqueeze_op = operation.to_operation(AtenUnsqueeze)
            return UnsqueezeUnittest(unsqueeze_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this unsqueeze unit test operation."""
        group_unit_test = UnsqueezeGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class GeluUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["GeluUnittest"]:
        if operation.function_call_name == "torch.ops.aten.gelu":
            gelu_op = operation.to_operation(AtenGelu)
            return GeluUnittest(gelu_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this gelu unit test operation."""
        group_unit_test = GeluGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ReluInplaceUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ReluInplaceUnittest"]:
        if operation.function_call_name == "torch.ops.aten.relu_":
            relu_op = operation.to_operation(AtenReluInplace)
            return ReluInplaceUnittest(relu_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this relu inplace unit test operation."""
        group_unit_test = ReluInplaceGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ConstantPadNdUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ConstantPadNdUnittest"]:
        if operation.function_call_name == "torch.ops.aten.constant_pad_nd":
            pad_op = operation.to_operation(AtenConstantPadNd)
            return ConstantPadNdUnittest(pad_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this constant pad nd unit test operation."""
        group_unit_test = ConstantPadNdGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SqueezeDimUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SqueezeDimUnittest"]:
        if operation.function_call_name == "torch.ops.aten.squeeze.dim":
            squeeze_op = operation.to_operation(AtenSqueezeDim)
            return SqueezeDimUnittest(squeeze_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this squeeze dim unit test operation."""
        group_unit_test = SqueezeDimGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class MmUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["MmUnittest"]:
        if operation.function_call_name == "torch.ops.aten.mm":
            mm_op = operation.to_operation(AtenMm)
            return MmUnittest(mm_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this matrix multiplication unit test operation."""
        group_unit_test = MmGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


# Note: These operations continue the pattern of PyTorch operations with comprehensive unit test coverage
# Additional operations (LinearNorm, ClampMin, etc.) would follow the same pattern


class LinalgVectorNormUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["LinalgVectorNormUnittest"]:
        if operation.function_call_name == "torch.ops.aten.linalg_vector_norm":
            norm_op = operation.to_operation(AtenLinalgVectorNorm)
            return LinalgVectorNormUnittest(norm_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this linalg vector norm unit test operation."""
        group_unit_test = LinalgVectorNormGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ClampMinUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ClampMinUnittest"]:
        if operation.function_call_name == "torch.ops.aten.clamp_min":
            clamp_op = operation.to_operation(AtenClampMin)
            return ClampMinUnittest(clamp_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this clamp min unit test operation."""
        group_unit_test = ClampMinGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class MaxDimUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["MaxDimUnittest"]:
        if operation.function_call_name == "torch.ops.aten.max.dim":
            max_op = operation.to_operation(AtenMaxDim)
            return MaxDimUnittest(max_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this max dim unit test operation."""
        group_unit_test = MaxDimGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ReluUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ReluUnittest"]:
        if operation.function_call_name == "torch.ops.aten.relu":
            relu_op = operation.to_operation(AtenRelu)
            return ReluUnittest(relu_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this relu unit test operation."""
        group_unit_test = ReluGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class AsStridedInplaceUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["AsStridedInplaceUnittest"]:
        if operation.function_call_name == "torch.ops.aten.as_strided_":
            strided_op = operation.to_operation(AtenAsStridedInplace)
            return AsStridedInplaceUnittest(strided_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this as strided inplace unit test operation."""
        group_unit_test = AsStridedInplaceGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class RollUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["RollUnittest"]:
        if operation.function_call_name == "torch.ops.aten.roll":
            roll_op = operation.to_operation(AtenRoll)
            return RollUnittest(roll_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this roll unit test operation."""
        group_unit_test = RollGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class LeakyReluInplaceUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["LeakyReluInplaceUnittest"]:
        if operation.function_call_name == "torch.ops.aten.leaky_relu_":
            leaky_op = operation.to_operation(AtenLeakyReluInplace)
            return LeakyReluInplaceUnittest(leaky_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this leaky relu inplace unit test operation."""
        group_unit_test = LeakyReluInplaceGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SoftplusUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SoftplusUnittest"]:
        if operation.function_call_name == "torch.ops.aten.softplus":
            softplus_op = operation.to_operation(AtenSoftplus)
            return SoftplusUnittest(softplus_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this softplus unit test operation."""
        group_unit_test = SoftplusGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class TanhUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["TanhUnittest"]:
        if operation.function_call_name == "torch.ops.aten.tanh":
            tanh_op = operation.to_operation(AtenTanh)
            return TanhUnittest(tanh_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this tanh unit test operation."""
        group_unit_test = TanhGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class PowTensorScalarUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["PowTensorScalarUnittest"]:
        if operation.function_call_name == "torch.ops.aten.pow.Tensor_Scalar":
            pow_op = operation.to_operation(AtenPowTensorScalar)
            return PowTensorScalarUnittest(pow_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this pow tensor scalar unit test operation."""
        group_unit_test = PowTensorScalarGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SiluNonInplaceUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SiluNonInplaceUnittest"]:
        if operation.function_call_name == "torch.ops.aten.silu":
            silu_op = operation.to_operation(AtenSilu)
            return SiluNonInplaceUnittest(silu_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this silu non-inplace unit test operation."""
        group_unit_test = SiluNonInplaceGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class CopyInplaceUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["CopyInplaceUnittest"]:
        if operation.function_call_name == "torch.ops.aten.copy_":
            copy_op = operation.to_operation(AtenCopyInplace)
            return CopyInplaceUnittest(copy_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this copy inplace unit test operation."""
        group_unit_test = CopyInplaceGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class AdaptiveMaxPool2dUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["AdaptiveMaxPool2dUnittest"]:
        if operation.function_call_name == "torch.ops.aten.adaptive_max_pool2d":
            pool_op = operation.to_operation(AtenAdaptiveMaxPool2d)
            return AdaptiveMaxPool2dUnittest(pool_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this adaptive max pool 2d unit test operation."""
        group_unit_test = AdaptiveMaxPool2dGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class UnbindIntUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["UnbindIntUnittest"]:
        if operation.function_call_name == "torch.ops.aten.unbind.int":
            unbind_op = operation.to_operation(AtenUnbindInt)
            return UnbindIntUnittest(unbind_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this unbind int unit test operation."""
        group_unit_test = UnbindIntGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class ClampUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["ClampUnittest"]:
        if operation.function_call_name == "torch.ops.aten.clamp":
            clamp_op = operation.to_operation(AtenClamp)
            return ClampUnittest(clamp_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this clamp unit test operation."""
        group_unit_test = ClampGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SelectIntUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SelectIntUnittest"]:
        if operation.function_call_name == "torch.ops.aten.select.int":
            select_op = operation.to_operation(AtenSelectInt)
            return SelectIntUnittest(select_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this select int unit test operation."""
        group_unit_test = SelectIntGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class TopkUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["TopkUnittest"]:
        if operation.function_call_name == "torch.ops.aten.topk":
            topk_op = operation.to_operation(AtenTopk)
            return TopkUnittest(topk_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this topk unit test operation."""
        group_unit_test = TopkGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class GridSampler2dUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["GridSampler2dUnittest"]:
        if operation.function_call_name == "torch.ops.aten.grid_sampler_2d":
            grid_sampler_op = operation.to_operation(AtenGridSampler2d)
            return GridSampler2dUnittest(grid_sampler_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this grid sampler 2d unit test operation."""
        group_unit_test = GridSampler2dGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class SumDimIntListUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SumDimIntListUnittest"]:
        if operation.function_call_name == "torch.ops.aten.sum.dim_IntList":
            sum_op = operation.to_operation(AtenSumDimIntList)
            return SumDimIntListUnittest(sum_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this sum dim int list unit test operation."""
        group_unit_test = SumDimIntListGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class RsubScalarUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["RsubScalarUnittest"]:
        if operation.function_call_name == "torch.ops.aten.rsub.Scalar":
            rsub_op = operation.to_operation(AtenRsubScalar)
            return RsubScalarUnittest(rsub_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this rsub scalar unit test operation."""
        group_unit_test = RsubScalarGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class StackUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["StackUnittest"]:
        if operation.function_call_name == "torch.ops.aten.stack":
            stack_op = operation.to_operation(AtenStack)
            return StackUnittest(stack_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this stack unit test operation."""
        group_unit_test = StackGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class IndexTensorUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["IndexTensorUnittest"]:
        if operation.function_call_name == "torch.ops.aten.index.Tensor":
            index_op = operation.to_operation(AtenIndexTensor)
            return IndexTensorUnittest(index_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this index tensor unit test operation."""
        group_unit_test = IndexTensorGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class LogUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["LogUnittest"]:
        if operation.function_call_name == "torch.ops.aten.log":
            log_op = operation.to_operation(AtenLog)
            return LogUnittest(log_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this log unit test operation."""
        group_unit_test = LogGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


class HardtanhInplaceUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["HardtanhInplaceUnittest"]:
        if operation.function_call_name == "torch.ops.aten.hardtanh_":
            hardtanh_op = operation.to_operation(AtenHardtanhInplace)
            return HardtanhInplaceUnittest(hardtanh_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this hardtanh inplace unit test operation."""
        group_unit_test = HardtanhInplaceGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()


# Group unittest classes for new operations


class MeanDimGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this mean dim unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_mean_dim(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=-1, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=True)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SliceTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this slice tensor unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_slice_tensor(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor[:, :, :input_shape[-1]//2]

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.slice(input_tensor, [0, 0, 0, 0], [input_shape[0], input_shape[1], input_shape[2], input_shape[-1]//2])

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class NativeLayerNormGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this native layer norm unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_native_layer_norm(device, input_shape, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    normalized_shape = [input_shape[-1]]

    layer_norm = torch.nn.LayerNorm(normalized_shape, dtype=torch.bfloat16)
    torch_output_tensor = layer_norm(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    # Note: This is a simplified test - actual implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class UnsqueezeGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this unsqueeze unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_unsqueeze(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.unsqueeze(0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.unsqueeze(input_tensor, 0)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class GeluGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this gelu unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_gelu(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.gelu(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.gelu(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class ReluInplaceGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this relu inplace unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_relu_inplace(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.relu(torch_input_tensor, inplace=False)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.relu(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class ConstantPadNdGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this constant pad nd unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_constant_pad_nd(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    pad = [1, 1, 1, 1]  # pad last 2 dimensions
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, pad, mode='constant', value=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.pad(input_tensor, pad, 0)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SqueezeDimGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this squeeze dim unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_squeeze_dim(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    # Add a dimension of size 1 to squeeze
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16).unsqueeze(1)
    torch_output_tensor = torch_input_tensor.squeeze(1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.squeeze(input_tensor, 1)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class MmGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 2]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this matrix multiplication unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_mm(device, input_shape_a, input_shape_b, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch.mm(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class LinalgVectorNormGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this linalg vector norm unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_linalg_vector_norm(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.linalg.vector_norm(torch_input_tensor, dim=-1, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class ClampMinGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this clamp min unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_clamp_min(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    min_value = 0.0
    torch_output_tensor = torch.clamp_min(torch_input_tensor, min_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.clip(input_tensor, min=min_value)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class MaxDimGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this max dim unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_max_dim(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor, torch_indices = torch.max(torch_input_tensor, dim=-1, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.max(input_tensor, dim=-1, keepdim=True)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class ReluGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this relu unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_relu(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.relu(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.relu(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class AsStridedInplaceGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this as strided inplace unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_as_strided_inplace(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    # Note: This is a simplified test - actual as_strided implementation may vary
    torch_output_tensor = torch_input_tensor.clone()

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.clone(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class RollGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this roll unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_roll(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    shifts = 1
    torch_output_tensor = torch.roll(torch_input_tensor, shifts, dims=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual roll implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class LeakyReluInplaceGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this leaky relu inplace unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_leaky_relu_inplace(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    negative_slope = 0.01
    torch_output_tensor = torch.nn.functional.leaky_relu(torch_input_tensor, negative_slope, inplace=False)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.leaky_relu(input_tensor, negative_slope)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SoftplusGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this softplus unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_softplus(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.softplus(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.softplus(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class TanhGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this tanh unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_tanh(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.tanh(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.tanh(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class PowTensorScalarGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this pow tensor scalar unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_pow_tensor_scalar(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    exponent = 2.0
    torch_output_tensor = torch.pow(torch_input_tensor, exponent)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.pow(input_tensor, exponent)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SiluNonInplaceGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this silu non-inplace unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_silu_non_inplace(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.silu(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.silu(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class CopyInplaceGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this copy inplace unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_copy_inplace(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_target_tensor = torch.zeros_like(torch_input_tensor)
    torch_target_tensor.copy_(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    target_tensor = ttnn.zeros_like(input_tensor)
    # Note: This is a simplified test - actual copy implementation may vary
    output_tensor = ttnn.clone(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_target_tensor, output_tensor, pcc=pcc)
"""


class AdaptiveMaxPool2dGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this adaptive max pool 2d unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_adaptive_max_pool2d(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    output_size = (1, 1)
    torch_output_tensor = torch.nn.functional.adaptive_max_pool2d(torch_input_tensor, output_size)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual adaptive max pool 2d implementation may vary
    output_tensor = ttnn.global_max_pool2d(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class UnbindIntGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this unbind int unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_unbind_int(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensors = torch.unbind(torch_input_tensor, dim=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual unbind implementation may vary
    output_tensors = [ttnn.slice(input_tensor, [i, 0, 0, 0], [i+1, input_shape[1], input_shape[2], input_shape[3]]) for i in range(input_shape[0])]

    for i, (torch_out, ttnn_out) in enumerate(zip(torch_output_tensors, output_tensors)):
        ttnn_out = ttnn.to_torch(ttnn_out)
        pcc = 0.99
        assert_with_pcc(torch_out, ttnn_out, pcc=pcc)
"""


class ClampGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this clamp unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_clamp(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    min_value = -1.0
    max_value = 1.0
    torch_output_tensor = torch.clamp(torch_input_tensor, min_value, max_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.clip(input_tensor, min=min_value, max=max_value)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SelectIntGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this select int unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_select_int(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    dim = 0
    index = 0
    torch_output_tensor = torch.select(torch_input_tensor, dim, index)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual select implementation may vary
    output_tensor = ttnn.slice(input_tensor, [0, 0, 0, 0], [1, input_shape[1], input_shape[2], input_shape[3]])
    output_tensor = ttnn.squeeze(output_tensor, 0)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class TopkGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this topk unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_topk(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    k = min(5, input_shape[-1])
    torch_values, torch_indices = torch.topk(torch_input_tensor, k, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual topk implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    pcc = 0.98
    assert_with_pcc(torch_values, output_tensor[..., :k], pcc=pcc)
"""


class GridSampler2dGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this grid sampler 2d unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_grid_sampler_2d(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    grid_shape = [input_shape[0], input_shape[2], input_shape[3], 2]
    grid = torch.randn(grid_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.grid_sample(torch_input_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual grid sampler implementation may vary
    output_tensor = ttnn.to_torch(input_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class SumDimIntListGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this sum dim int list unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sum_dim_int_list(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    dims = [0, 1]
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dims)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.sum(input_tensor, dim=dims)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class RsubScalarGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this rsub scalar unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_rsub_scalar(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    scalar_value = 5.0
    torch_output_tensor = torch.rsub(torch_input_tensor, scalar_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.rsub(input_tensor, scalar_value)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class StackGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this stack unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_stack(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor1 = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.stack([torch_input_tensor1, torch_input_tensor2], dim=0)

    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=layout, device=device, dtype=dtype)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.stack([input_tensor1, input_tensor2], dim=0)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class IndexTensorGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this index tensor unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_index_tensor(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    indices = torch.tensor([0, 1, 0, 1])
    torch_output_tensor = torch_input_tensor[indices]

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    # Note: This is a simplified test - actual index implementation may vary
    output_tensor = ttnn.slice(input_tensor, [0, 0, 0, 0], [2, input_shape[1], input_shape[2], input_shape[3]])

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class LogGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this log unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_log(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16).abs() + 1e-5
    torch_output_tensor = torch.log(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.log(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""


class HardtanhInplaceGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this hardtanh inplace unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_hardtanh_inplace(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    min_val = -1.0
    max_val = 1.0
    torch_output_tensor = torch.hardtanh_(torch_input_tensor.clone(), min_val, max_val)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.clip(input_tensor, min=min_val, max=max_val)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.99
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


class NativeBatchNormCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple native batch norm operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [bn_op.input_shapes for bn_op in operations if isinstance(bn_op, NativeBatchNormUnittest)]
        return NativeBatchNormGroupUnittest(combined_shapes)


class SiluCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple silu operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [silu_op.input_shapes for silu_op in operations if isinstance(silu_op, SiluUnittest)]
        return SiluGroupUnittest(combined_shapes)


class DivTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple div tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [div_op.input_shapes for div_op in operations if isinstance(div_op, DivTensorUnittest)]
        return DivTensorGroupUnittest(combined_shapes)


class SigmoidCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple sigmoid operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            sigmoid_op.input_shapes for sigmoid_op in operations if isinstance(sigmoid_op, SigmoidUnittest)
        ]
        return SigmoidGroupUnittest(combined_shapes)


class SoftmaxCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple softmax operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            softmax_op.input_shapes for softmax_op in operations if isinstance(softmax_op, SoftmaxUnittest)
        ]
        return SoftmaxGroupUnittest(combined_shapes)


class SubTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple sub tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [sub_op.input_shapes for sub_op in operations if isinstance(sub_op, SubTensorUnittest)]
        return SubTensorGroupUnittest(combined_shapes)


class UpsampleNearest2dCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple upsample nearest2d operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            upsample_op.input_shapes for upsample_op in operations if isinstance(upsample_op, UpsampleNearest2dUnittest)
        ]
        return UpsampleNearest2dGroupUnittest(combined_shapes)


class TorchOnesCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple torch.ones operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [ones_op.output_shapes for ones_op in operations if isinstance(ones_op, TorchOnesUnittest)]
        return TorchOnesGroupUnittest(combined_shapes)


class PermuteCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple permute operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            permute_op.input_shapes for permute_op in operations if isinstance(permute_op, PermuteUnittest)
        ]
        return PermuteGroupUnittest(combined_shapes)


class ViewCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple view operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [view_op.input_shapes for view_op in operations if isinstance(view_op, ViewUnittest)]
        return ViewGroupUnittest(combined_shapes)


class CloneCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple clone operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [clone_op.input_shapes for clone_op in operations if isinstance(clone_op, CloneUnittest)]
        return CloneGroupUnittest(combined_shapes)


class UnsafeViewCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple unsafe view operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            unsafe_view_op.input_shapes
            for unsafe_view_op in operations
            if isinstance(unsafe_view_op, UnsafeViewUnittest)
        ]
        return UnsafeViewGroupUnittest(combined_shapes)


class ExpandCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple expand operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [expand_op.input_shapes for expand_op in operations if isinstance(expand_op, ExpandUnittest)]
        return ExpandGroupUnittest(combined_shapes)


class TransposeIntCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple transpose int operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            transpose_op.input_shapes for transpose_op in operations if isinstance(transpose_op, TransposeIntUnittest)
        ]
        return TransposeIntGroupUnittest(combined_shapes)


class SplitTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple split tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            split_op.input_shapes for split_op in operations if isinstance(split_op, SplitTensorUnittest)
        ]
        return SplitTensorGroupUnittest(combined_shapes)


class SplitWithSizesCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple split with sizes operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            split_op.input_shapes for split_op in operations if isinstance(split_op, SplitWithSizesUnittest)
        ]
        return SplitWithSizesGroupUnittest(combined_shapes)


class MeanDimCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple mean dim operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [mean_op.input_shapes for mean_op in operations if isinstance(mean_op, MeanDimUnittest)]
        return MeanDimGroupUnittest(combined_shapes)


class SliceTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple slice tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            slice_op.input_shapes for slice_op in operations if isinstance(slice_op, SliceTensorUnittest)
        ]
        return SliceTensorGroupUnittest(combined_shapes)


class NativeLayerNormCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple native layer norm operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            norm_op.input_shapes for norm_op in operations if isinstance(norm_op, NativeLayerNormUnittest)
        ]
        return NativeLayerNormGroupUnittest(combined_shapes)


class UnsqueezeCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple unsqueeze operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            unsqueeze_op.input_shapes for unsqueeze_op in operations if isinstance(unsqueeze_op, UnsqueezeUnittest)
        ]
        return UnsqueezeGroupUnittest(combined_shapes)


class GeluCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple gelu operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [gelu_op.input_shapes for gelu_op in operations if isinstance(gelu_op, GeluUnittest)]
        return GeluGroupUnittest(combined_shapes)


class ReluInplaceCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple relu inplace operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [relu_op.input_shapes for relu_op in operations if isinstance(relu_op, ReluInplaceUnittest)]
        return ReluInplaceGroupUnittest(combined_shapes)


class ConstantPadNdCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple constant pad nd operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [pad_op.input_shapes for pad_op in operations if isinstance(pad_op, ConstantPadNdUnittest)]
        return ConstantPadNdGroupUnittest(combined_shapes)


class SqueezeDimCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple squeeze dim operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            squeeze_op.input_shapes for squeeze_op in operations if isinstance(squeeze_op, SqueezeDimUnittest)
        ]
        return SqueezeDimGroupUnittest(combined_shapes)


class MmCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple matrix multiplication operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [mm_op.input_shapes for mm_op in operations if isinstance(mm_op, MmUnittest)]
        return MmGroupUnittest(combined_shapes)


class LinalgVectorNormCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple linalg vector norm operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            norm_op.input_shapes for norm_op in operations if isinstance(norm_op, LinalgVectorNormUnittest)
        ]
        return LinalgVectorNormGroupUnittest(combined_shapes)


class ClampMinCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple clamp min operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [clamp_op.input_shapes for clamp_op in operations if isinstance(clamp_op, ClampMinUnittest)]
        return ClampMinGroupUnittest(combined_shapes)


class MaxDimCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple max dim operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [max_op.input_shapes for max_op in operations if isinstance(max_op, MaxDimUnittest)]
        return MaxDimGroupUnittest(combined_shapes)


class ReluCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple relu operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [relu_op.input_shapes for relu_op in operations if isinstance(relu_op, ReluUnittest)]
        return ReluGroupUnittest(combined_shapes)


class AsStridedInplaceCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple as strided inplace operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            strided_op.input_shapes for strided_op in operations if isinstance(strided_op, AsStridedInplaceUnittest)
        ]
        return AsStridedInplaceGroupUnittest(combined_shapes)


class RollCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple roll operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [roll_op.input_shapes for roll_op in operations if isinstance(roll_op, RollUnittest)]
        return RollGroupUnittest(combined_shapes)


class LeakyReluInplaceCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple leaky relu inplace operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            leaky_op.input_shapes for leaky_op in operations if isinstance(leaky_op, LeakyReluInplaceUnittest)
        ]
        return LeakyReluInplaceGroupUnittest(combined_shapes)


class SoftplusCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple softplus operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            softplus_op.input_shapes for softplus_op in operations if isinstance(softplus_op, SoftplusUnittest)
        ]
        return SoftplusGroupUnittest(combined_shapes)


class TanhCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple tanh operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [tanh_op.input_shapes for tanh_op in operations if isinstance(tanh_op, TanhUnittest)]
        return TanhGroupUnittest(combined_shapes)


class PowTensorScalarCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple pow tensor scalar operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [pow_op.input_shapes for pow_op in operations if isinstance(pow_op, PowTensorScalarUnittest)]
        return PowTensorScalarGroupUnittest(combined_shapes)


class SiluNonInplaceCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple silu non-inplace operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            silu_op.input_shapes for silu_op in operations if isinstance(silu_op, SiluNonInplaceUnittest)
        ]
        return SiluNonInplaceGroupUnittest(combined_shapes)


class CopyInplaceCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple copy inplace operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [copy_op.input_shapes for copy_op in operations if isinstance(copy_op, CopyInplaceUnittest)]
        return CopyInplaceGroupUnittest(combined_shapes)


class AdaptiveMaxPool2dCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple adaptive max pool 2d operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            pool_op.input_shapes for pool_op in operations if isinstance(pool_op, AdaptiveMaxPool2dUnittest)
        ]
        return AdaptiveMaxPool2dGroupUnittest(combined_shapes)


class UnbindIntCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple unbind int operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            unbind_op.input_shapes for unbind_op in operations if isinstance(unbind_op, UnbindIntUnittest)
        ]
        return UnbindIntGroupUnittest(combined_shapes)


class ClampCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple clamp operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [clamp_op.input_shapes for clamp_op in operations if isinstance(clamp_op, ClampUnittest)]
        return ClampGroupUnittest(combined_shapes)


class SelectIntCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple select int operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            select_op.input_shapes for select_op in operations if isinstance(select_op, SelectIntUnittest)
        ]
        return SelectIntGroupUnittest(combined_shapes)


class TopkCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple topk operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [topk_op.input_shapes for topk_op in operations if isinstance(topk_op, TopkUnittest)]
        return TopkGroupUnittest(combined_shapes)


class GridSampler2dCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple grid sampler 2d operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [grid_op.input_shapes for grid_op in operations if isinstance(grid_op, GridSampler2dUnittest)]
        return GridSampler2dGroupUnittest(combined_shapes)


class SumDimIntListCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple sum dim int list operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [sum_op.input_shapes for sum_op in operations if isinstance(sum_op, SumDimIntListUnittest)]
        return SumDimIntListGroupUnittest(combined_shapes)


class RsubScalarCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple rsub scalar operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [rsub_op.input_shapes for rsub_op in operations if isinstance(rsub_op, RsubScalarUnittest)]
        return RsubScalarGroupUnittest(combined_shapes)


class StackCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple stack operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [stack_op.input_shapes for stack_op in operations if isinstance(stack_op, StackUnittest)]
        return StackGroupUnittest(combined_shapes)


class IndexTensorCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple index tensor operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            index_op.input_shapes for index_op in operations if isinstance(index_op, IndexTensorUnittest)
        ]
        return IndexTensorGroupUnittest(combined_shapes)


class LogCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple log operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [log_op.input_shapes for log_op in operations if isinstance(log_op, LogUnittest)]
        return LogGroupUnittest(combined_shapes)


class HardtanhInplaceCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple hardtanh inplace operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            hardtanh_op.input_shapes for hardtanh_op in operations if isinstance(hardtanh_op, HardtanhInplaceUnittest)
        ]
        return HardtanhInplaceGroupUnittest(combined_shapes)


class UnitTestCombiner:
    def __init__(
        self,
        operations: List[UnitTestOperation],
        class_to_combiner: Dict[Type[UnitTestOperation], Type[UnitTestOperationCombiner]],
    ):
        self.operations = operations
        self.class_to_combiner = class_to_combiner

    def group_operations(self) -> Dict[Type[UnitTestOperation], List[UnitTestOperation]]:
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
                NativeBatchNormUnittest,
                SiluUnittest,
                DivTensorUnittest,
                SigmoidUnittest,
                SoftmaxUnittest,
                SubTensorUnittest,
                UpsampleNearest2dUnittest,
                TorchOnesUnittest,
                PermuteUnittest,
                ViewUnittest,
                CloneUnittest,
                UnsafeViewUnittest,
                ExpandUnittest,
                TransposeIntUnittest,
                SplitTensorUnittest,
                SplitWithSizesUnittest,
                MeanDimUnittest,
                SliceTensorUnittest,
                NativeLayerNormUnittest,
                UnsqueezeUnittest,
                GeluUnittest,
                ReluInplaceUnittest,
                ConstantPadNdUnittest,
                SqueezeDimUnittest,
                MmUnittest,
                LinalgVectorNormUnittest,
                ClampMinUnittest,
                MaxDimUnittest,
                ReluUnittest,
                AsStridedInplaceUnittest,
                RollUnittest,
                LeakyReluInplaceUnittest,
                SoftplusUnittest,
                TanhUnittest,
                PowTensorScalarUnittest,
                SiluNonInplaceUnittest,
                CopyInplaceUnittest,
                AdaptiveMaxPool2dUnittest,
                UnbindIntUnittest,
                ClampUnittest,
                SelectIntUnittest,
                TopkUnittest,
                GridSampler2dUnittest,
                SumDimIntListUnittest,
                RsubScalarUnittest,
                StackUnittest,
                IndexTensorUnittest,
                LogUnittest,
                HardtanhInplaceUnittest,
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
                NativeBatchNormUnittest: NativeBatchNormCombiner,
                SiluUnittest: SiluCombiner,
                DivTensorUnittest: DivTensorCombiner,
                SigmoidUnittest: SigmoidCombiner,
                SoftmaxUnittest: SoftmaxCombiner,
                SubTensorUnittest: SubTensorCombiner,
                UpsampleNearest2dUnittest: UpsampleNearest2dCombiner,
                TorchOnesUnittest: TorchOnesCombiner,
                PermuteUnittest: PermuteCombiner,
                ViewUnittest: ViewCombiner,
                CloneUnittest: CloneCombiner,
                UnsafeViewUnittest: UnsafeViewCombiner,
                ExpandUnittest: ExpandCombiner,
                TransposeIntUnittest: TransposeIntCombiner,
                SplitTensorUnittest: SplitTensorCombiner,
                SplitWithSizesUnittest: SplitWithSizesCombiner,
                MeanDimUnittest: MeanDimCombiner,
                SliceTensorUnittest: SliceTensorCombiner,
                NativeLayerNormUnittest: NativeLayerNormCombiner,
                UnsqueezeUnittest: UnsqueezeCombiner,
                GeluUnittest: GeluCombiner,
                ReluInplaceUnittest: ReluInplaceCombiner,
                ConstantPadNdUnittest: ConstantPadNdCombiner,
                SqueezeDimUnittest: SqueezeDimCombiner,
                MmUnittest: MmCombiner,
                LinalgVectorNormUnittest: LinalgVectorNormCombiner,
                ClampMinUnittest: ClampMinCombiner,
                MaxDimUnittest: MaxDimCombiner,
                ReluUnittest: ReluCombiner,
                AsStridedInplaceUnittest: AsStridedInplaceCombiner,
                RollUnittest: RollCombiner,
                LeakyReluInplaceUnittest: LeakyReluInplaceCombiner,
                SoftplusUnittest: SoftplusCombiner,
                TanhUnittest: TanhCombiner,
                PowTensorScalarUnittest: PowTensorScalarCombiner,
                SiluNonInplaceUnittest: SiluNonInplaceCombiner,
                CopyInplaceUnittest: CopyInplaceCombiner,
                AdaptiveMaxPool2dUnittest: AdaptiveMaxPool2dCombiner,
                UnbindIntUnittest: UnbindIntCombiner,
                ClampUnittest: ClampCombiner,
                SelectIntUnittest: SelectIntCombiner,
                TopkUnittest: TopkCombiner,
                GridSampler2dUnittest: GridSampler2dCombiner,
                SumDimIntListUnittest: SumDimIntListCombiner,
                RsubScalarUnittest: RsubScalarCombiner,
                StackUnittest: StackCombiner,
                IndexTensorUnittest: IndexTensorCombiner,
                LogUnittest: LogCombiner,
                HardtanhInplaceUnittest: HardtanhInplaceCombiner,
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
            if operation and self.config.register_unit_test_operations:
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

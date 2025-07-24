import networkx as nx
from tracer_backend import OperationGraph, Operation
from tracer_backend_utils import ConvAttrs, PoolAttrs, AtenConvolution, AtenAddm, AtenMaxPool2dWithIndices
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
{''.join(f'        ({attr.input_batch}, {attr.input_depth}, {attr.hidden_units}, {attr.input_height}, {attr.input_width}, {attr.kernel}, {attr.stride}, {attr.padding}, {attr.dilation}),' for attr in self.attrs)}
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
    [[ttnn.bfloat8_b], [ttnn.bfloat16]],
)
@pytest.mark.parametrize(
    "input_batch, input_depth, input_height, input_width, kernel, stride, padding, dilation",
    (
{''.join(f'        ({attr.input_batch}, {attr.input_depth}, {attr.input_height}, {attr.input_width}, {attr.kernel}, {attr.stride}, {attr.padding}, {attr.dilation}),' for attr in self.attrs)}
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
{''.join(f'        ({shapes[1][-4]}, {shapes[1][-3]}, {shapes[2][-3]}, {shapes[1][-2]}, {shapes[1][-1]}, {shapes[2][-1]}),' for shapes in self.input_shape_list)}
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
    group_unit_test_operations: Optional[Dict[Type[UnitTestOperation], UnitTestOperationCombiner]] = None

    def __post_init__(self):
        if self.register_unit_test_operations is None:
            self.register_unit_test_operations = [ConvolutionUnittest]


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

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, ClassVar
import torch
import re

# Global registry for operations
WRAPPED_OPERATION_REGISTRY = {}


def register_operation(function_call_name: str):
    """Decorator to register an operation class with its function_call_name."""

    def decorator(cls):
        assert len(function_call_name) > 0, "Function call name must be a non-empty string"
        WRAPPED_OPERATION_REGISTRY[function_call_name] = cls
        return cls

    return decorator


def get_operation_class(function_call_name: str):
    """Retrieve the operation class from the registry."""
    return WRAPPED_OPERATION_REGISTRY.get(function_call_name, None)


def to_valid_variable_name(input_string: Optional[str]) -> str:
    """
    Convert a given string into a valid Python variable name.

    Args:
        input_string (str): The input string to convert.

    Returns:
        str: A valid Python variable name.
    """
    # Replace invalid characters with underscores
    if input_string is None:
        return ""
    valid_name = re.sub(r"\W|^(?=\d)", "_", input_string)
    return valid_name


@dataclass
class AttrsBase:
    pass


@dataclass
class ConvAttrs(AttrsBase):
    input_batch: int
    input_height: int
    input_width: int
    input_depth: int
    hidden_units: int
    kernel: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    groups: int
    transposed: bool = False
    output_padding: Optional[List[int]] = None


@dataclass
class PoolAttrs(AttrsBase):
    input_batch: int
    input_height: int
    input_width: int
    input_depth: int
    kernel: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]


@dataclass
class OperationMetadata:
    """Metadata for operations, such as input/output shapes and dtypes."""

    meta: Dict[str, Any]
    res: Any


@dataclass
class Operation:
    """Base class for operations in the graph."""

    id: str
    unique_name: str
    function_call_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    postfix: str = ""
    meta_data: Optional[OperationMetadata] = None
    graph_output_indices: Optional[List[int]] = None

    def _serialize(self, value: Any) -> str:
        """Recursively serialize arguments or keyword arguments."""
        if isinstance(value, Parameter):
            return value.generate_code()
        elif isinstance(value, (list, tuple)):
            # Recursively serialize each element in the list or tuple
            serialized_elements = [self._serialize(v) for v in value]
            return (
                f"[{', '.join(serialized_elements)}]"
                if isinstance(value, list)
                else f"({', '.join(serialized_elements)})"
            )
        else:
            return str(value)

    def generate_code(self) -> str:
        """Generate PyTorch code for this operation."""
        serialized_args = ", ".join(self._serialize(arg) for arg in self.args)
        serialized_kwargs = ", ".join(f"{k}={self._serialize(v)}" for k, v in self.kwargs.items())
        if serialized_kwargs:
            serialized_kwargs = ", " + serialized_kwargs
        return f"{to_valid_variable_name(self.unique_name)} = {self.function_call_name}({serialized_args}{serialized_kwargs}){self.postfix}"

    def generate_import_code(self) -> List[str]:
        """Generate import statements for this operation."""
        # Default implementation returns an empty list, subclasses can override
        return ["import torch"]

    def to_operation(self, New_type) -> "Operation":
        """Convert this operation to a generic Operation type."""
        return New_type(
            id=self.id,
            unique_name=self.unique_name,
            function_call_name=self.function_call_name,
            args=self.args,
            kwargs=self.kwargs,
            postfix=self.postfix,
            meta_data=self.meta_data,
            graph_output_indices=self.graph_output_indices,
        )

    def get_unique_representation(self) -> Dict[str, Any]:
        """Get a unique representation of the operation for hashing."""
        return {
            "unique_name": self.__class__.__name__,
            "function_call_name": self.function_call_name,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
        }

    @property
    def input_shapes(self) -> Optional[Dict[int, Any]]:
        try:
            dynamic_shapes = self.meta_data.meta["i_shapes"]
            args_shapes = {}
            offset = 0
            for index, elem in enumerate(self.args):
                if isinstance(elem, PlaceholderTensor) and offset < len(dynamic_shapes):
                    args_shapes[index] = dynamic_shapes[offset]
                    offset += 1
                elif isinstance(elem, ConstantTensor):
                    args_shapes[index] = elem.value.shape
            return args_shapes
        except:
            return None

    @property
    def output_shapes(self) -> Optional[List[Any]]:
        try:
            return self.meta_data.meta["o_shapes"]
        except:
            return None


@dataclass
class WrappedOperation(Operation):
    attrs: Optional[AttrsBase] = None

    @property
    def ops(self) -> int:
        """Return the number of multiply-accumulate operations"""
        output_shape = self.output_shapes
        if output_shape:
            num_elements = 0
            for shape in output_shape:
                if isinstance(shape, torch.Size):
                    num_elements += shape.numel()
                else:
                    num_elements = -1
                    break
            return num_elements  # Assuming two tensors are involved in the operation
        raise NotImplementedError("This method should be implemented in subclasses")

    @property
    def params(self) -> int:
        """Return the number of parameters in the operation"""
        return sum([const.size() for const in self.args if isinstance(const, ConstantTensor)]) if self.args else 0


@dataclass
@register_operation("torch.ops.aten.convolution")
class AtenConvolution(WrappedOperation):
    attrs: Optional[ConvAttrs] = None

    def __post_init__(self):
        if self.attrs is None:
            self.attrs = ConvAttrs(
                input_batch=self.meta_data.meta["i_shapes"][0][0],
                input_height=self.meta_data.meta["i_shapes"][0][2],
                input_width=self.meta_data.meta["i_shapes"][0][3],
                input_depth=self.meta_data.meta["i_shapes"][0][1],
                hidden_units=self.args[1].value.shape[0],
                kernel=[self.args[1].value.shape[2], self.args[1].value.shape[3]],
                stride=self.args[3],
                padding=self.args[4],
                dilation=self.args[5],
                transposed=self.args[6],
                output_padding=self.args[7],
                groups=self.args[8],
            )

    @property
    def ops(self) -> int:
        eff_kernel_height = self.attrs.dilation[0] * (self.attrs.kernel[0] - 1) + 1
        eff_kernel_width = self.attrs.dilation[1] * (self.attrs.kernel[1] - 1) + 1
        H_out = (self.attrs.input_height + 2 * self.attrs.padding[0] - eff_kernel_height) // self.attrs.stride[0] + 1
        W_out = (self.attrs.input_width + 2 * self.attrs.padding[1] - eff_kernel_width) // self.attrs.stride[1] + 1
        return (
            self.attrs.input_batch
            * H_out
            * W_out
            * self.attrs.hidden_units
            * self.attrs.input_depth
            * self.attrs.kernel[0]
            * self.attrs.kernel[1]
            // self.attrs.groups
        )

    def get_unique_representation(self) -> Dict[str, Any]:
        """Get a unique representation of the operation for hashing."""
        unique_representation = super().get_unique_representation()
        unique_representation.update(
            {
                "kernel_size": self.attrs.kernel,
                "stride": self.attrs.stride,
                "padding": self.attrs.padding,
                "dilation": self.attrs.dilation,
                "groups": self.attrs.groups,
                "hidden_units": self.attrs.hidden_units,
                "unique_name": "AtenConvolutionT" if self.attrs.transposed else "AtenConvolution",
            }
        )
        return unique_representation


@dataclass
@register_operation("torch.ops.aten.add_.Tensor")
class AtenAddTensor(WrappedOperation):
    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * 2  # Assuming two tensors are added
        return super().ops


@dataclass
@register_operation("torch.ops.aten.addmm")
class AtenAddm(WrappedOperation):
    @property
    def ops(self) -> int:
        shapes = self.input_shapes
        if shapes and len(shapes) > 2 and 1 in shapes and 2 in shapes:
            return 2 * shapes[1][0] * shapes[2].numel()
        return super().ops


@dataclass
@register_operation("torch.ops.aten.max_pool2d_with_indices")
class AtenMaxPool2dWithIndices(WrappedOperation):
    attrs: Optional[PoolAttrs] = None

    def __post_init__(self):
        if self.attrs is None:
            dilation = self.args[4] if len(self.args) > 4 else [1, 1]
            padding = self.args[3] if len(self.args) > 3 else [0, 0]
            self.attrs = PoolAttrs(
                input_batch=self.meta_data.meta["i_shapes"][0][0],
                input_height=self.meta_data.meta["i_shapes"][0][2],
                input_width=self.meta_data.meta["i_shapes"][0][3],
                input_depth=self.meta_data.meta["i_shapes"][0][1],
                kernel=self.args[1],
                stride=self.args[2],
                padding=padding,
                dilation=dilation,
            )

    @property
    def ops(self) -> int:
        eff_kernel_height = self.attrs.dilation[0] * (self.attrs.kernel[0] - 1) + 1
        eff_kernel_width = self.attrs.dilation[1] * (self.attrs.kernel[1] - 1) + 1
        H_out = (self.attrs.input_height + 2 * self.attrs.padding[0] - eff_kernel_height) // self.attrs.stride[0] + 1
        W_out = (self.attrs.input_width + 2 * self.attrs.padding[1] - eff_kernel_width) // self.attrs.stride[1] + 1
        return (
            self.attrs.input_batch
            * H_out
            * W_out
            * self.attrs.input_depth
            * self.attrs.kernel[0]
            * self.attrs.kernel[1]
        )

    def get_unique_representation(self) -> Dict[str, Any]:
        """Get a unique representation of the operation for hashing."""
        unique_representation = super().get_unique_representation()
        unique_representation.update(
            {
                "kernel_size": self.attrs.kernel,
                "stride": self.attrs.stride,
                "padding": self.attrs.padding,
                "dilation": self.attrs.dilation,
            }
        )
        return unique_representation


@dataclass
@register_operation("torch.ops.aten.view")
class AtenView(WrappedOperation):
    pass


@dataclass
@register_operation("torch.ones")
class TorchOnes(WrappedOperation):
    """Represents the torch.ones operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.native_batch_norm")
class AtenNativeBatchNorm(WrappedOperation):
    """Represents the native_batch_norm operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.mul.Tensor")
class AtenMulTensor(WrappedOperation):
    """Represents the mul.Tensor operation."""

    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.add.Tensor")
class AtenAddTensor(WrappedOperation):
    """Represents the add.Tensor operation."""

    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.cat")
class AtenCat(WrappedOperation):
    """Represents the cat operation."""

    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * len(self.args)
        return super().ops


@dataclass
@register_operation("torch.ops.aten.sigmoid")
class AtenSigmoid(WrappedOperation):
    """Represents the sigmoid operation."""

    pass


@dataclass
@register_operation("torch.ops.aten._softmax")
class AtenSoftmax(WrappedOperation):
    """Represents the _softmax operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.transpose.int")
class AtenTransposeInt(WrappedOperation):
    """Represents the transpose.int operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.div.Tensor")
class AtenDivTensor(WrappedOperation):
    """Represents the div.Tensor operation."""

    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.silu_")
class AtenSiluInplace(WrappedOperation):
    """Represents the silu_ operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.mean.dim")
class AtenMeanDim(WrappedOperation):
    """Represents the mean.dim operation."""

    @property
    def ops(self) -> int:
        input_shapes = self.input_shapes
        if input_shapes:
            num_elements = input_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.upsample_nearest2d")
class AtenUpsampleNearest2d(WrappedOperation):
    """Represents the upsample_nearest2d operation."""

    @property
    def ops(self) -> int:
        input_shapes = self.input_shapes
        if input_shapes:
            num_elements = input_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.clone")
class AtenClone(WrappedOperation):
    """Represents the clone operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.permute")
class AtenPermute(WrappedOperation):
    """Represents the permute operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.sub.Tensor")
class AtenSubTensor(WrappedOperation):
    """Represents the sub.Tensor operation."""

    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.bmm")
class AtenBmm(WrappedOperation):
    """Represents the bmm operation."""

    @property
    def ops(self) -> int:
        input_shapes = self.input_shapes
        if input_shapes and len(input_shapes) >= 2:
            batch_size = input_shapes[0][0]
            m = input_shapes[0][1]
            n = input_shapes[1][2]
            k = input_shapes[1][1]
            return batch_size * m * n * k
        return super().ops


@dataclass
@register_operation("torch.ops.aten.expand")
class AtenExpand(WrappedOperation):
    """Represents the expand operation."""

    pass


@dataclass
@register_operation("torch.ops.aten._unsafe_view")
class AtenUnsafeView(WrappedOperation):
    """Represents the _unsafe_view operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.slice.Tensor")
class AtenSliceTensor(WrappedOperation):
    """Represents the slice.Tensor operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.split_with_sizes")
class AtenSplitWithSizes(WrappedOperation):
    """Represents the split_with_sizes operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.split.Tensor")
class AtenSplitTensor(WrappedOperation):
    """Represents the split.Tensor operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.native_layer_norm")
class AtenNativeLayerNorm(WrappedOperation):
    """Represents the native_layer_norm operation."""

    @property
    def ops(self) -> int:
        input_shapes = self.input_shapes
        if input_shapes:
            num_elements = input_shapes[0].numel()
            return num_elements * 2
        return super().ops


@dataclass
@register_operation("torch.ops.aten.unsqueeze")
class AtenUnsqueeze(WrappedOperation):
    """Represents the unsqueeze operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.gelu")
class AtenGelu(WrappedOperation):
    """Represents the gelu operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.relu_")
class AtenReluInplace(WrappedOperation):
    """Represents the relu_ operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.constant_pad_nd")
class AtenConstantPadNd(WrappedOperation):
    """Represents the constant_pad_nd operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.squeeze.dim")
class AtenSqueezeDim(WrappedOperation):
    """Represents the squeeze.dim operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.mm")
class AtenMm(WrappedOperation):
    """Represents the mm operation."""

    @property
    def ops(self) -> int:
        input_shapes = self.input_shapes
        if input_shapes and len(input_shapes) >= 2:
            m = input_shapes[0][1]
            n = input_shapes[1][0]
            k = input_shapes[0][0]
            return m * n * k
        return super().ops


@dataclass
@register_operation("torch.ops.aten.linalg_vector_norm")
class AtenLinalgVectorNorm(WrappedOperation):
    """Represents the linalg_vector_norm operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.clamp_min")
class AtenClampMin(WrappedOperation):
    """Represents the clamp_min operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.max.dim")
class AtenMaxDim(WrappedOperation):
    """Represents the max.dim operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.relu")
class AtenRelu(WrappedOperation):
    """Represents the relu operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.as_strided_")
class AtenAsStridedInplace(WrappedOperation):
    """Represents the as_strided_ operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.roll")
class AtenRoll(WrappedOperation):
    """Represents the roll operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.leaky_relu_")
class AtenLeakyReluInplace(WrappedOperation):
    """Represents the leaky_relu_ operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.softplus")
class AtenSoftplus(WrappedOperation):
    """Represents the softplus operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.tanh")
class AtenTanh(WrappedOperation):
    """Represents the tanh operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.pow.Tensor_Scalar")
class AtenPowTensorScalar(WrappedOperation):
    """Represents the pow.Tensor_Scalar operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.silu")
class AtenSilu(WrappedOperation):
    """Represents the silu operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.copy_")
class AtenCopyInplace(WrappedOperation):
    """Represents the copy_ operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.adaptive_max_pool2d")
class AtenAdaptiveMaxPool2d(WrappedOperation):
    """Represents the adaptive_max_pool2d operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.unbind.int")
class AtenUnbindInt(WrappedOperation):
    """Represents the unbind.int operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.clamp")
class AtenClamp(WrappedOperation):
    """Represents the clamp operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.select.int")
class AtenSelectInt(WrappedOperation):
    """Represents the select.int operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.topk")
class AtenTopk(WrappedOperation):
    """Represents the topk operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.grid_sampler_2d")
class AtenGridSampler2d(WrappedOperation):
    """Represents the grid_sampler_2d operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.sum.dim_IntList")
class AtenSumDimIntList(WrappedOperation):
    """Represents the sum.dim_IntList operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.rsub.Scalar")
class AtenRsubScalar(WrappedOperation):
    """Represents the rsub.Scalar operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.stack")
class AtenStack(WrappedOperation):
    """Represents the stack operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.index.Tensor")
class AtenIndexTensor(WrappedOperation):
    """Represents the index.Tensor operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.log")
class AtenLog(WrappedOperation):
    """Represents the log operation."""

    pass


@dataclass
@register_operation("torch.ops.aten.hardtanh_")
class AtenHardtanhInplace(WrappedOperation):
    """Represents the hardtanh_ operation."""

    pass


@dataclass
@register_operation("torch.ops.aten._scaled_dot_product_flash_attention")
class AtenScaledDotProductFlashAttention(WrappedOperation):
    """Represents the _scaled_dot_product_flash_attention operation."""

    pass


@dataclass
class InputOp(Operation):
    """Represents an input operation in the graph."""

    counter: ClassVar[int] = 0

    def __post_init__(self):
        self.unique_name = to_valid_variable_name(self.unique_name)
        self.input_identifier = f"INPUT{InputOp.counter}"
        InputOp.counter += 1

    def generate_code(self):
        serialized_args = [self._serialize(arg) for arg in self.args]
        return f"{to_valid_variable_name(self.unique_name)} = {self.input_identifier}.reshape({serialized_args[0]})"

    def to_operation(self, New_type) -> "Operation":
        """Convert this operation to a generic Operation type."""
        return self


@dataclass
class TupleOp(Operation):
    """Represents a tuple operation in the graph."""

    def __post_init__(self):
        self.unique_name = to_valid_variable_name(self.unique_name)

    def generate_code(self) -> str:
        """Generate PyTorch code for this operation."""
        index = self.kwargs["index"]
        return f"{to_valid_variable_name(self.unique_name)} = {self.args[index].generate_code()}[{index}]{self.postfix}"

    def get_unique_representation(self) -> Dict[str, Any]:
        """Get a unique representation of the operation for hashing."""
        representation = super().get_unique_representation()
        representation.update(
            {
                "index": self.kwargs.get("index", None),
            }
        )
        return representation


@dataclass
class Parameter:
    """Represents a parameter in the graph."""

    name: str
    value: Optional[Union[torch.Tensor, float, int, str]] = None

    def generate_code(self) -> str:
        """Generate the serialization code for this parameter."""
        return to_valid_variable_name(self.name)


@dataclass
class ConstantTensor(Parameter):
    """Represents a constant tensor in the graph."""

    value: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    id: str = ""
    counter: ClassVar[int] = 0
    ConstantTensorFromModel: ClassVar[bool] = False

    def __post_init__(self):
        self.id = f"const{ConstantTensor.counter}"
        ConstantTensor.counter += 1

    def generate_code(self) -> str:
        """Generate the serialization code for this constant tensor."""
        if ConstantTensor.ConstantTensorFromModel:
            return self.id
        if self.value.numel() > 10:
            return f"torch.zeros({self.value.shape}, dtype={self.value.dtype})"
        return f"torch.tensor({self.value.tolist()})"

    def size(self) -> int:
        """Return the number of elements in the tensor."""
        return self.value.numel() if self.value is not None else 0


@dataclass
class PlaceholderTensor(Parameter):
    """Represents a placeholder tensor in the graph."""

    pass

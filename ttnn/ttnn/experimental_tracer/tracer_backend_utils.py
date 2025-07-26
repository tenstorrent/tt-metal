from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import torch
import re


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

    def to_operation(self, New_type) -> "Operation":
        """Convert this operation to a generic Operation type."""
        return New_type(
            unique_name=self.unique_name,
            function_call_name=self.function_call_name,
            args=self.args,
            kwargs=self.kwargs,
            postfix=self.postfix,
            meta_data=self.meta_data,
            graph_output_indices=self.graph_output_indices,
        )

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
    def output_shapes(self) -> List[Any]:
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
        raise NotImplementedError("This method should be implemented in subclasses")

    @property
    def params(self) -> int:
        """Return the number of parameters in the operation"""
        return sum([const.size() for const in self.args if isinstance(const, ConstantTensor)]) if self.args else 0


@dataclass
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


@dataclass
class AtenAddTensor(WrappedOperation):
    @property
    def ops(self) -> int:
        output_shapes = self.output_shapes
        if output_shapes:
            num_elements = output_shapes[0].numel()
            return num_elements * 2  # Assuming two tensors are added
        return super().ops


@dataclass
class AtenAddm(WrappedOperation):
    @property
    def ops(self) -> int:
        shapes = self.input_shapes
        if shapes and len(shapes) > 2 and 1 in shapes and 2 in shapes:
            return 2 * shapes[1][0] * shapes[2].numel()
        return super().ops


@dataclass
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


@dataclass
class InputOp(Operation):
    """Represents an input operation in the graph."""

    def __post_init__(self):
        self.unique_name = to_valid_variable_name(self.unique_name)


@dataclass
class TupleOp(Operation):
    """Represents a tuple operation in the graph."""

    def __post_init__(self):
        self.unique_name = to_valid_variable_name(self.unique_name)

    def generate_code(self) -> str:
        """Generate PyTorch code for this operation."""
        index = self.kwargs["index"]
        return f"{to_valid_variable_name(self.unique_name)} = {self.args[index].generate_code()}[{index}]{self.postfix}"


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

    def generate_code(self) -> str:
        """Generate the serialization code for this constant tensor."""
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

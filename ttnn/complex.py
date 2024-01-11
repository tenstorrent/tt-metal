# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import sys
import tt_lib as ttl
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
from typing import Union
from ttnn.tensor import (
    Tensor,
    has_storage_type_of,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    DEVICE_STORAGE_TYPE,
)
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
import torch
import torch.nn.functional as F

THIS_MODULE = sys.modules[__name__]

__all__ = []


class ComplexTensor:
    def __init__(self, real: Tensor, imag: Tensor):
        self.real = real
        self.imag = imag

    def as_ttl_complex_tensor(self):
        return ttl.tensor.complex_tensor(self.real.value, self.imag.value)

    @staticmethod
    def build(value: ttl.tensor.complex.ComplexTensor):
        return ComplexTensor(Tensor(value.real), Tensor(value.imag))

    def __add__(self, other: "ComplexTensor"):
        return ComplexTensor.build(ttl.tensor.complex_add(self.as_ttl_complex_tensor(), other.as_ttl_complex_tensor()))

    def __sub__(self, other: "ComplexTensor"):
        return ComplexTensor.build(ttl.tensor.complex_sub(self.as_ttl_complex_tensor(), other.as_ttl_complex_tensor()))

    def __mul__(self, other: "ComplexTensor"):
        return ComplexTensor.build(ttl.tensor.complex_mul(self.as_ttl_complex_tensor(), other.as_ttl_complex_tensor()))

    def __div__(self, other: "ComplexTensor"):
        return ComplexTensor.build(ttl.tensor.complex_div(self.as_ttl_complex_tensor(), other.as_ttl_complex_tensor()))


setattr(THIS_MODULE, "ComplexTensor", ComplexTensor)
__all__ += ["ComplexTensor"]


def register_ttl_complex_binary_function(name, ttl_complex_function, torch_function):
    def _torch_complex(c_input_tensor_a: ComplexTensor, c_input_tensor_b: ComplexTensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(c_input_tensor_a.real.value)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a_r = ttnn.to_torch(input_tensor_a)

        input_tensor_a = ttnn.from_device(c_input_tensor_a.imag.value)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a_i = ttnn.to_torch(input_tensor_a)

        input_tensor_a = input_tensor_a_r + 1j * input_tensor_a_i

        input_tensor_b = ttnn.from_device(c_input_tensor_b.real.value)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b_r = ttnn.to_torch(input_tensor_b)

        input_tensor_b = ttnn.from_device(c_input_tensor_b.imag.value)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b_i = ttnn.to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b_r + 1j * input_tensor_b_i

        assert torch_function, f"Torch function not implemented for {str(ttl_complex_function)}"
        return torch_function(input_tensor_a, input_tensor_b)

    @decorate_operation(torch_function=_torch_complex, name=name)
    def complex_function(
        input_tensor_a: ComplexTensor,
        input_tensor_b: ComplexTensor,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_tensor_a: ComplexTensor, input_tensor_b: ComplexTensor) -> ComplexTensor
        Applies {name} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.
        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)
        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
        Example::
            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b)
            >>> print(output)
            Tensor([ 1, 0], dtype=bfloat16 )
        """
        original_shape = input_tensor_a.real.shape

        input_tensor_a_r = _reshape_to_4D(input_tensor_a.real)
        input_tensor_a_i = _reshape_to_4D(input_tensor_a.imag)
        input_tensor_a = ComplexTensor(input_tensor_a_r, input_tensor_a_i)

        input_tensor_b_r = _reshape_to_4D(input_tensor_b.real)
        input_tensor_b_i = _reshape_to_4D(input_tensor_b.imag)
        input_tensor_b = ComplexTensor(input_tensor_b_r, input_tensor_b_i)

        if not isinstance(input_tensor_a, ComplexTensor) or not isinstance(input_tensor_b, ComplexTensor):
            raise TypeError("Expected both arguments to be a ttnn.ComplexTensor")

        if not has_storage_type_of(input_tensor_a.real, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b.real, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.as_ttl_complex_tensor()
        ttl_input_tensor_b = input_tensor_b.as_ttl_complex_tensor()

        ttl_output_tensor = ttl_complex_function(
            ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config
        )

        output_tensor_r = Tensor(ttl_output_tensor.real)
        output_tensor_r = reshape(output_tensor_r, original_shape)

        output_tensor_i = Tensor(ttl_output_tensor.imag)
        output_tensor_i = reshape(output_tensor_i, original_shape)

        output_tensor = ComplexTensor(output_tensor_r, output_tensor_i)
        return output_tensor

    setattr(THIS_MODULE, name, complex_function)
    __all__.append(name)
    return complex_function


def register_ttl_complex_unary_function(name, ttl_complex_function, torch_function):
    def _torch_complex(input_tensor: Tensor, **_):
        import torch
        import ttnn

        input_tensor = ttnn.from_device(input_tensor)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.to_torch(input_tensor)
        assert torch_function, f"Torch function not implemented for {str(ttl_complex_function)}"
        return torch_function(input_tensor)

    @decorate_operation(torch_function=_torch_complex, name=name)
    def complex_function(input_tensor_a: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
        f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor)
            >>> print(output)
            Tensor([ 0, 2], dtype=bfloat16 )

        """

        original_shape = input_tensor_a.real.shape

        input_tensor_a_r = _reshape_to_4D(input_tensor_a.real)
        input_tensor_a_i = _reshape_to_4D(input_tensor_a.imag)
        input_tensor_a = ComplexTensor(input_tensor_a_r, input_tensor_a_i)

        if not isinstance(input_tensor_a, ComplexTensor):
            raise TypeError("Expected both arguments to be a ttnn.ComplexTensor")

        if not has_storage_type_of(input_tensor_a.real, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.as_ttl_complex_tensor()

        ttl_output_tensor = ttl_complex_function(ttl_input_tensor_a, output_mem_config=memory_config)

        if isinstance(ttl_output_tensor, (ttl.tensor.Tensor,)):
            output_tensor = Tensor(ttl_output_tensor)
        else:
            output_tensor_r = Tensor(ttl_output_tensor.real)
            output_tensor_r = reshape(output_tensor_r, original_shape)

            output_tensor_i = Tensor(ttl_output_tensor.imag)
            output_tensor_i = reshape(output_tensor_i, original_shape)

            output_tensor = ComplexTensor(output_tensor_r, output_tensor_i)

        return output_tensor

    setattr(THIS_MODULE, name, complex_function)
    __all__.append(name)
    return complex_function


# reference
def torch_cadd(a, b):
    return a + b


def torch_csub(a, b):
    return a - b


def torch_cmul(a, b):
    return a * b


def torch_cdiv(a, b):
    return a / b


def torch_cabs(a):
    return abs(a)


def torch_crecip(a):
    return 1.0 / a


def torch_real(a):
    return a.real


def torch_imag(a):
    return a.imag


def torch_is_real(a):
    return a.imag == 0


def torch_is_imag(a):
    return a.real == 0


def torch_angle(a):
    return a.angle()


def torch_conj(a):
    return a.conj()


# register functions
TTL_COMPLEX_BINARY_FUNCTIONS = [
    ("complex_add", ttl.tensor.complex_add, torch_cadd),
    ("complex_sub", ttl.tensor.complex_sub, torch_csub),
    ("complex_mul", ttl.tensor.complex_mul, torch_cmul),
    ("complex_div", ttl.tensor.complex_div, torch_cdiv),
]

for complex_function_name, ttl_complex_function, torch_function in TTL_COMPLEX_BINARY_FUNCTIONS:
    register_ttl_complex_binary_function(complex_function_name, ttl_complex_function, torch_function)


TTL_COMPLEX_UNARY_FUNCTIONS = [
    ("complex_abs", ttl.tensor.complex_abs, torch_cabs),
    ("complex_recip", ttl.tensor.complex_recip, torch_crecip),
    ("complex_real", ttl.tensor.real, torch_real),
    ("complex_imag", ttl.tensor.imag, torch_imag),
    ("complex_angle", ttl.tensor.angle, torch_angle),
    ("complex_conj", ttl.tensor.conj, torch_conj),
    ("complex_polar", ttl.tensor.polar, torch.polar),
    ("complex_is_real", ttl.tensor.is_real, torch_is_real),
    ("complex_is_imag", ttl.tensor.is_imag, torch_is_imag),
]

for complex_function_name, ttl_complex_function, torch_function in TTL_COMPLEX_UNARY_FUNCTIONS:
    register_ttl_complex_unary_function(complex_function_name, ttl_complex_function, torch_function)

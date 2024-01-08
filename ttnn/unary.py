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


def register_ttl_unary_function(name, ttl_unary_function, torch_function):
    def _torch_unary(input_tensor: Tensor, **_):
        import torch
        import ttnn

        input_tensor = ttnn.from_device(input_tensor)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.to_torch(input_tensor)
        assert torch_function, f"Torch function not implemented for {str(ttl_unary_function)}"
        return torch_function(input_tensor)

    @decorate_operation(torch_function=_torch_unary, name=name)
    def unary_function(input_tensor: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
        original_shape = input_tensor.shape
        input_tensor = _reshape_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{name}"
    unary_function.__doc__ = f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor)

        """
    setattr(THIS_MODULE, name, unary_function)
    __all__.append(name)
    return unary_function


def register_ttl_unary_function_with_float_parameter(name, ttl_unary_function, torch_function):
    def _torch_unary(input_tensor: Tensor, parameter, **_):
        import torch
        import ttnn

        input_tensor = ttnn.from_device(input_tensor)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.to_torch(input_tensor)
        assert torch_function, f"Torch function not implemented for {str(ttl_unary_function)}"
        return torch_function(input_tensor, parameter)

    @decorate_operation(torch_function=_torch_unary, name=name)
    def unary_function(
        input_tensor: Tensor, parameter: float, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG
    ) -> Tensor:
        original_shape = input_tensor.shape
        input_tensor = _reshape_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor, parameter, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{name}"
    unary_function.__doc__ = f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor, 2)

        """

    setattr(THIS_MODULE, name, unary_function)
    __all__.append(name)
    return unary_function


# register functions
def torch_cbrt(_t: torch.Tensor):
    return torch.sgn(_t) * torch.pow(torch.abs(_t), 1.0 / 3)


def torch_log_sigmoid(_t: torch.Tensor):
    return torch.log(torch.sigmoid(_t))


def ttl_pow(_t: ttl.tensor.Tensor, _exponent: Union[int, float], **kw_args):
    from math import floor

    if floor(_exponent) == _exponent:
        return ttl.tensor.power(_t, int(_exponent), **kw_args)
    return ttl.tensor.power_fp(_t, _exponent, **kw_args)


TTL_UNARY_FUNCTIONS = [
    ("exp", ttl.tensor.exp, torch.exp),
    ("tanh", ttl.tensor.tanh, torch.tanh),
    ("gelu", ttl.tensor.gelu, torch.nn.functional.gelu),
    ("relu", ttl.tensor.relu, torch.relu),
    ("rsqrt", ttl.tensor.rsqrt, torch.rsqrt),
    ("abs", ttl.tensor.abs, torch.abs),
    ("acos", ttl.tensor.acos, torch.acos),
    ("acosh", ttl.tensor.acosh, torch.acosh),
    ("asin", ttl.tensor.asin, torch.asin),
    ("asinh", ttl.tensor.asinh, torch.asinh),
    ("atan", ttl.tensor.acos, lambda _x: -_x.atan()),
    ("atanh", ttl.tensor.atanh, torch.atanh),
    ("cbrt", ttl.tensor.cbrt, torch_cbrt),
    ("clone", ttl.tensor.clone, torch.clone),
    ("cos", ttl.tensor.cos, torch.cos),
    ("cosh", ttl.tensor.cosh, torch.cosh),
    ("deg2rad", ttl.tensor.deg2rad, torch.deg2rad),
    ("digamma", ttl.tensor.digamma, torch.digamma),
    ("erf", ttl.tensor.erf, torch.erf),
    ("erfc", ttl.tensor.erfc, torch.erfc),
    ("erfinv", ttl.tensor.erfinv, torch.erfinv),
    ("exp", ttl.tensor.exp, torch.exp),
    ("exp2", ttl.tensor.exp2, torch.exp2),
    ("expm1", ttl.tensor.expm1, torch.expm1),
    # ("geglu", ttl.tensor.geglu, None),
    ("gelu", ttl.tensor.gelu, None),
    # ("glu", ttl.tensor.glu, F.glu),
    ("hardsigmoid", ttl.tensor.hardsigmoid, F.hardsigmoid),
    ("hardswish", ttl.tensor.hardswish, F.hardswish),
    ("hardtanh", ttl.tensor.hardtanh, F.hardtanh),
    ("i0", ttl.tensor.i0, torch.i0),
    ("identity", ttl.tensor.identity, None),
    ("isfinite", ttl.tensor.isfinite, torch.isfinite),
    ("isinf", ttl.tensor.isinf, torch.isinf),
    ("isnan", ttl.tensor.isnan, torch.isnan),
    ("isneginf", ttl.tensor.isneginf, torch.isneginf),
    ("isposinf", ttl.tensor.isposinf, torch.isposinf),
    ("lerp", ttl.tensor.lerp, torch.lerp),
    ("lgamma", ttl.tensor.lgamma, torch.lgamma),
    ("log", ttl.tensor.log, torch.log),
    ("logical_not_unary", ttl.tensor.logical_not_unary, torch.logical_not),
    ("log10", ttl.tensor.log10, torch.log10),
    ("log1p", ttl.tensor.log1p, torch.log1p),
    ("log2", ttl.tensor.log2, torch.log2),
    ("log_sigmoid", ttl.tensor.log_sigmoid, torch_log_sigmoid),
    # ("logaddexp", ttl.tensor.logaddexp, torch.logaddexp),
    # ("logaddexp2", ttl.tensor.logaddexp2, torch.logaddexp2),
    # ("max", ttl.tensor.max, torch.max),
    # ("min", ttl.tensor.min, torch.min),
    ("mish", ttl.tensor.mish, lambda _x: F.mish(_x.to(torch.float))),
    ("move", ttl.tensor.move, None),
    ("multigammaln", ttl.tensor.multigammaln, None),
    ("neg", ttl.tensor.neg, torch.neg),
    ("rad2deg", ttl.tensor.rad2deg, torch.rad2deg),
    ("recip", ttl.tensor.recip, lambda _x: 1.0 / _x),
    # ("reglu", ttl.tensor.reglu, None),
    ("relu", ttl.tensor.relu, torch.relu),
    ("relu6", ttl.tensor.relu6, F.relu6),
    ("rsqrt", ttl.tensor.rsqrt, torch.rsqrt),
    ("sigmoid", ttl.tensor.sigmoid, torch.sigmoid),
    ("sign", ttl.tensor.sign, torch.sign),
    # ("signbit", ttl.tensor.signbit, torch.signbit), #TODO: improve tt_dnn impl
    ("silu", ttl.tensor.silu, F.silu),
    ("sin", ttl.tensor.sin, torch.sin),
    ("sinh", ttl.tensor.sinh, torch.sinh),
    ("sqrt", ttl.tensor.sqrt, torch.sqrt),
    ("square", ttl.tensor.square, torch.square),
    # ("sum",ttl.tensor.sum,torch.sum),
    # ("swiglu", ttl.tensor.swiglu, None),
    ("swish", ttl.tensor.swish, F.hardswish),
    ("tan", ttl.tensor.tan, torch.tan),
    ("tanh", ttl.tensor.tanh, torch.tanh),
    ("tanhshrink", ttl.tensor.tanhshrink, F.tanhshrink),
    ("tril", ttl.tensor.tril, torch.tril),
    ("triu", ttl.tensor.triu, torch.triu),
]


for unary_function_name, ttl_unary_function, torch_function in TTL_UNARY_FUNCTIONS:
    register_ttl_unary_function(unary_function_name, ttl_unary_function, torch_function)


def torch_logical_andi(x, *args, **kwargs):
    value = kwargs.pop("immediate")
    result = torch.logical_and(x, torch.tensor(value, dtype=torch.int32))
    return result


def torch_logical_ori(x, *args, **kwargs):
    value = kwargs.pop("immediate")
    result = torch.logical_or(x, torch.tensor(value, dtype=torch.int32))
    return result


def torch_logical_xori(x, *args, **kwargs):
    value = kwargs.pop("immediate")
    result = torch.logical_xor(x, torch.tensor(value, dtype=torch.int32))
    return result


def torch_logical_noti(x, *args, **kwargs):
    immediate = kwargs.pop("immediate")
    result = torch.logical_not(torch.full_like(x, immediate)).to(torch.int32)
    return result


def torch_rpow(x, *args, **kwargs):
    dim = kwargs["factor"]
    return torch.pow(dim, x)


def torch_rsub(x, *args, **kwargs):
    dim = kwargs["factor"]
    return torch.sub(dim, x)


def torch_rdiv(x, *args, **kwargs):
    dim = kwargs["factor"]
    return dim / x


TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAMETER = [
    ("pow", ttl_pow, torch.pow),
    ("elu", ttl.tensor.elu, F.elu),
    ("relu_max", ttl.tensor.relu_max, None),
    ("relu_min", ttl.tensor.relu_min, None),
    ("threshold", ttl.tensor.threshold, torch.threshold),
    ("leaky_relu", ttl.tensor.leaky_relu, F.leaky_relu),
    ("hardshrink", ttl.tensor.hardshrink, F.hardshrink),
    ("prelu", ttl.tensor.prelu, torch.prelu),
    ("polygamma", ttl.tensor.polygamma, torch.polygamma),
    ("logit", ttl.tensor.logit, torch.logit),
    ("logical_ori", ttl.tensor.logical_ori, torch_logical_ori),
    ("logical_andi", ttl.tensor.logical_andi, torch_logical_andi),
    ("logical_xori", ttl.tensor.logical_xori, torch_logical_xori),
    ("logical_noti", ttl.tensor.logical_noti, torch_logical_noti),
    ("rdiv", ttl.tensor.rdiv, torch_rdiv),
    ("rsub", ttl.tensor.rsub, torch_rsub),
    ("rpow", ttl.tensor.rpow, torch_rpow),
]

for unary_function_name, ttl_unary_function, torch_function in TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAMETER:
    register_ttl_unary_function_with_float_parameter(unary_function_name, ttl_unary_function, torch_function)

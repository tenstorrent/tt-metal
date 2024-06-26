# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

import ttnn


THIS_MODULE = sys.modules[__name__]


def register_ttnn_cpp_unary_function(unary_function):
    import torch

    def torch_heaviside(x, *args, **kwargs):
        value = kwargs.pop("scalar")
        result = torch.heaviside(x, torch.tensor(value, dtype=x.dtype))
        return result

    def torch_cbrt(x, *args, **kwargs):
        return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3)

    def torch_multigammaln(x, *args, **kwargs):
        result = torch.lgamma(x)
        result += torch.lgamma(x - 0.5)
        result += torch.lgamma(x - 1.0)
        result += torch.lgamma(x - 1.5)
        result += 3.434189657547
        return result

    def torch_prelu(x, *args, **kwargs):
        weight = kwargs.pop("scalar")
        result = torch.nn.functional.prelu(x, torch.tensor(weight, dtype=x.dtype))
        return result

    name_to_golden_function = {
        "abs": torch.abs,
        "acos": torch.acos,
        "asin": torch.asin,
        "atan": torch.atan,
        "cos": torch.cos,
        "erfinv": torch.erfinv,
        "exp2": torch.exp2,
        "expm1": torch.expm1,
        "eqz": lambda x: torch.eq(x, 0),
        "gez": lambda x: torch.ge(x, 0),
        "gtz": lambda x: torch.gt(x, 0),
        "i0": torch.i0,
        "isfinite": torch.isfinite,
        "isinf": torch.inf,
        "isnan": torch.isnan,
        "isneginf": torch.isneginf,
        "isposinf": torch.isposinf,
        "lez": lambda x: torch.le(x, 0),
        "log": torch.log,
        "log10": torch.log10,
        "log2": torch.log2,
        "log_sigmoid": torch.nn.functional.logsigmoid,
        "logical_not": torch.logical_not,
        "ltz": lambda x: torch.lt(x, 0),
        "neg": torch.neg,
        "nez": lambda x: torch.ne(x, 0),
        "reciprocal": torch.reciprocal,
        "relu": torch.relu,
        "relu6": torch.nn.functional.relu6,
        "sigmoid": torch.sigmoid,
        "sign": torch.sign,
        "signbit": torch.signbit,
        "silu": torch.nn.functional.silu,
        "sin": torch.sin,
        "sqrt": torch.sqrt,
        "square": torch.square,
        "tan": torch.tan,
        "tanh": torch.tanh,
        # Unaries with fast_and_approximate_mode
        "exp": torch.exp,
        "erf": torch.erf,
        "erfc": torch.erfc,
        "gelu": torch.nn.functional.gelu,
        "rsqrt": torch.rsqrt,
        # Unaries with float parameter
        "elu": torch.nn.functional.elu,
        "heaviside": torch_heaviside,
        "leaky_relu": torch.nn.functional.leaky_relu,
        # "prelu": torch_prelu, # Alias for leaky_relu. TODO(#8544): implement PReLU properly
        # Other unaries (composite operations)
        "softplus": torch.nn.functional.softplus,
        "acosh": torch.acosh,
        "asinh": torch.asinh,
        "atanh": torch.atanh,
        "cbrt": torch_cbrt,
        "cosh": torch.cosh,
        "deg2rad": torch.deg2rad,
        "digamma": torch.digamma,
        "hardswish": torch.nn.functional.hardswish,
        "hardsigmoid": torch.nn.functional.hardsigmoid,
        "hardtanh": torch.nn.functional.hardtanh,
        "lgamma": torch.lgamma,
        "log1p": torch.log1p,
        "mish": lambda _x: torch.nn.functional.mish(_x.to(torch.float)),
        "multigammaln": torch_multigammaln,
        "rad2deg": torch.rad2deg,
        "sigmoid_accurate": torch.sigmoid,
        "sinh": torch.sinh,
        "softsign": torch.nn.functional.softsign,
        "swish": torch.nn.functional.hardswish,
        "tanhshrink": ttl.tensor.tanhshrink,
        "tril": torch.tril,
        "triu": torch.triu,
    }

    golden_keys = set(name_to_golden_function.keys())
    function_names = {function.name for function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS}
    if golden_keys != function_names:
        raise ImportError(f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}")

    def _golden_function(input_tensor: ttnn.Tensor, **_):
        torch_function = name_to_golden_function[unary_function.name]
        return torch_function(input_tensor)

    operation = ttnn.register_operation(golden_function=_golden_function)(unary_function)
    setattr(THIS_MODULE, unary_function.name, operation)


TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn._ttnn.operations.unary.abs,
    ttnn._ttnn.operations.unary.acos,
    ttnn._ttnn.operations.unary.asin,
    ttnn._ttnn.operations.unary.atan,
    ttnn._ttnn.operations.unary.cos,
    ttnn._ttnn.operations.unary.erfinv,
    ttnn._ttnn.operations.unary.exp2,
    ttnn._ttnn.operations.unary.expm1,
    ttnn._ttnn.operations.unary.eqz,
    ttnn._ttnn.operations.unary.gez,
    ttnn._ttnn.operations.unary.gtz,
    ttnn._ttnn.operations.unary.i0,
    ttnn._ttnn.operations.unary.isfinite,
    ttnn._ttnn.operations.unary.isinf,
    ttnn._ttnn.operations.unary.isnan,
    ttnn._ttnn.operations.unary.isneginf,
    ttnn._ttnn.operations.unary.isposinf,
    ttnn._ttnn.operations.unary.lez,
    ttnn._ttnn.operations.unary.log,
    ttnn._ttnn.operations.unary.log10,
    ttnn._ttnn.operations.unary.log2,
    ttnn._ttnn.operations.unary.logical_not,
    ttnn._ttnn.operations.unary.ltz,
    ttnn._ttnn.operations.unary.neg,
    ttnn._ttnn.operations.unary.nez,
    ttnn._ttnn.operations.unary.reciprocal,
    ttnn._ttnn.operations.unary.relu,
    ttnn._ttnn.operations.unary.relu6,
    ttnn._ttnn.operations.unary.sigmoid,
    ttnn._ttnn.operations.unary.sign,
    ttnn._ttnn.operations.unary.signbit,
    ttnn._ttnn.operations.unary.silu,
    ttnn._ttnn.operations.unary.sin,
    ttnn._ttnn.operations.unary.sqrt,
    ttnn._ttnn.operations.unary.square,
    ttnn._ttnn.operations.unary.tan,
    ttnn._ttnn.operations.unary.tanh,
    # Unaries with fast_and_approximate_mode
    ttnn._ttnn.operations.unary.exp,
    ttnn._ttnn.operations.unary.erf,
    ttnn._ttnn.operations.unary.erfc,
    ttnn._ttnn.operations.unary.gelu,
    ttnn._ttnn.operations.unary.rsqrt,
    # Unaries with float parameter
    ttnn._ttnn.operations.unary.elu,
    ttnn._ttnn.operations.unary.heaviside,
    ttnn._ttnn.operations.unary.leaky_relu,
    # ttnn._ttnn.operations.unary.prelu,  # Alias for leaky_relu. TODO(#8544): implement PReLU properly
    # Other unaries (composite operations)
    ttnn._ttnn.operations.unary.log_sigmoid,
    ttnn._ttnn.operations.unary.softplus,
    ttnn._ttnn.operations.unary.acosh,
    ttnn._ttnn.operations.unary.asinh,
    ttnn._ttnn.operations.unary.atanh,
    ttnn._ttnn.operations.unary.cbrt,
    ttnn._ttnn.operations.unary.cosh,
    ttnn._ttnn.operations.unary.deg2rad,
    ttnn._ttnn.operations.unary.digamma,
    ttnn._ttnn.operations.unary.hardswish,
    ttnn._ttnn.operations.unary.hardsigmoid,
    ttnn._ttnn.operations.unary.hardtanh,
    ttnn._ttnn.operations.unary.lgamma,
    ttnn._ttnn.operations.unary.log1p,
    ttnn._ttnn.operations.unary.mish,
    ttnn._ttnn.operations.unary.multigammaln,
    ttnn._ttnn.operations.unary.rad2deg,
    ttnn._ttnn.operations.unary.sigmoid_accurate,
    ttnn._ttnn.operations.unary.sinh,
    ttnn._ttnn.operations.unary.softsign,
    ttnn._ttnn.operations.unary.swish,
    ttnn._ttnn.operations.unary.tanhshrink,
    ttnn._ttnn.operations.unary.tril,
    ttnn._ttnn.operations.unary.triu,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_unary_function(unary_function)


def prelu(*args, **kwargs):  # Alias for leaky_relu. TODO(#8544): implement PReLU properly
    leaky_relu = getattr(THIS_MODULE, "leaky_relu")
    return leaky_relu(*args, **kwargs)


def _is_scalar(value):
    return isinstance(value, (int, float))


def register_ttl_unary_function_with_float(name, ttl_unary_function, param):
    def _golden_function(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_golden_function = {
            "logit": torch.logit,
            "polygamma": torch.special.polygamma,
        }
        torch_function = name_to_golden_function[name]
        return torch_function(input_tensor, parameter)

    def _unary_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint8, ttnn.uint16),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_unary_validate_input_tensors,
        golden_function=_golden_function,
    )
    def unary_function(
        input_tensor: ttnn.Tensor, parameter: float, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_unary_function(input_tensor, parameter, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(unary_function, ttnn.decorators.Operation):
        unary_function.__name__ = f"ttnn.{(name)}"
        unary_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Applies the {name} function to the elements of the input tensor :attr:`input_tensor` with :attr:`{param}` parameter.

            .. math::
                {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

            Args:
                * :attr:`input_tensor`
                * :attr:`{param}`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{(name)}(tensor, {param})

            """
    setattr(THIS_MODULE, name, unary_function)


TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("logit", ttl.tensor.logit, "eps"),  # composite
    ("polygamma", ttl.tensor.polygamma, "parameter"),  # composite
]

for unary_function_name, ttl_unary_function, param in TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAM:
    register_ttl_unary_function_with_float(unary_function_name, ttl_unary_function, param)


def _is_scalar(value):
    return isinstance(value, (int, float))


def register_ttl_activation_function_with_float(name, ttl_activation_function, param):
    def _golden_function(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_torch_function = {
            "hardshrink": torch.nn.functional.hardshrink,
            "softshrink": torch.nn.functional.softshrink,
            "tanhshrink": torch.nn.functional.tanhshrink,
        }
        torch_function = name_to_torch_function[name]

        if name == "heaviside" or name == "prelu":
            return torch_function(input_tensor, scalar=parameter)
        else:
            return torch_function(input_tensor, parameter)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        golden_function=_golden_function,
    )
    def activation_function(
        input_tensor: ttnn.Tensor, parameter: float, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, parameter, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(activation_function, ttnn.decorators.Operation):
        activation_function.__name__ = f"ttnn.{(name)}"
        activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Applies the {name} function to the elements of the input tensor :attr:`input_tensor` with :attr:`{param}` parameter.

            .. math::
                {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

            Args:
                * :attr:`input_tensor`
                * :attr:`{param}`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{(name)}(tensor, {param})

            """
    setattr(THIS_MODULE, name, activation_function)


TTL_ACTIVATION_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("hardshrink", ttl.tensor.hardshrink, "lambda"),  # composite
    ("celu", ttl.tensor.celu, "alpha"),  # composite
    ("softshrink", ttl.tensor.softshrink, "lambda"),  # composite
]

for activation_function_name, ttl_activation_function, param in TTL_ACTIVATION_FUNCTIONS_WITH_FLOAT_PARAM:
    register_ttl_activation_function_with_float(activation_function_name, ttl_activation_function, param)


def register_ttl_activation_function_with_two_float_params(name, ttl_activation_function, param1_name, param2_name):
    def _golden_function(input_tensor: ttnn.Tensor, parameter1, parameter2, **_):
        import torch

        name_to_torch_function = {
            "clip": torch.clamp,
            "threshold": torch.nn.functional.threshold,
            "softplus": torch.nn.functional.softplus,
        }
        torch_function = name_to_torch_function[name]
        return torch_function(input_tensor, parameter1, parameter2)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        golden_function=_golden_function,
    )
    def activation_function(
        input_tensor: ttnn.Tensor,
        parameter1: float,
        parameter2: float,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter1) or not _is_scalar(parameter2):
            raise TypeError("Expected parameters to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, parameter1, parameter2, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(activation_function, ttnn.decorators.Operation):
        activation_function.__name__ = f"ttnn.{(name)}"
        activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Applies the {name} function to the elements of the input tensor :attr:`input_tensor` with :attr:`{param1_name}` and :attr:`{param2_name}`  parameters.

            .. math::
                {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param1_name} \\; , \\; {param2_name})

            Args:
                * :attr:`input_tensor`
                * :attr:`{param1_name}`
                * :attr:`{param2_name}`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{(name)}(tensor, {param1_name}, {param2_name})

            """
    setattr(THIS_MODULE, name, activation_function)


TTL_ACTIVATION_FUNCTIONS_WITH_TWO_FLOAT_PARAMS = [
    ("clip", ttl.tensor.clip, "min", "max"),  # composite
    ("threshold", ttl.tensor.threshold, "value", "threshold"),  # composite
]

for (
    activation_function_name,
    ttl_activation_function,
    param1,
    param2,
) in TTL_ACTIVATION_FUNCTIONS_WITH_TWO_FLOAT_PARAMS:
    register_ttl_activation_function_with_two_float_params(
        activation_function_name, ttl_activation_function, param1, param2
    )


def torch_reglu(input_tensor, *args, **kwargs):
    import torch

    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.relu(tensB)


def torch_swiglu(input_tensor, *args, **kwargs):
    import torch

    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.silu(tensB)


def torch_geglu(input_tensor, *args, **kwargs):
    import torch

    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.gelu(tensB)


def register_ttl_activation_function_glu(name, ttl_activation_function, param):
    def _golden_function(input_tensor: ttnn.Tensor, dim: int = -1, **_):
        import torch

        name_to_torch_function = {
            "glu": torch.nn.functional.glu,
            "reglu": torch_reglu,
            "swiglu": torch_swiglu,
            "geglu": torch_geglu,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor, dim=dim)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        golden_function=_golden_function,
    )
    def activation_function(
        input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        input_shape = tuple(input_tensor.shape)
        last_dim = input_shape[-1]
        glu_shape = input_shape[:-1] + (int(last_dim / 2),)

        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(dim):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, dim, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(glu_shape))
        return output_tensor

    if isinstance(activation_function, ttnn.decorators.Operation):
        activation_function.__name__ = f"ttnn.{(name)}"
        activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Applies the {name} function to the elements of the input tensor :attr:`input_tensor` split along :attr:`{param}`.

            .. math::
                {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

            Args:
                * :attr:`input_tensor`
                * :attr:`{param}`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((32, 64), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{(name)}(tensor, {param})

            """
    setattr(THIS_MODULE, name, activation_function)


TTL_ACTIVATION_FUNCTIONS_GLU = [
    ("glu", ttl.tensor.glu, "dim"),  # composite
    ("reglu", ttl.tensor.reglu, "dim"),  # composite
    ("swiglu", ttl.tensor.swiglu, "dim"),  # composite
    ("geglu", ttl.tensor.geglu, "dim"),  # composite
]


for activation_function_name, ttl_activation_function, param in TTL_ACTIVATION_FUNCTIONS_GLU:
    register_ttl_activation_function_glu(activation_function_name, ttl_activation_function, param)


__all__ = []

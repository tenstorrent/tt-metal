# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def register_ttnn_cpp_unary_function(unary_function):
    import torch

    def torch_cbrt(x, *args, **kwargs):
        return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3)

    def torch_multigammaln(x, *args, **kwargs):
        result = torch.lgamma(x)
        result += torch.lgamma(x - 0.5)
        result += torch.lgamma(x - 1.0)
        result += torch.lgamma(x - 1.5)
        result += 3.434189657547
        return result

    def _golden_function(input_tensor: ttnn.Tensor, **_):
        name_to_golden_function = {
            "abs": torch.abs,
            "atan": torch.atan,
            "cos": torch.cos,
            "erfinv": torch.erfinv,
            "exp2": torch.exp2,
            "expm1": torch.expm1,
            "eqz": lambda x: torch.eq(x, 0),
            "floor": torch.floor,
            "ceil": torch.ceil,
            "gez": lambda x: torch.ge(x, 0),
            "gtz": lambda x: torch.gt(x, 0),
            "i0": torch.i0,
            "identity": torch.clone,
            "isfinite": torch.isfinite,
            "isinf": torch.isinf,
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
            "leaky_relu": torch.nn.functional.leaky_relu,
            # "prelu": torch_prelu, # Alias for leaky_relu. TODO(#8544): implement PReLU properly
            # Other unaries (composite operations)
            "softplus": torch.nn.functional.softplus,
            "sigmoid_accurate": torch.sigmoid,
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
            "sinh": torch.sinh,
            "softsign": torch.nn.functional.softsign,
            "swish": torch.nn.functional.hardswish,
            "tril": torch.tril,
            "triu": torch.triu,
        }

        golden_keys = set(name_to_golden_function.keys())
        function_names = {function.__name__.split(".")[-1] for function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS}
        if golden_keys != function_names:
            raise ImportError(
                f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}"
            )

        torch_function = name_to_golden_function[unary_function.__name__.split(".")[-1]]
        return torch_function(input_tensor)

    ttnn.attach_golden_function(unary_function, golden_function=_golden_function)


TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.abs,
    ttnn.atan,
    ttnn.cos,
    ttnn.erfinv,
    ttnn.exp2,
    ttnn.expm1,
    ttnn.eqz,
    ttnn.floor,
    ttnn.ceil,
    ttnn.gez,
    ttnn.gtz,
    ttnn.i0,
    ttnn.identity,
    ttnn.isfinite,
    ttnn.isinf,
    ttnn.isnan,
    ttnn.isneginf,
    ttnn.isposinf,
    ttnn.lez,
    ttnn.log,
    ttnn.log10,
    ttnn.log2,
    ttnn.logical_not,
    ttnn.ltz,
    ttnn.neg,
    ttnn.nez,
    ttnn.relu,
    ttnn.relu6,
    ttnn.sigmoid,
    ttnn.sign,
    ttnn.signbit,
    ttnn.silu,
    ttnn.sin,
    ttnn.sqrt,
    ttnn.square,
    ttnn.tan,
    ttnn.tanh,
    # Unaries with fast_and_approximate_mode
    ttnn.exp,
    ttnn.erf,
    ttnn.erfc,
    ttnn.gelu,
    ttnn.rsqrt,
    # Unaries with float parameter
    ttnn.leaky_relu,
    # ttnn.prelu,  # Alias for leaky_relu. TODO(#8544): implement PReLU properly
    # Unaries using op_chain
    ttnn.log_sigmoid,
    ttnn.softplus,
    ttnn.sigmoid_accurate,
    # Other unaries (composite operations - tt_eager dependency)
    ttnn.asinh,
    ttnn.atanh,
    ttnn.cbrt,
    ttnn.cosh,
    ttnn.deg2rad,
    ttnn.digamma,
    ttnn.hardswish,
    ttnn.hardsigmoid,
    ttnn.hardtanh,
    ttnn.lgamma,
    ttnn.log1p,
    ttnn.mish,
    ttnn.multigammaln,
    ttnn.rad2deg,
    ttnn.sinh,
    ttnn.softsign,
    ttnn.swish,
    ttnn.tril,
    ttnn.triu,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_unary_function(unary_function)


def _golden_function_asin(input_tensor_a, *args, device, **kwargs):
    import torch

    return torch.nan_to_num(
        torch.asin(input_tensor_a), nan=device.sfpu_nan(), posinf=device.sfpu_inf(), neginf=-device.sfpu_inf()
    )


ttnn.attach_golden_function(ttnn.asin, golden_function=_golden_function_asin)


def _golden_function_acos(input_tensor_a, *args, device, **kwargs):
    import torch

    return torch.nan_to_num(
        torch.acos(input_tensor_a), nan=device.sfpu_nan(), posinf=device.sfpu_inf(), neginf=-device.sfpu_inf()
    )


ttnn.attach_golden_function(ttnn.acos, golden_function=_golden_function_acos)


def _golden_function_acosh(input_tensor_a, *args, device, **kwargs):
    import torch

    return torch.nan_to_num(
        torch.acosh(input_tensor_a), nan=device.sfpu_nan(), posinf=device.sfpu_inf(), neginf=-device.sfpu_inf()
    )


ttnn.attach_golden_function(ttnn.acosh, golden_function=_golden_function_acosh)


def _golden_function_reciprocal(input_tensor_a, *args, device, **kwargs):
    import torch

    return torch.nan_to_num(
        torch.reciprocal(input_tensor_a), nan=device.sfpu_nan(), posinf=device.sfpu_inf(), neginf=-device.sfpu_inf()
    )


ttnn.attach_golden_function(ttnn.reciprocal, golden_function=_golden_function_reciprocal)


def _golden_function_pow(input_tensor_a, exponent, *args, **kwargs):
    import torch

    return torch.pow(input_tensor_a, exponent)


ttnn.attach_golden_function(ttnn.pow, golden_function=_golden_function_pow)


def _golden_function_elu(input_tensor_a, *args, alpha=1.0, **kwargs):
    import torch

    return torch.nn.functional.elu(input_tensor_a, alpha=alpha)


ttnn.attach_golden_function(ttnn.elu, golden_function=_golden_function_elu)


def _golden_function_relu_min(input_tensor_a, *args, lower_limit, **kwargs):
    import torch

    return torch.max(input_tensor_a, torch.tensor(lower_limit))


ttnn.attach_golden_function(ttnn.relu_min, golden_function=_golden_function_relu_min)


def _golden_function_relu_max(input_tensor_a, *args, upper_limit, **kwargs):
    import torch

    return torch.relu(torch.min(input_tensor_a, torch.tensor(upper_limit)))


ttnn.attach_golden_function(ttnn.relu_max, golden_function=_golden_function_relu_max)


def _golden_function_heaviside(input_tensor_a, *args, value, **kwargs):
    import torch

    return torch.heaviside(input_tensor_a, torch.tensor(value, dtype=input_tensor_a.dtype))


ttnn.attach_golden_function(ttnn.heaviside, golden_function=_golden_function_heaviside)


def _golden_function_polygamma(input_tensor_a, k, *args, **kwargs):
    import torch

    return torch.special.polygamma(n=k, input=input_tensor_a)


ttnn.attach_golden_function(ttnn.polygamma, golden_function=_golden_function_polygamma)


def _golden_function_clamp(input_tensor_a, min, max, *args, **kwargs):
    import torch

    return torch.clamp(input=input_tensor_a, min=min, max=max)


ttnn.attach_golden_function(ttnn.clamp, golden_function=_golden_function_clamp)


def _golden_function_clip(input_tensor_a, min, max, *args, **kwargs):
    import torch

    return torch.clip(input=input_tensor_a, min=min, max=max)


ttnn.attach_golden_function(ttnn.clip, golden_function=_golden_function_clip)


def _golden_function_round(input_tensor_a, decimal, *args, **kwargs):
    import torch

    return torch.round(input=input_tensor_a, decimals=decimal)


ttnn.attach_golden_function(ttnn.round, golden_function=_golden_function_round)


def _golden_function_selu(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.selu(input_tensor_a)


ttnn.attach_golden_function(ttnn.selu, golden_function=_golden_function_selu)


def _golden_function_tanhshrink(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.tanhshrink(input=input_tensor_a)


ttnn.attach_golden_function(ttnn.tanhshrink, golden_function=_golden_function_tanhshrink)


def _golden_function_threshold(input_tensor_a, threshold, value, *args, **kwargs):
    import torch

    return torch.threshold(input_tensor_a, threshold, value)


ttnn.attach_golden_function(ttnn.threshold, golden_function=_golden_function_threshold)


def _golden_function_trunc(input_tensor_a, *args, **kwargs):
    import torch

    return torch.trunc(input=input_tensor_a)


ttnn.attach_golden_function(ttnn.trunc, golden_function=_golden_function_trunc)


def _golden_function_rsub(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.sub(value, input_tensor_a)


ttnn.attach_golden_function(ttnn.rsub, golden_function=_golden_function_rsub)


def _golden_function_rdiv(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.div(torch.tensor(value, dtype=input_tensor_a.dtype), input_tensor_a)


ttnn.attach_golden_function(ttnn.rdiv, golden_function=_golden_function_rdiv)


def _golden_function_bitwise_left_shift(input_tensor_a, shift_amt, *args, **kwargs):
    import torch

    return torch.bitwise_left_shift(input_tensor_a, shift_amt)


ttnn.attach_golden_function(ttnn.bitwise_left_shift, golden_function=_golden_function_bitwise_left_shift)


def _golden_function_bitwise_right_shift(input_tensor_a, shift_amt, *args, **kwargs):
    import torch

    return torch.bitwise_right_shift(input_tensor_a, shift_amt)


ttnn.attach_golden_function(ttnn.bitwise_right_shift, golden_function=_golden_function_bitwise_right_shift)


def _golden_function_bitwise_and(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_and(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_and, golden_function=_golden_function_bitwise_and)


def _golden_function_bitwise_or(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_or(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_or, golden_function=_golden_function_bitwise_or)


def _golden_function_bitwise_xor(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_xor(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_xor, golden_function=_golden_function_bitwise_xor)


def _golden_function_bitwise_not(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_not(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_not, golden_function=_golden_function_bitwise_not)


def _golden_function_glu(input_tensor_a, dim, *args, **kwargs):
    import torch

    return torch.nn.functional.glu(input_tensor_a, dim)


ttnn.attach_golden_function(ttnn.glu, golden_function=_golden_function_glu)


def _golden_function_reglu(input_tensor_a, dim, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"
    split_size = input_tensor_a.size(-1) // 2
    split_tensors = torch.split(input_tensor_a, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.relu(tensB)


ttnn.attach_golden_function(ttnn.reglu, golden_function=_golden_function_reglu)


def _golden_function_geglu(input_tensor_a, dim, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"
    split_size = input_tensor_a.size(-1) // 2
    split_tensors = torch.split(input_tensor_a, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.gelu(tensB)


ttnn.attach_golden_function(ttnn.geglu, golden_function=_golden_function_geglu)


def _golden_function_swiglu(input_tensor_a, dim, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"
    split_size = input_tensor_a.size(-1) // 2
    split_tensors = torch.split(input_tensor_a, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.silu(tensB)


ttnn.attach_golden_function(ttnn.swiglu, golden_function=_golden_function_swiglu)


def _golden_function_logical_not_(input_tensor_a, *args, **kwargs):
    import torch

    return input_tensor_a.logical_not_()


ttnn.attach_golden_function(ttnn.logical_not_, golden_function=_golden_function_logical_not_)


def _golden_function_hardshrink(input_tensor_a, *args, lambd=0.5, **kwargs):
    import torch

    return torch.nn.functional.hardshrink(input_tensor_a, lambd=lambd)


ttnn.attach_golden_function(ttnn.hardshrink, golden_function=_golden_function_hardshrink)


def _golden_function_softshrink(input_tensor_a, *args, lambd=0.5, **kwargs):
    import torch

    return torch.nn.functional.softshrink(input_tensor_a, lambd=lambd)


ttnn.attach_golden_function(ttnn.softshrink, golden_function=_golden_function_softshrink)


def _golden_function_logit(input_tensor_a, *args, eps=None, device, **kwargs):
    import torch

    return torch.nan_to_num(
        torch.special.logit(input_tensor_a, eps=eps),
        nan=device.sfpu_nan(),
        posinf=device.sfpu_inf(),
        neginf=-device.sfpu_inf(),
    )


ttnn.attach_golden_function(ttnn.logit, golden_function=_golden_function_logit)


def _golden_function_celu(input_tensor_a, *args, alpha=1.0, **kwargs):
    import torch

    return torch.celu(input_tensor_a, alpha=alpha)


ttnn.attach_golden_function(ttnn.celu, golden_function=_golden_function_celu)


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

    doc = f"""{(name)}(input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

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

    @ttnn.register_python_operation(name=f"ttnn.{name}", golden_function=_golden_function, doc=doc)
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


def _golden_function_glu(input_tensor_a, dim, *args, **kwargs):
    import torch

    return torch.nn.functional.glu(input_tensor_a, dim)


ttnn.attach_golden_function(ttnn.glu, golden_function=_golden_function_glu)


def _golden_function_reglu(input_tensor_a, dim, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"

    split_size = input_tensor_a.size(-1) // 2
    split_tensors = torch.split(input_tensor_a, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.relu(tensB)


ttnn.attach_golden_function(ttnn.reglu, golden_function=_golden_function_reglu)


def _golden_function_geglu(input_tensor_a, dim, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"

    split_size = input_tensor_a.size(-1) // 2
    split_tensors = torch.split(input_tensor_a, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]

    return tensA * torch.nn.functional.gelu(tensB)


ttnn.attach_golden_function(ttnn.geglu, golden_function=_golden_function_geglu)


def _golden_function_swiglu(input_tensor_a, dim, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"

    split_size = input_tensor_a.size(-1) // 2
    split_tensors = torch.split(input_tensor_a, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]

    return tensA * torch.nn.functional.silu(tensB)


ttnn.attach_golden_function(ttnn.swiglu, golden_function=_golden_function_swiglu)


def _golden_function_normalize_global(input_tensor_a, *args, **kwargs):
    import torch

    mx = torch.mean(input_tensor_a, [0, 1, 2, 3], keepdim=True)
    sx = torch.std(input_tensor_a, [0, 1, 2, 3], keepdim=True)
    input_tensor_a = (input_tensor_a - mx) / sx

    return input_tensor_a


ttnn.attach_golden_function(ttnn.normalize_global, golden_function=_golden_function_normalize_global)


def _golden_function_rpow(input_tensor_a, dim, *args, **kwargs):
    import torch

    return torch.pow(dim, input_tensor_a)


ttnn.attach_golden_function(ttnn.rpow, golden_function=_golden_function_rpow)


def _golden_function_frac(input_tensor_a, *args, **kwargs):
    import torch

    return torch.frac(input_tensor_a)


ttnn.attach_golden_function(ttnn.frac, golden_function=_golden_function_frac)


def _golden_function_rdiv(input_tensor_a, value, *args, round_mode=None, **kwargs):
    import torch

    if round_mode == "None":
        round_mode = None

    return torch.div(torch.full_like(input_tensor_a, value), input_tensor_a, rounding_mode=round_mode)


ttnn.attach_golden_function(ttnn.rdiv, golden_function=_golden_function_rdiv)
__all__ = []

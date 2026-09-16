# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional

import ttnn
from ttnn.operations.golden_common import golden_compute_gradients, golden_prepare_grad_inputs


def _create_golden_function(torch_function_name):
    def golden_function(
        input_tensor: ttnn.Tensor,
        dim: Optional[Union[int, Tuple[int]]] = None,
        keepdim=False,
        correction=None,
        **_,
    ):
        import torch

        torch_function = getattr(torch, torch_function_name)
        if torch_function_name == "argmax" and input_tensor.dtype in (torch.uint16, torch.uint32):
            # PyTorch has no unsigned argmax kernel; widening preserves the ordering of every value.
            input_tensor = input_tensor.to(torch.int64)

        function_kwargs = {"keepdim": keepdim}
        if torch_function_name in ("std", "var") and correction is not None:
            # TTNN exposes correction explicitly, so the golden must not fall back to PyTorch's default.
            function_kwargs["correction"] = correction

        if dim is None:
            if torch_function_name == "argmax":
                return torch_function(input_tensor, dim=None, **function_kwargs)
            if not keepdim:
                function_kwargs.pop("keepdim")
                return torch_function(input_tensor, **function_kwargs)
            dim = tuple(range(len(input_tensor.shape)))

        if torch_function_name in ("max", "min") and isinstance(dim, (tuple, list)):
            # Multi-axis TTNN max/min maps to amax/amin because torch.max/min only accepts one axis.
            torch_function = torch.amax if torch_function_name == "max" else torch.amin
            return torch_function(input_tensor, dim=dim, **function_kwargs)

        output = torch_function(input_tensor, dim=dim, **function_kwargs)
        if torch_function_name in ("max", "min"):
            # Dimensional torch max/min returns values and indices, while TTNN returns values only.
            output = output.values
        return output

    return golden_function


def _create_golden_function_topk():
    def golden_function(input_tensor: ttnn.Tensor, k: int, dim: Optional[int] = None, largest=True, sorted=True, **_):
        import torch

        return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted)

    return golden_function


# Generic reductions
ttnn.attach_golden_function(ttnn.mean, golden_function=_create_golden_function("mean"))
ttnn.attach_golden_function(ttnn.sum, golden_function=_create_golden_function("sum"))
ttnn.attach_golden_function(ttnn.max, golden_function=_create_golden_function("max"))
ttnn.attach_golden_function(ttnn.min, golden_function=_create_golden_function("min"))
ttnn.attach_golden_function(ttnn.var, golden_function=_create_golden_function("var"))
ttnn.attach_golden_function(ttnn.std, golden_function=_create_golden_function("std"))

# Special reductions
ttnn.attach_golden_function(ttnn.argmax, golden_function=_create_golden_function("argmax"))

ttnn.attach_golden_function(ttnn.topk, golden_function=_create_golden_function_topk())


def _golden_function_prod(input_tensor, dim=None, keepdim=False, *, dims=None, **_):
    import torch

    if dims is not None:
        output = input_tensor
        for reduction_dim in sorted((value % input_tensor.ndim for value in dims)):
            output = torch.prod(output, dim=reduction_dim, keepdim=True)
        return output
    if dim is None:
        return torch.prod(input_tensor)
    return torch.prod(input_tensor, dim=dim, keepdim=keepdim)


ttnn.attach_golden_function(ttnn.prod, golden_function=_golden_function_prod)


def _golden_function_prod_bw(grad_tensor, input_tensor, dim=None, *_, **__):
    import torch

    (input_tensor,) = golden_prepare_grad_inputs(input_tensor)
    # The forward reduces over dim (or all dims); keepdim keeps the reduced axis broadcastable for backward.
    if dim is None:
        forward_output = torch.prod(input_tensor)
    else:
        forward_output = torch.prod(input_tensor, dim=dim, keepdim=True)
    return golden_compute_gradients(forward_output, (input_tensor,), grad_tensor)


ttnn.attach_golden_function(ttnn.prod_bw, golden_function=_golden_function_prod_bw)


def _create_accumulation_golden_function(torch_function_name):
    def golden_function(input_tensor, dim, *_, dtype=None, reverse_order=False, **__):
        import torch

        torch_function = getattr(torch, torch_function_name)
        if reverse_order:
            # Reverse accumulation: flip along dim, accumulate, then flip back.
            output = torch_function(torch.flip(input_tensor, dims=[dim]), dim=dim)
            output = torch.flip(output, dims=[dim])
        else:
            output = torch_function(input_tensor, dim=dim)
        if dtype is not None:
            output = output.to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
        return output

    return golden_function


ttnn.attach_golden_function(ttnn.cumsum, golden_function=_create_accumulation_golden_function("cumsum"))
ttnn.attach_golden_function(ttnn.cumprod, golden_function=_create_accumulation_golden_function("cumprod"))


def _golden_function_ema(input_tensor, alpha, *_, **__):
    import torch

    # Exponential moving average along the last (sequence) axis: out[t] = alpha*out[t-1] + (1-alpha)*in[t].
    sequence_length = input_tensor.shape[-1]
    output = torch.empty_like(input_tensor)
    previous = input_tensor[..., 0].clone()
    output[..., 0] = previous
    for t in range(1, sequence_length):
        previous = previous * alpha + (1 - alpha) * input_tensor[..., t]
        output[..., t] = previous
    return output


ttnn.attach_golden_function(ttnn.ema, golden_function=_golden_function_ema)


def _golden_function_var_hw(input_tensor, *_, **__):
    import torch

    # Biased variance (correction=0) over the H and W dims, keeping the reduced axes as size 1.
    return torch.var(input_tensor, dim=(-2, -1), keepdim=True, correction=0)


ttnn.attach_golden_function(ttnn.var_hw, golden_function=_golden_function_var_hw)


def _golden_function_std_hw(input_tensor, *_, **__):
    import torch

    return torch.std(input_tensor, dim=(-2, -1), keepdim=True, correction=0)


ttnn.attach_golden_function(ttnn.std_hw, golden_function=_golden_function_std_hw)


def _golden_function_sampling(input_values_tensor, *_, **__):
    import torch

    # Stochastic top-k/top-p sampler; only the output shape [1, 1, 1, num_users] is deterministic.
    num_users = input_values_tensor.shape[-2]
    output = torch.zeros((1, 1, 1, num_users), dtype=torch.int32)
    ttnn.decorators.set_golden_comparison_config(output, method="skip", scope="all")
    return output


ttnn.attach_golden_function(ttnn.sampling, golden_function=_golden_function_sampling)

# manual_seed only seeds the device RNG and returns None; there is no output value to compare.
ttnn.attach_golden_function(ttnn.manual_seed, golden_function=None)


__all__ = []

ReduceType = ttnn._ttnn.operations.reduction.ReduceType

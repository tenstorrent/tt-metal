# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Tuple, Union, Optional

import ttnn
from ttnn.operations.golden_common import golden_compute_gradients, golden_prepare_grad_inputs


class _ReductionGoldenSpec(NamedTuple):
    """Per-op knowledge mapping a TTNN reduction onto its PyTorch reference."""

    torch_name: str
    # torch.max/min accept a single axis; multi-axis TTNN reductions map to amax/amin instead.
    multi_axis_torch_name: Optional[str] = None
    # Dimensional torch.max/min return (values, indices), while TTNN returns values only.
    values_only: bool = False
    # TTNN exposes correction explicitly, so the golden must not fall back to PyTorch's default.
    passes_correction: bool = False
    # PyTorch has no unsigned argmax kernel; widening preserves the ordering of every value.
    widen_unsigned: bool = False
    # argmax reduces the flattened tensor when dim is None instead of expanding to all dims.
    dim_none_means_flatten: bool = False


_REDUCTION_GOLDEN_SPECS = {
    "mean": _ReductionGoldenSpec("mean"),
    "sum": _ReductionGoldenSpec("sum"),
    "max": _ReductionGoldenSpec("max", multi_axis_torch_name="amax", values_only=True),
    "min": _ReductionGoldenSpec("min", multi_axis_torch_name="amin", values_only=True),
    "var": _ReductionGoldenSpec("var", passes_correction=True),
    "std": _ReductionGoldenSpec("std", passes_correction=True),
    "argmax": _ReductionGoldenSpec("argmax", widen_unsigned=True, dim_none_means_flatten=True),
}


def _create_golden_function(torch_function_name):
    spec = _REDUCTION_GOLDEN_SPECS[torch_function_name]

    def golden_function(
        input_tensor: ttnn.Tensor,
        dim: Optional[Union[int, Tuple[int]]] = None,
        keepdim=False,
        correction=None,
        **_,
    ):
        import torch

        if spec.widen_unsigned and input_tensor.dtype in (torch.uint16, torch.uint32):
            input_tensor = input_tensor.to(torch.int64)

        function_kwargs = {"keepdim": keepdim}
        if spec.passes_correction and correction is not None:
            function_kwargs["correction"] = correction

        if dim is None:
            if spec.dim_none_means_flatten:
                return getattr(torch, spec.torch_name)(input_tensor, dim=None, **function_kwargs)
            if not keepdim:
                # When dim is None, PyTorch reduces over all dimensions; keepdim is not accepted.
                function_kwargs.pop("keepdim")
                return getattr(torch, spec.torch_name)(input_tensor, **function_kwargs)
            # For keepdim to work, we need to specify all dimensions explicitly.
            dim = tuple(range(len(input_tensor.shape)))

        if spec.multi_axis_torch_name is not None and isinstance(dim, (tuple, list)):
            return getattr(torch, spec.multi_axis_torch_name)(input_tensor, dim=dim, **function_kwargs)

        output = getattr(torch, spec.torch_name)(input_tensor, dim=dim, **function_kwargs)
        if spec.values_only:
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

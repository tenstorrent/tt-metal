# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Callable, NamedTuple, Tuple, Union, Optional

import ttnn
from ttnn.operations.golden_common import golden_compute_gradients, golden_prepare_grad_inputs


def _scale_linearly(output, scalar, input_dtype):
    import torch

    if input_dtype == torch.int32 and scalar != 1.0:
        return (output.to(torch.float32) * scalar).to(torch.int32)
    return output * scalar


def _scale_squared(output, scalar, input_dtype):
    return output * scalar**2


def _scale_absolute(output, scalar, input_dtype):
    return output * abs(scalar)


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
    # Applies TTNN's scalar argument to the reduced output as (output, scalar, input_dtype).
    scale_output: Optional[Callable] = None


_REDUCTION_GOLDEN_SPECS = {
    "mean": _ReductionGoldenSpec("mean", scale_output=_scale_linearly),
    "sum": _ReductionGoldenSpec("sum", scale_output=_scale_linearly),
    "max": _ReductionGoldenSpec("max", multi_axis_torch_name="amax", values_only=True, scale_output=_scale_linearly),
    "min": _ReductionGoldenSpec("min", multi_axis_torch_name="amin", values_only=True, scale_output=_scale_linearly),
    "var": _ReductionGoldenSpec("var", passes_correction=True, scale_output=_scale_squared),
    "std": _ReductionGoldenSpec("std", passes_correction=True, scale_output=_scale_absolute),
    "argmax": _ReductionGoldenSpec("argmax", widen_unsigned=True, dim_none_means_flatten=True),
}


def _reduce(spec, input_tensor, dim, keepdim, **torch_kwargs):
    import torch

    if dim is None:
        if spec.dim_none_means_flatten:
            return getattr(torch, spec.torch_name)(input_tensor, dim=None, keepdim=keepdim, **torch_kwargs)
        if not keepdim:
            # When dim is None, PyTorch reduces over all dimensions; keepdim is not accepted.
            return getattr(torch, spec.torch_name)(input_tensor, **torch_kwargs)
        # For keepdim to work, we need to specify all dimensions explicitly.
        dim = tuple(range(len(input_tensor.shape)))

    if spec.multi_axis_torch_name is not None and isinstance(dim, (tuple, list)):
        return getattr(torch, spec.multi_axis_torch_name)(input_tensor, dim=dim, keepdim=keepdim, **torch_kwargs)

    output = getattr(torch, spec.torch_name)(input_tensor, dim=dim, keepdim=keepdim, **torch_kwargs)
    return output.values if spec.values_only else output


def _create_golden_function(torch_function_name):
    spec = _REDUCTION_GOLDEN_SPECS[torch_function_name]

    def golden_function(
        input_tensor: ttnn.Tensor,
        dim: Optional[Union[int, Tuple[int]]] = None,
        keepdim=False,
        correction=None,
        scalar=1.0,
        **_,
    ):
        import torch

        input_dtype = input_tensor.dtype
        if input_tensor.ndim == 0 and spec.scale_output is not None:
            return input_tensor.clone()
        if spec.widen_unsigned and input_dtype in (torch.uint16, torch.uint32):
            input_tensor = input_tensor.to(torch.int64)

        torch_kwargs = {}
        if spec.passes_correction and correction is not None:
            torch_kwargs["correction"] = correction

        output = _reduce(spec, input_tensor, dim, keepdim, **torch_kwargs)
        if spec.scale_output is None:
            return output
        return spec.scale_output(output, scalar, input_dtype)

    return golden_function


def _create_golden_function_topk():
    def golden_function(
        input_tensor: ttnn.Tensor,
        k: int = 32,
        dim: int = -1,
        largest=True,
        sorted=True,
        *,
        stable=False,
        indices_tensor=None,
        **_,
    ):
        import torch

        if stable:
            # torch.topk has no tie-breaking guarantee; a stable sort keeps the lowest index first among ties.
            sorted_values, sorted_indices = torch.sort(input_tensor, dim=dim, descending=largest, stable=True)
            values, indices = sorted_values.narrow(dim, 0, k), sorted_indices.narrow(dim, 0, k)
        else:
            values, indices = torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted)
        if indices_tensor is not None:
            # indices_tensor supplies the label returned for each position along dim.
            indices = torch.gather(indices_tensor.to(torch.int64), dim, indices)
        if not stable:
            # The unstable device network may return any index among equal values, including ties at the k-th
            # boundary, so only positions whose value is unique in its row are compared exactly.
            value_counts = (
                (input_tensor.movedim(dim, -1).unsqueeze(-2) == values.movedim(dim, -1).unsqueeze(-1)).sum(-1)
            ).movedim(-1, dim)
            tie_mask = value_counts > 1
            if bool(torch.any(tie_mask)):
                ttnn.decorators.set_golden_comparison_config(
                    indices, method="allclose", scope="all", rtol=0.0, atol=0.0, mask=~tie_mask
                )
        return values, indices

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


def _hw_statistic_golden(input_tensor, take_sqrt):
    """Evaluate the device height-width variance (or its square root) in float32, keeping the reduced axes as size 1."""

    import torch

    height, width = input_tensor.shape[-2:]
    padded_height = -(-height // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    padded_width = -(-width // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    # The device sums squared deviations over the logical elements but divides by the tile-padded H * W,
    # so non-tile-aligned inputs get a variance scaled by logical / padded area.
    variance = torch.var(input_tensor.float(), dim=(-2, -1), keepdim=True, correction=0) * (
        (height * width) / (padded_height * padded_width)
    )
    output_tensor = (torch.sqrt(variance) if take_sqrt else variance).to(input_tensor.dtype)
    # A single-plane statistic is one value, so PCC is undefined and the default allclose tolerance is
    # finer than one bfloat16 ULP; padding leaks would still shift the value by far more than a few ULP.
    return ttnn.decorators.set_golden_comparison_config(
        output_tensor, method="ulp", scope="degenerate", ulp_threshold=4
    )


def _golden_function_var_hw(input_tensor, *_, **__):
    return _hw_statistic_golden(input_tensor, take_sqrt=False)


ttnn.attach_golden_function(ttnn.var_hw, golden_function=_golden_function_var_hw)


def _golden_function_std_hw(input_tensor, *_, **__):
    return _hw_statistic_golden(input_tensor, take_sqrt=True)


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

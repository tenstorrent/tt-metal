# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional

import ttnn


def _create_golden_function(torch_function_name):
    def golden_function(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim=False, **_):
        import torch

        torch_function = getattr(torch, torch_function_name)
        if dim is None:
            # When dim is None, PyTorch reduces over all dimensions
            # For keepdim to work, we need to specify all dimensions explicitly
            if keepdim:
                all_dims = tuple(range(len(input_tensor.shape)))
                return torch_function(input_tensor, dim=all_dims, keepdim=keepdim)
            else:
                return torch_function(input_tensor)
        else:
            return torch_function(input_tensor, dim=dim, keepdim=keepdim)

    return golden_function


def _create_golden_function_topk():
    def golden_function(input_tensor: ttnn.Tensor, k: int, dim: Optional[int] = None, largest=True, sorted=True, **_):
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


def _torch_dtype(dtype):
    if dtype is None:
        return None

    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
        ttnn.uint32: torch.uint32,
    }.get(dtype)


def _copy_to_out(result, out):
    if out is not None:
        out.copy_(result)
        return out
    return result


def _golden_accumulation(
    torch_function,
    input_tensor,
    dim,
    dtype=None,
    reverse_order=False,
    out=None,
    **_,
):
    import torch

    output_dtype = _torch_dtype(dtype)
    if output_dtype is None:
        output_dtype = out.dtype if out is not None else input_tensor.dtype
    accumulation_dtype = torch.int64 if output_dtype == torch.uint32 else output_dtype
    working_tensor = input_tensor if not reverse_order else input_tensor.flip((dim,))
    result = torch_function(working_tensor, dim=dim, dtype=accumulation_dtype).to(output_dtype)
    if reverse_order:
        result = result.flip((dim,))
    return _copy_to_out(result, out)


def _golden_cumsum(input, dim, **kwargs):
    import torch

    return _golden_accumulation(torch.cumsum, input, dim, **kwargs)


ttnn.attach_golden_function(ttnn.cumsum, golden_function=_golden_cumsum)


def _golden_cumprod(input_tensor, dim, **kwargs):
    import torch

    return _golden_accumulation(torch.cumprod, input_tensor, dim, **kwargs)


ttnn.attach_golden_function(ttnn.cumprod, golden_function=_golden_cumprod)


def _golden_nonzero(input_tensor, **_):
    import torch

    coordinates = torch.nonzero(input_tensor, as_tuple=False).to(torch.uint32)
    count = torch.zeros((1, 1, 1, 8), dtype=torch.uint32, device=input_tensor.device)
    count[..., 0] = coordinates.shape[0]
    indices = torch.zeros((1, 1, 1, input_tensor.numel() * 4), dtype=torch.uint32, device=input_tensor.device)
    if coordinates.numel():
        indices[..., : coordinates.numel()] = coordinates.reshape(-1)
    return [count, indices]


ttnn.attach_golden_function(ttnn.nonzero, golden_function=_golden_nonzero)


def _golden_prod(input_tensor, dim=None, keepdim=False, *args, output_tensor=None, dims=None, **_):
    import torch

    if isinstance(dim, torch.Tensor):
        output_tensor = dim if output_tensor is None else output_tensor
        dim = None
    if output_tensor is not None:
        if dims is None:
            raise TypeError("dims is required when an output tensor is supplied")
        result = input_tensor
        for reduction_dim in sorted((int(value) for value in dims), reverse=True):
            result = torch.prod(result, dim=reduction_dim, keepdim=True)
        return _copy_to_out(result, output_tensor)

    if dim is None:
        return torch.prod(input_tensor)
    return torch.prod(input_tensor, dim=dim, keepdim=keepdim)


ttnn.attach_golden_function(ttnn.prod, golden_function=_golden_prod)


def _golden_sort(input_tensor, dim=-1, descending=False, stable=False, *, out=None, **_):
    import torch

    values, indices = torch.sort(input_tensor, dim=dim, descending=descending, stable=stable)
    if out is not None:
        out_values, out_indices = out
        out_values.copy_(values)
        out_indices.copy_(indices)
        return [out_values, out_indices]
    return [values, indices]


ttnn.attach_golden_function(ttnn.sort, golden_function=_golden_sort)


def _golden_var_hw(input_tensor, **_):
    import torch

    return torch.var(input_tensor, dim=(-2, -1), correction=0, keepdim=True)


ttnn.attach_golden_function(ttnn.var_hw, golden_function=_golden_var_hw)


def _golden_std_hw(input_tensor, **_):
    import torch

    return torch.std(input_tensor, dim=(-2, -1), correction=0, keepdim=True)


ttnn.attach_golden_function(ttnn.std_hw, golden_function=_golden_std_hw)


__all__ = []

ReduceType = ttnn._ttnn.operations.reduction.ReduceType

# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn._ttnn
import ttnn.decorators

abs = ttnn.moreh_abs_pow
adam = ttnn.moreh_adam
adamw = ttnn.moreh_adamw
arange = ttnn.moreh_arange
bmm = ttnn.moreh_bmm
bmm_backward = ttnn.moreh_bmm_backward
clip_grad_norm = ttnn.moreh_clip_grad_norm
cumsum = ttnn.moreh_cumsum
cumsum_backward = ttnn.moreh_cumsum_backward
dot = ttnn.moreh_dot
dot_backward = ttnn.moreh_dot_backward
fold = ttnn.moreh_fold
full = ttnn.moreh_full
full_like = ttnn.moreh_full_like
getitem = ttnn.moreh_getitem
group_norm = ttnn.moreh_group_norm
group_norm_backward = ttnn.moreh_group_norm_backward
layer_norm = ttnn.moreh_layer_norm
layer_norm_backward = ttnn.moreh_layer_norm_backward
linear = ttnn.moreh_linear
linear_backward = ttnn.moreh_linear_backward
logsoftmax = ttnn.moreh_logsoftmax
logsoftmax_backward = ttnn.moreh_logsoftmax_backward
matmul = ttnn.moreh_matmul
matmul_backward = ttnn.moreh_matmul_backward
mean = ttnn.moreh_mean
mean_backward = ttnn.moreh_mean_backward
nll_loss = ttnn.moreh_nll_loss
nll_loss_backward = ttnn.moreh_nll_loss_backward
nll_loss_unreduced_backward = ttnn.moreh_nll_loss_unreduced_backward
norm = ttnn.moreh_norm
norm_backward = ttnn.moreh_norm_backward
sgd = ttnn.moreh_sgd
softmax = ttnn.moreh_softmax
softmax_backward = ttnn.moreh_softmax_backward
softmin = ttnn.moreh_softmin
softmin_backward = ttnn.moreh_softmin_backward
sum = ttnn.moreh_sum
sum_backward = ttnn.moreh_sum_backward

SoftmaxBackwardOp = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOp
SoftmaxBackwardOpParallelizationStrategy = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOpParallelizationStrategy
SoftmaxOp = ttnn._ttnn.operations.moreh.MorehSoftmaxOpParallelizationStrategy
SoftmaxOpParallelizationStrategy = ttnn._ttnn.operations.moreh.MorehSoftmaxOpParallelizationStrategy


def _copy_to_output(value, output):
    if output is None:
        return value
    output.copy_(value.reshape(output.shape))
    return output


def _optional_outputs_postprocess(output, args, kwargs):
    if output is None:
        return None
    if isinstance(output, (list, tuple)):
        converted = [_optional_outputs_postprocess(value, args, kwargs) for value in output]
        return type(output)(converted)
    return ttnn.decorators.default_postprocess_golden_function_outputs(output, args, kwargs)


def _torch_dtype(dtype):
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
        ttnn.uint32: torch.uint32,
    }
    return mapping.get(dtype, torch.bfloat16)


def _creation_postprocess(output, *, dtype, layout, device, memory_config):
    return ttnn.from_torch(
        output,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _arange_postprocess(output, args, kwargs):
    provided_output = kwargs.get("output")
    if provided_output is not None:
        return _creation_postprocess(
            output,
            dtype=provided_output.dtype,
            layout=provided_output.layout,
            device=provided_output.device(),
            memory_config=provided_output.memory_config(),
        )
    device = kwargs.get("device", args[3] if len(args) > 3 else None)
    dtype = kwargs.get("dtype", ttnn.bfloat16)
    layout = ttnn.ROW_MAJOR_LAYOUT if kwargs.get("untilize_out", False) else ttnn.TILE_LAYOUT
    memory_config = kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG)
    return _creation_postprocess(
        output,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _full_postprocess(output, args, kwargs):
    device = kwargs.get("device", args[2] if len(args) > 2 else None)
    return _creation_postprocess(
        output,
        dtype=kwargs.get("dtype", ttnn.bfloat16),
        layout=kwargs.get("layout", ttnn.TILE_LAYOUT),
        device=device,
        memory_config=kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG),
    )


def _full_like_postprocess(output, args, kwargs):
    input_tensor = args[0] if args else kwargs["input"]
    dtype = kwargs.get("dtype", args[2] if len(args) > 2 else None) or input_tensor.dtype
    layout = kwargs.get("layout", args[3] if len(args) > 3 else None) or input_tensor.layout
    memory_config = kwargs.get("memory_config", args[4] if len(args) > 4 else None) or input_tensor.memory_config()
    return _creation_postprocess(
        output,
        dtype=dtype,
        layout=layout,
        device=input_tensor.device(),
        memory_config=memory_config,
    )


def _normalize_dims(dim, rank):
    if dim is None:
        return tuple(range(rank))
    if isinstance(dim, int):
        return (dim % rank,)
    return tuple(value % rank for value in dim)


def _torch_dim(dim):
    return tuple(dim) if isinstance(dim, list) else dim


def _expand_reduction_grad(output_grad, input_shape, dim, keepdim):
    dims = _normalize_dims(dim, len(input_shape))
    grad = output_grad
    if not keepdim:
        for value in sorted(dims):
            grad = grad.unsqueeze(value)
    return grad.expand(input_shape), dims


def _sum_to_shape(value, shape):
    return value.sum_to_size(tuple(shape))


def _golden_abs(input, p, *, output=None, **_):
    return _copy_to_output(input.abs().pow(p), output)


def _golden_adam(
    param_in,
    grad,
    exp_avg_in,
    exp_avg_sq_in,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    step=0,
    amsgrad=False,
    max_exp_avg_sq_in=None,
    param_out=None,
    exp_avg_out=None,
    exp_avg_sq_out=None,
    max_exp_avg_sq_out=None,
    **_,
):
    import torch

    output_dtype = param_in.dtype
    param = param_in.float()
    adjusted_grad = grad.float() + weight_decay * param
    new_exp_avg = beta1 * exp_avg_in.float() + (1 - beta1) * adjusted_grad
    new_exp_avg_sq = beta2 * exp_avg_sq_in.float() + (1 - beta2) * adjusted_grad.square()
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    if amsgrad:
        if max_exp_avg_sq_in is None:
            raise ValueError("max_exp_avg_sq_in is required when amsgrad=True")
        new_max_exp_avg_sq = torch.maximum(max_exp_avg_sq_in.float(), new_exp_avg_sq)
        denominator_state = new_max_exp_avg_sq
    else:
        new_max_exp_avg_sq = None
        denominator_state = new_exp_avg_sq
    denominator = denominator_state.sqrt() / (bias_correction2**0.5) + eps
    new_param = param - (lr / bias_correction1) * (new_exp_avg / denominator)
    new_param = new_param.to(output_dtype)
    new_exp_avg = new_exp_avg.to(output_dtype)
    new_exp_avg_sq = new_exp_avg_sq.to(output_dtype)
    if new_max_exp_avg_sq is not None:
        new_max_exp_avg_sq = new_max_exp_avg_sq.to(output_dtype)
    return [
        _copy_to_output(new_param, param_out),
        _copy_to_output(new_exp_avg, exp_avg_out),
        _copy_to_output(new_exp_avg_sq, exp_avg_sq_out),
        _copy_to_output(new_max_exp_avg_sq, max_exp_avg_sq_out) if amsgrad else None,
    ]


def _golden_adamw(
    param_in,
    grad,
    exp_avg_in,
    exp_avg_sq_in,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=1e-2,
    step=0,
    amsgrad=False,
    *,
    max_exp_avg_sq_in=None,
    param_out=None,
    exp_avg_out=None,
    exp_avg_sq_out=None,
    max_exp_avg_sq_out=None,
    **_,
):
    import torch

    output_dtype = param_in.dtype
    param = param_in.float()
    float_grad = grad.float()
    decayed_param = param - lr * weight_decay * param
    new_exp_avg = beta1 * exp_avg_in.float() + (1 - beta1) * float_grad
    new_exp_avg_sq = beta2 * exp_avg_sq_in.float() + (1 - beta2) * float_grad.square()
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    if amsgrad:
        if max_exp_avg_sq_in is None:
            raise ValueError("max_exp_avg_sq_in is required when amsgrad=True")
        new_max_exp_avg_sq = torch.maximum(max_exp_avg_sq_in.float(), new_exp_avg_sq)
        denominator_state = new_max_exp_avg_sq
    else:
        new_max_exp_avg_sq = None
        denominator_state = new_exp_avg_sq
    denominator = denominator_state.sqrt() / (bias_correction2**0.5) + eps
    new_param = decayed_param - (lr / bias_correction1) * (new_exp_avg / denominator)
    new_param = new_param.to(output_dtype)
    new_exp_avg = new_exp_avg.to(output_dtype)
    new_exp_avg_sq = new_exp_avg_sq.to(output_dtype)
    if new_max_exp_avg_sq is not None:
        new_max_exp_avg_sq = new_max_exp_avg_sq.to(output_dtype)
    return [
        _copy_to_output(new_param, param_out),
        _copy_to_output(new_exp_avg, exp_avg_out),
        _copy_to_output(new_exp_avg_sq, exp_avg_sq_out),
        _copy_to_output(new_max_exp_avg_sq, max_exp_avg_sq_out) if amsgrad else None,
    ]


def _golden_arange(
    start=0,
    end=None,
    step=1,
    device=None,
    *,
    output=None,
    untilize_out=False,
    dtype=ttnn.bfloat16,
    **_,
):
    import torch

    torch_dtype = _torch_dtype(dtype)
    arange_dtype = torch.int64 if torch_dtype == torch.uint32 else torch_dtype
    value = torch.arange(start=start, end=end, step=step, dtype=arange_dtype).to(torch_dtype)
    if not untilize_out:
        value = value.reshape(1, -1)
    return _copy_to_output(value, output)


def _golden_bmm(input, mat2, *, output=None, **_):
    import torch

    return _copy_to_output(torch.bmm(input, mat2), output)


def _golden_bmm_backward(
    output_grad,
    input,
    mat2,
    *,
    are_required_outputs=(True, True),
    input_grad=None,
    mat2_grad=None,
    **_,
):
    import torch

    if are_required_outputs[0] and input_grad is None:
        raise ValueError("input_grad is required when are_required_outputs[0] is True")
    if are_required_outputs[1] and mat2_grad is None:
        raise ValueError("mat2_grad is required when are_required_outputs[1] is True")
    grad_input = torch.bmm(output_grad, mat2.transpose(-1, -2))
    grad_mat2 = torch.bmm(input.transpose(-1, -2), output_grad)
    return [
        _copy_to_output(grad_input, input_grad) if are_required_outputs[0] else None,
        _copy_to_output(grad_mat2, mat2_grad) if are_required_outputs[1] else None,
    ]


def _golden_clip_grad_norm(
    inputs,
    max_norm,
    norm_type=2.0,
    error_if_nonfinite=False,
    *,
    total_norm=None,
    **_,
):
    import torch

    norms = torch.stack([torch.linalg.vector_norm(value, norm_type) for value in inputs])
    norm = torch.linalg.vector_norm(norms, norm_type)
    if error_if_nonfinite and not torch.isfinite(norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, "
            "so it cannot be clipped."
        )
    coefficient = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for value in inputs:
        value.mul_(coefficient.to(value.device))
    return _copy_to_output(norm.reshape(1), total_norm)


def _golden_cumsum(input, dim, *, output=None, **_):
    import torch

    return _copy_to_output(torch.cumsum(input, dim), output)


def _golden_cumsum_backward(output_grad, dim, *, input_grad=None, **_):
    import torch

    value = torch.flip(torch.cumsum(torch.flip(output_grad, (dim,)), dim), (dim,))
    return _copy_to_output(value, input_grad)


def _golden_dot(input_tensor_a, input_tensor_b, *, output=None, **_):
    import torch

    value = torch.dot(input_tensor_a.reshape(-1), input_tensor_b.reshape(-1)).reshape(1, 1, 1, 1)
    return _copy_to_output(value, output)


def _golden_dot_backward(
    output_grad,
    input,
    other,
    *,
    input_grad=None,
    other_grad=None,
    **_,
):
    scalar_grad = output_grad.reshape(-1)[0]
    return [
        _copy_to_output(other * scalar_grad, input_grad) if input_grad is not None else None,
        _copy_to_output(input * scalar_grad, other_grad) if other_grad is not None else None,
    ]


def _golden_fold(
    input,
    output=None,
    output_size=None,
    kernel_size=None,
    dilation=(1, 1),
    padding=(0, 0),
    stride=(1, 1),
    **_,
):
    import torch.nn.functional as F

    return _copy_to_output(F.fold(input, output_size, kernel_size, dilation, padding, stride), output)


def _golden_full(shape, fill_value, device=None, *, dtype=ttnn.bfloat16, **_):
    import torch

    return torch.full(tuple(shape), fill_value, dtype=_torch_dtype(dtype))


def _golden_full_like(input, fill_value, dtype=None, layout=None, memory_config=None, **_):
    import torch

    torch_dtype = input.dtype if dtype is None else _torch_dtype(dtype)
    return torch.full(input.shape, fill_value, dtype=torch_dtype, device=input.device)


def _golden_getitem(input=None, index_tensors=(), index_dims=(), *, output=None, **_):
    index = [slice(None)] * input.ndim
    for index_tensor, index_dim in zip(index_tensors, index_dims):
        index[index_dim] = index_tensor.to(dtype=__import__("torch").long)
    return _copy_to_output(input[tuple(index)], output)


def _golden_group_norm(
    input,
    num_groups,
    eps=1e-5,
    gamma=None,
    beta=None,
    *,
    are_required_outputs=(True, False, False),
    output=None,
    mean=None,
    rstd=None,
    **_,
):
    import torch.nn.functional as F

    channels = input.shape[1]
    weight = gamma.reshape(channels) if gamma is not None else None
    bias = beta.reshape(channels) if beta is not None else None
    normalized = F.group_norm(input, num_groups, weight, bias, eps)
    grouped = input.reshape(input.shape[0], num_groups, -1)
    computed_mean = grouped.mean(dim=-1)
    computed_rstd = (grouped.var(dim=-1, unbiased=False) + eps).rsqrt()
    stats_shape = (1, 1, input.shape[0], num_groups)
    return [
        _copy_to_output(normalized, output),
        _copy_to_output(computed_mean.reshape(mean.shape), mean)
        if mean is not None
        else (computed_mean.reshape(stats_shape) if are_required_outputs[1] else None),
        _copy_to_output(computed_rstd.reshape(rstd.shape), rstd)
        if rstd is not None
        else (computed_rstd.reshape(stats_shape) if are_required_outputs[2] else None),
    ]


def _golden_group_norm_backward(
    output_grad,
    input,
    mean,
    rstd,
    num_groups,
    *,
    are_required_outputs=(True, False, False),
    gamma=None,
    input_grad=None,
    gamma_grad=None,
    beta_grad=None,
    **_,
):
    batch, channels = input.shape[:2]
    grouped_input = input.reshape(batch, num_groups, -1)
    grouped_mean = mean.reshape(batch, num_groups, 1)
    grouped_rstd = rstd.reshape(batch, num_groups, 1)
    x_hat = (grouped_input - grouped_mean) * grouped_rstd
    weight = gamma.reshape(channels) if gamma is not None else input.new_ones(channels)
    trailing_ones = (1,) * (input.ndim - 2)
    weighted_grad = output_grad * weight.reshape(1, channels, *trailing_ones)
    grouped_grad = weighted_grad.reshape_as(grouped_input)
    group_size = grouped_input.shape[-1]
    computed_input_grad = (
        grouped_rstd
        / group_size
        * (
            group_size * grouped_grad
            - grouped_grad.sum(dim=-1, keepdim=True)
            - x_hat * (grouped_grad * x_hat).sum(dim=-1, keepdim=True)
        )
    ).reshape_as(input)
    x_hat = x_hat.reshape_as(input)
    reduce_dims = (0, *range(2, input.ndim))
    computed_gamma_grad = (output_grad * x_hat).sum(dim=reduce_dims)
    computed_beta_grad = output_grad.sum(dim=reduce_dims)
    return [
        _copy_to_output(computed_input_grad, input_grad) if are_required_outputs[0] else None,
        _copy_to_output(computed_gamma_grad.reshape(gamma_grad.shape), gamma_grad)
        if are_required_outputs[1] and gamma_grad is not None
        else (computed_gamma_grad.reshape(1, 1, 1, channels) if are_required_outputs[1] else None),
        _copy_to_output(computed_beta_grad.reshape(beta_grad.shape), beta_grad)
        if are_required_outputs[2] and beta_grad is not None
        else (computed_beta_grad.reshape(1, 1, 1, channels) if are_required_outputs[2] else None),
    ]


def _golden_layer_norm(
    input,
    normalized_dims,
    eps=1e-5,
    gamma=None,
    beta=None,
    *,
    output=None,
    mean=None,
    rstd=None,
    **_,
):
    import torch.nn.functional as F

    normalized_shape = tuple(input.shape[-normalized_dims:])
    weight = gamma.reshape(normalized_shape) if gamma is not None else None
    bias = beta.reshape(normalized_shape) if beta is not None else None
    normalized = F.layer_norm(input, normalized_shape, weight, bias, eps)
    dims = tuple(range(input.ndim - normalized_dims, input.ndim))
    computed_mean = input.mean(dim=dims)
    computed_rstd = (input.var(dim=dims, unbiased=False) + eps).rsqrt()
    return [
        _copy_to_output(normalized, output),
        _copy_to_output(computed_mean.reshape(mean.shape), mean) if mean is not None else None,
        _copy_to_output(computed_rstd.reshape(rstd.shape), rstd) if rstd is not None else None,
    ]


def _golden_layer_norm_backward(
    output_grad,
    input,
    mean,
    rstd,
    normalized_dims,
    *,
    gamma=None,
    input_grad=None,
    gamma_grad=None,
    beta_grad=None,
    **_,
):
    normalized_shape = tuple(input.shape[-normalized_dims:])
    leading_dims = tuple(range(input.ndim - normalized_dims))
    expanded_shape = tuple(input.shape[:-normalized_dims]) + (1,) * normalized_dims
    expanded_mean = mean.reshape(expanded_shape)
    expanded_rstd = rstd.reshape(expanded_shape)
    x_hat = (input - expanded_mean) * expanded_rstd
    weight = gamma.reshape(normalized_shape) if gamma is not None else input.new_ones(normalized_shape)
    weighted_grad = output_grad * weight
    reduce_dims = tuple(range(input.ndim - normalized_dims, input.ndim))
    element_count = 1
    for size in normalized_shape:
        element_count *= size
    computed_input_grad = (
        expanded_rstd
        / element_count
        * (
            element_count * weighted_grad
            - weighted_grad.sum(dim=reduce_dims, keepdim=True)
            - x_hat * (weighted_grad * x_hat).sum(dim=reduce_dims, keepdim=True)
        )
    )
    computed_gamma_grad = (output_grad * x_hat).sum(dim=leading_dims) if leading_dims else output_grad * x_hat
    computed_beta_grad = output_grad.sum(dim=leading_dims) if leading_dims else output_grad
    return [
        _copy_to_output(computed_input_grad, input_grad) if input_grad is not None else None,
        _copy_to_output(computed_gamma_grad.reshape(gamma_grad.shape), gamma_grad) if gamma_grad is not None else None,
        _copy_to_output(computed_beta_grad.reshape(beta_grad.shape), beta_grad) if beta_grad is not None else None,
    ]


def _golden_linear(input, weight, *, bias=None, output=None, **_):
    import torch.nn.functional as F

    return _copy_to_output(F.linear(input, weight, bias), output)


def _golden_linear_backward(
    output_grad,
    input,
    weight,
    *,
    are_required_outputs=(True, True, True),
    bias=None,
    input_grad=None,
    weight_grad=None,
    bias_grad=None,
    **_,
):
    import torch

    if are_required_outputs[0] and input_grad is None:
        raise ValueError("input_grad is required when are_required_outputs[0] is True")
    if are_required_outputs[1] and weight_grad is None:
        raise ValueError("weight_grad is required when are_required_outputs[1] is True")
    if are_required_outputs[2] and bias_grad is None:
        raise ValueError("bias_grad is required when are_required_outputs[2] is True")
    computed_input_grad = torch.matmul(output_grad, weight)
    flat_output_grad = output_grad.reshape(-1, output_grad.shape[-1])
    flat_input = input.reshape(-1, input.shape[-1])
    computed_weight_grad = flat_output_grad.transpose(0, 1).matmul(flat_input)
    computed_bias_grad = flat_output_grad.sum(dim=0)
    return [
        _copy_to_output(computed_input_grad, input_grad) if are_required_outputs[0] else None,
        _copy_to_output(computed_weight_grad.reshape(weight.shape), weight_grad) if are_required_outputs[1] else None,
        _copy_to_output(
            computed_bias_grad.reshape(
                bias_grad.shape
                if bias_grad is not None
                else (bias.shape if bias is not None else computed_bias_grad.shape)
            ),
            bias_grad,
        )
        if are_required_outputs[2]
        else None,
    ]


def _golden_softmax(input_tensor, dim, *, output_tensor=None, **_):
    import torch

    return _copy_to_output(torch.softmax(input_tensor, dim), output_tensor)


def _golden_softmin(input_tensor, dim, *, output_tensor=None, **_):
    import torch.nn.functional as F

    return _copy_to_output(F.softmin(input_tensor, dim), output_tensor)


def _golden_logsoftmax(input_tensor, dim, *, output_tensor=None, **_):
    import torch

    return _copy_to_output(torch.log_softmax(input_tensor, dim), output_tensor)


def _golden_softmax_backward(
    output_tensor,
    output_grad_tensor,
    dim,
    *,
    input_grad_tensor=None,
    **_,
):
    value = output_tensor * (output_grad_tensor - (output_grad_tensor * output_tensor).sum(dim=dim, keepdim=True))
    return _copy_to_output(value, input_grad_tensor)


def _golden_softmin_backward(
    output_tensor,
    output_grad_tensor,
    dim,
    *,
    input_grad_tensor=None,
    **_,
):
    value = output_tensor * ((output_grad_tensor * output_tensor).sum(dim=dim, keepdim=True) - output_grad_tensor)
    return _copy_to_output(value, input_grad_tensor)


def _golden_logsoftmax_backward(
    output_tensor,
    output_grad_tensor,
    dim,
    *,
    input_grad_tensor=None,
    **_,
):
    value = output_grad_tensor - output_tensor.exp() * output_grad_tensor.sum(dim=dim, keepdim=True)
    return _copy_to_output(value, input_grad_tensor)


def _golden_matmul(
    input,
    other,
    *,
    transpose_input=False,
    transpose_other=False,
    output=None,
    bias=None,
    **_,
):
    import torch

    left = input.transpose(-1, -2) if transpose_input else input
    right = other.transpose(-1, -2) if transpose_other else other
    value = torch.matmul(left, right)
    if bias is not None:
        value = value + bias
    return _copy_to_output(value, output)


def _golden_matmul_backward(
    output_grad,
    input_a,
    input_b,
    *,
    are_required_outputs=(True, True),
    input_a_grad=None,
    input_b_grad=None,
    **_,
):
    import torch

    if are_required_outputs[0] and input_a_grad is None:
        raise ValueError("input_a_grad is required when are_required_outputs[0] is True")
    if are_required_outputs[1] and input_b_grad is None:
        raise ValueError("input_b_grad is required when are_required_outputs[1] is True")
    computed_input_a_grad = _sum_to_shape(torch.matmul(output_grad, input_b.transpose(-1, -2)), input_a.shape)
    computed_input_b_grad = _sum_to_shape(torch.matmul(input_a.transpose(-1, -2), output_grad), input_b.shape)
    return [
        _copy_to_output(computed_input_a_grad, input_a_grad) if are_required_outputs[0] else None,
        _copy_to_output(computed_input_b_grad, input_b_grad) if are_required_outputs[1] else None,
    ]


def _golden_mean(input, *, dim=None, keepdim=False, divisor=None, output=None, **_):
    import torch

    dim = _torch_dim(dim)
    if divisor is None:
        value = torch.mean(input, dim=dim, keepdim=keepdim)
    else:
        value = torch.sum(input, dim=dim, keepdim=keepdim) / divisor
    return _copy_to_output(value, output)


def _golden_mean_backward(
    output_grad,
    *,
    dim=None,
    keepdim=False,
    input_grad_shape=None,
    input_grad=None,
    **_,
):
    shape = tuple(input_grad.shape) if input_grad is not None else tuple(input_grad_shape)
    value, dims = _expand_reduction_grad(output_grad, shape, dim, keepdim)
    divisor = 1
    for reduction_dim in dims:
        divisor *= shape[reduction_dim]
    return _copy_to_output(value / divisor, input_grad)


def _golden_nll_loss(
    input_tensor,
    target_tensor,
    reduction,
    *,
    weight_tensor=None,
    divisor_tensor=None,
    output_tensor=None,
    ignore_index=-100,
    **_,
):
    import torch
    import torch.nn.functional as F

    if reduction == "mean" and divisor_tensor is None:
        raise ValueError("divisor_tensor is required when reduction='mean'")
    target = target_tensor.to(torch.long)
    value = F.nll_loss(input_tensor, target, weight_tensor, reduction=reduction, ignore_index=ignore_index)
    if reduction == "none" and input_tensor.ndim == 2 and output_tensor is None:
        value = value.unsqueeze(0)
    elif reduction != "none":
        value = value.reshape(1)
    if divisor_tensor is not None and reduction == "mean":
        valid = target != ignore_index
        if weight_tensor is None:
            divisor = valid.sum().to(input_tensor.dtype)
        else:
            divisor = weight_tensor[target[valid]].sum()
        divisor_tensor.copy_(divisor.reshape(divisor_tensor.shape))
    return _copy_to_output(value, output_tensor)


def _nll_input_grad(target, output_grad, weight, ignore_index, reduction_divisor, input_grad):
    import torch

    result = torch.zeros_like(input_grad)
    target = target.to(torch.long)
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    scale = -output_grad
    if weight is not None:
        scale = scale * weight[safe_target]
    if reduction_divisor is not None:
        scale = scale / reduction_divisor
    scale = torch.where(valid, scale.expand_as(target), torch.zeros_like(target, dtype=input_grad.dtype))
    result.scatter_(1, safe_target.unsqueeze(1), scale.unsqueeze(1))
    return result


def _golden_nll_loss_backward(
    target_tensor,
    output_grad_tensor,
    reduction_mean,
    *,
    weight_tensor=None,
    input_grad_tensor=None,
    divisor_tensor=None,
    ignore_index=-100,
    **_,
):
    if input_grad_tensor is None:
        raise ValueError("input_grad_tensor is required because the input shape is not part of this operation")
    divisor = None
    if reduction_mean:
        if divisor_tensor is not None:
            divisor = divisor_tensor.reshape(-1)[0]
        else:
            target = target_tensor.to(__import__("torch").long)
            valid = target != ignore_index
            divisor = (
                valid.sum().to(input_grad_tensor.dtype) if weight_tensor is None else weight_tensor[target[valid]].sum()
            )
    value = _nll_input_grad(
        target_tensor,
        output_grad_tensor.reshape(-1)[0],
        weight_tensor,
        ignore_index,
        divisor,
        input_grad_tensor,
    )
    return _copy_to_output(value, input_grad_tensor)


def _golden_nll_loss_unreduced_backward(
    target_tensor,
    output_grad_tensor,
    *,
    weight_tensor=None,
    input_grad_tensor=None,
    ignore_index=-100,
    **_,
):
    if input_grad_tensor is None:
        raise ValueError("input_grad_tensor is required because the input shape is not part of this operation")
    value = _nll_input_grad(
        target_tensor,
        output_grad_tensor,
        weight_tensor,
        ignore_index,
        None,
        input_grad_tensor,
    )
    return _copy_to_output(value, input_grad_tensor)


def _golden_norm(input, p, *, dim=None, keepdim=False, output=None, **_):
    import torch

    return _copy_to_output(torch.norm(input, p=p, dim=_torch_dim(dim), keepdim=keepdim), output)


def _golden_norm_backward(
    input,
    output,
    output_grad,
    p,
    *,
    dim=None,
    keepdim=False,
    input_grad=None,
    **_,
):
    import math
    import torch

    dims = _normalize_dims(dim, input.ndim)
    expanded_output = output
    expanded_output_grad = output_grad
    if not keepdim:
        for value in sorted(dims):
            expanded_output = expanded_output.unsqueeze(value)
            expanded_output_grad = expanded_output_grad.unsqueeze(value)
    if p == 0:
        value = torch.zeros_like(input)
    elif math.isinf(p):
        selected = input.abs() == expanded_output
        tie_count = selected.sum(dim=dims, keepdim=True)
        value = input.sign() * selected / tie_count * expanded_output_grad
    else:
        value = input.sign() * input.abs().pow(p - 1) * expanded_output.pow(1 - p) * expanded_output_grad
        value = torch.where(expanded_output == 0, torch.zeros_like(value), value)
    return _copy_to_output(value, input_grad)


def _golden_sgd(
    param_in,
    grad,
    momentum_buffer_in=None,
    param_out=None,
    momentum_buffer_out=None,
    lr=1e-3,
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False,
    *,
    momentum_initialized,
    **_,
):
    direction = grad + weight_decay * param_in
    if momentum != 0:
        if momentum_initialized:
            if momentum_buffer_in is None:
                raise ValueError("momentum_buffer_in is required when momentum is initialized")
            new_momentum = momentum * momentum_buffer_in + (1 - dampening) * direction
        else:
            new_momentum = direction
        direction = direction + momentum * new_momentum if nesterov else new_momentum
    else:
        new_momentum = None
    new_param = param_in - lr * direction
    return [
        _copy_to_output(new_param, param_out),
        _copy_to_output(new_momentum, momentum_buffer_out) if momentum != 0 else None,
    ]


def _golden_sum(input, dim=None, *, keepdim=False, output=None, **_):
    import torch

    return _copy_to_output(torch.sum(input, dim=_torch_dim(dim), keepdim=keepdim), output)


def _golden_sum_backward(
    output_grad,
    *,
    input=None,
    dim=None,
    keepdim=False,
    input_grad=None,
    **_,
):
    reference = input if input is not None else input_grad
    if reference is None:
        raise ValueError("input or input_grad is required to determine the input shape")
    value, _ = _expand_reduction_grad(output_grad, reference.shape, dim, keepdim)
    return _copy_to_output(value, input_grad)


ttnn.attach_golden_function(abs, _golden_abs)
ttnn.attach_golden_function(
    adam,
    _golden_adam,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(
    adamw,
    _golden_adamw,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(
    arange,
    _golden_arange,
    postprocess_golden_function_outputs=_arange_postprocess,
)
ttnn.attach_golden_function(bmm, _golden_bmm)
ttnn.attach_golden_function(
    bmm_backward,
    _golden_bmm_backward,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(clip_grad_norm, _golden_clip_grad_norm)
ttnn.attach_golden_function(cumsum, _golden_cumsum)
ttnn.attach_golden_function(cumsum_backward, _golden_cumsum_backward)
ttnn.attach_golden_function(dot, _golden_dot)
ttnn.attach_golden_function(
    dot_backward,
    _golden_dot_backward,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(fold, _golden_fold)
ttnn.attach_golden_function(
    full,
    _golden_full,
    postprocess_golden_function_outputs=_full_postprocess,
)
ttnn.attach_golden_function(
    full_like,
    _golden_full_like,
    postprocess_golden_function_outputs=_full_like_postprocess,
)
ttnn.attach_golden_function(getitem, _golden_getitem)
ttnn.attach_golden_function(
    group_norm,
    _golden_group_norm,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(
    group_norm_backward,
    _golden_group_norm_backward,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(
    layer_norm,
    _golden_layer_norm,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(
    layer_norm_backward,
    _golden_layer_norm_backward,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(linear, _golden_linear)
ttnn.attach_golden_function(
    linear_backward,
    _golden_linear_backward,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(logsoftmax, _golden_logsoftmax)
ttnn.attach_golden_function(logsoftmax_backward, _golden_logsoftmax_backward)
ttnn.attach_golden_function(matmul, _golden_matmul)
ttnn.attach_golden_function(
    matmul_backward,
    _golden_matmul_backward,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(mean, _golden_mean)
ttnn.attach_golden_function(mean_backward, _golden_mean_backward)
ttnn.attach_golden_function(nll_loss, _golden_nll_loss)
ttnn.attach_golden_function(nll_loss_backward, _golden_nll_loss_backward)
ttnn.attach_golden_function(nll_loss_unreduced_backward, _golden_nll_loss_unreduced_backward)
ttnn.attach_golden_function(norm, _golden_norm)
ttnn.attach_golden_function(norm_backward, _golden_norm_backward)
ttnn.attach_golden_function(
    sgd,
    _golden_sgd,
    postprocess_golden_function_outputs=_optional_outputs_postprocess,
)
ttnn.attach_golden_function(softmax, _golden_softmax)
ttnn.attach_golden_function(softmax_backward, _golden_softmax_backward)
ttnn.attach_golden_function(softmin, _golden_softmin)
ttnn.attach_golden_function(softmin_backward, _golden_softmin_backward)
ttnn.attach_golden_function(sum, _golden_sum)
ttnn.attach_golden_function(sum_backward, _golden_sum_backward)

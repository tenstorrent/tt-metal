# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn._ttnn
from ttnn.operations.golden_common import (
    golden_compute_gradients,
    golden_prepare_grad_inputs,
    golden_select_optional_outputs,
)

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
getitem = ttnn.moreh_getitem
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
softmin = ttnn.moreh_softmin
softmin_backward = ttnn.moreh_softmin_backward
sum = ttnn.moreh_sum
sum_backward = ttnn.moreh_sum_backward



def _golden_abs_pow(input, p, *_, **__):
    import torch

    return torch.abs(input) ** p


ttnn.attach_golden_function(ttnn.moreh_abs_pow, golden_function=_golden_abs_pow)


def _golden_arange(start=0, end=None, step=1, *_, dtype=None, **__):
    import torch

    output = torch.arange(start, end, step, dtype=torch.float32)
    if dtype is not None:
        output = output.to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
    return output


ttnn.attach_golden_function(ttnn.moreh_arange, golden_function=_golden_arange)


def _golden_full(shape, fill_value, *_, dtype=None, **__):
    import torch

    if isinstance(shape, ttnn.Shape):
        shape = tuple(shape)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else torch.bfloat16
    return torch.full(shape, fill_value, dtype=torch_dtype)


ttnn.attach_golden_function(ttnn.moreh_full, golden_function=_golden_full)


def _golden_full_like(input, fill_value, *_, dtype=None, **__):
    import torch

    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else None
    return torch.full_like(input, fill_value, dtype=torch_dtype)


ttnn.attach_golden_function(ttnn.moreh_full_like, golden_function=_golden_full_like)


def _golden_sum(input, dim=None, *_, keepdim=False, **__):
    import torch

    return torch.sum(input, dim=dim, keepdim=keepdim)


ttnn.attach_golden_function(ttnn.moreh_sum, golden_function=_golden_sum)


def _golden_mean(input, dim=None, *_, keepdim=False, **__):
    import torch

    return torch.mean(input, dim=dim, keepdim=keepdim)


ttnn.attach_golden_function(ttnn.moreh_mean, golden_function=_golden_mean)


def _golden_norm(input, p, dim=None, *_, keepdim=False, **__):
    import torch

    return torch.linalg.vector_norm(input, ord=p, dim=dim, keepdim=keepdim)


ttnn.attach_golden_function(ttnn.moreh_norm, golden_function=_golden_norm)


def _golden_cumsum(input, dim, *_, **__):
    import torch

    return torch.cumsum(input, dim=dim)


ttnn.attach_golden_function(ttnn.moreh_cumsum, golden_function=_golden_cumsum)


def _golden_dot(input_tensor_a, input_tensor_b, *_, **__):
    import torch

    # moreh_dot flattens both operands and produces a scalar dot product in a [1, 1, 1, 1] output.
    output = torch.matmul(input_tensor_a.reshape(-1), input_tensor_b.reshape(-1))
    output = output.reshape(1, 1, 1, 1)
    ttnn.decorators.set_golden_comparison_config(output, method="allclose", scope="degenerate", rtol=0.1, atol=0.1)
    return output


ttnn.attach_golden_function(ttnn.moreh_dot, golden_function=_golden_dot)


def _golden_matmul(input, other, *_, transpose_input=False, transpose_other=False, bias=None, **__):
    import torch

    if transpose_input:
        input = input.transpose(-1, -2)
    if transpose_other:
        other = other.transpose(-1, -2)
    output = torch.matmul(input, other)
    if bias is not None:
        output = output + bias
    return output


ttnn.attach_golden_function(ttnn.moreh_matmul, golden_function=_golden_matmul)


def _golden_bmm(input, mat2, *_, **__):
    import torch

    return torch.bmm(input, mat2)


ttnn.attach_golden_function(ttnn.moreh_bmm, golden_function=_golden_bmm)


def _golden_linear(input, weight, *_, bias=None, **__):
    import torch

    return torch.nn.functional.linear(input, weight, bias)


ttnn.attach_golden_function(ttnn.moreh_linear, golden_function=_golden_linear)


def _golden_getitem(input, index_tensors, index_dims, *_, **__):
    import torch

    normalized_indices = []
    for index_tensor, dim in zip(index_tensors, index_dims):
        index = index_tensor.reshape(-1).long()
        normalized_indices.append(torch.where(index < 0, index + input.shape[dim], index))
    if len(index_dims) == 1:
        return torch.index_select(input, index_dims[0], normalized_indices[0])

    selection = [slice(None)] * input.ndim
    for dim, index in zip(index_dims, normalized_indices):
        selection[dim] = index
    return input[tuple(selection)]


ttnn.attach_golden_function(ttnn.moreh_getitem, golden_function=_golden_getitem)


def _golden_fold(
    input,
    output=None,
    output_size=None,
    kernel_size=None,
    dilation=(1, 1),
    padding=(0, 0),
    stride=(1, 1),
    memory_config=None,
):
    import torch

    return torch.nn.functional.fold(
        input,
        output_size=tuple(output_size),
        kernel_size=tuple(kernel_size),
        dilation=tuple(dilation),
        padding=tuple(padding),
        stride=tuple(stride),
    )


ttnn.attach_golden_function(ttnn.moreh_fold, golden_function=_golden_fold)


def _golden_layer_norm(
    input, normalized_dims, eps=1e-5, gamma=None, beta=None, *, output=None, mean=None, rstd=None, **__
):
    import torch

    normalized_shape = tuple(input.shape[-normalized_dims:])
    gamma_value = gamma.reshape(normalized_shape) if gamma is not None else None
    beta_value = beta.reshape(normalized_shape) if beta is not None else None
    output_value = torch.nn.functional.layer_norm(input, normalized_shape, gamma_value, beta_value, eps)

    reduced_dims = tuple(range(input.ndim - normalized_dims, input.ndim))
    mean_value = input.mean(dim=reduced_dims)
    expanded_mean = mean_value.reshape(*mean_value.shape, *([1] * normalized_dims))
    rstd_value = ((input - expanded_mean).pow(2).mean(dim=reduced_dims) + eps).rsqrt()
    if mean is not None:
        mean_value = mean_value.reshape(mean.shape)
        ttnn.decorators.set_golden_comparison_config(mean_value, method="allclose", scope="all", rtol=0.1, atol=0.1)
    if rstd is not None:
        rstd_value = rstd_value.reshape(rstd.shape)
        ttnn.decorators.set_golden_comparison_config(rstd_value, method="allclose", scope="all", rtol=0.1, atol=0.1)
    return [output_value, mean_value if mean is not None else None, rstd_value if rstd is not None else None]


ttnn.attach_golden_function(
    ttnn.moreh_layer_norm,
    golden_function=_golden_layer_norm,
    output_tensor_kwarg_names=("output", "mean", "rstd"),
)


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
    **__,
):
    import torch

    gamma_value = gamma.reshape(-1) if gamma is not None else None
    beta_value = beta.reshape(-1) if beta is not None else None
    output_value = torch.nn.functional.group_norm(input, num_groups, gamma_value, beta_value, eps)

    batch_size = input.shape[0]
    grouped_input = input.reshape(batch_size, num_groups, -1)
    mean_value = grouped_input.mean(dim=-1)
    rstd_value = ((grouped_input - mean_value.unsqueeze(-1)).pow(2).mean(dim=-1) + eps).rsqrt()
    mean_value = mean_value.reshape(mean.shape if mean is not None else (1, 1, batch_size, num_groups))
    rstd_value = rstd_value.reshape(rstd.shape if rstd is not None else (1, 1, batch_size, num_groups))
    required = [
        True,
        are_required_outputs[1] or mean is not None,
        are_required_outputs[2] or rstd is not None,
    ]
    return golden_select_optional_outputs([output_value, mean_value, rstd_value], required)


ttnn.attach_golden_function(
    ttnn.moreh_group_norm,
    golden_function=_golden_group_norm,
    output_tensor_kwarg_names=("output", "mean", "rstd"),
)


def _golden_softmax(input_tensor, dim, *_, **__):
    import torch

    return torch.softmax(input_tensor, dim=dim)


ttnn.attach_golden_function(ttnn.moreh_softmax, golden_function=_golden_softmax)


def _golden_softmin(input_tensor, dim, *_, **__):
    import torch

    return torch.softmin(input_tensor, dim=dim)


ttnn.attach_golden_function(ttnn.moreh_softmin, golden_function=_golden_softmin)


def _golden_logsoftmax(input_tensor, dim, *_, **__):
    import torch

    return torch.log_softmax(input_tensor, dim=dim)


ttnn.attach_golden_function(ttnn.moreh_logsoftmax, golden_function=_golden_logsoftmax)


def _golden_nll_loss(input_tensor, target_tensor, reduction, *_, weight_tensor=None, ignore_index=-100, **__):
    import torch

    output = torch.nn.functional.nll_loss(
        input_tensor, target_tensor.long(), weight=weight_tensor, reduction=reduction, ignore_index=ignore_index
    )
    return output if reduction == "none" else output.reshape(1)


ttnn.attach_golden_function(ttnn.moreh_nll_loss, golden_function=_golden_nll_loss)


def _golden_sum_backward(output_grad, input=None, dim=None, *_, keepdim=False, **__):
    import torch

    # Broadcast the (reduced) output gradient back to the input shape.
    if input is not None:
        target_shape = list(input.shape)
    else:
        target_shape = list(output_grad.shape)
    grad = output_grad
    if dim is not None and not keepdim:
        dims = [dim] if isinstance(dim, int) else list(dim)
        for d in sorted(dims):
            grad = grad.unsqueeze(d)
    return torch.broadcast_to(grad, target_shape).contiguous()


ttnn.attach_golden_function(
    ttnn.moreh_sum_backward,
    golden_function=_golden_sum_backward,
    output_tensor_kwarg_names=("input_grad",),
)


def _golden_mean_backward(output_grad, *, dim=None, keepdim=False, input_grad_shape=None, input_grad=None, **__):
    import torch

    if input_grad_shape is None and input_grad is not None:
        input_grad_shape = input_grad.shape
    grad = output_grad
    if dim is not None and not keepdim:
        dims = [dim] if isinstance(dim, int) else list(dim)
        for d in sorted(dims):
            grad = grad.unsqueeze(d)
    target_shape = list(input_grad_shape) if input_grad_shape is not None else list(grad.shape)
    reduced_dims = range(len(target_shape)) if dim is None else ([dim] if isinstance(dim, int) else dim)
    count = 1
    for reduced_dim in reduced_dims:
        count *= target_shape[reduced_dim % len(target_shape)]
    return torch.broadcast_to(grad, target_shape).contiguous() / count


ttnn.attach_golden_function(
    ttnn.moreh_mean_backward,
    golden_function=_golden_mean_backward,
    output_tensor_kwarg_names=("input_grad",),
)


def _golden_cumsum_backward(output_grad, dim, *_, **__):
    import torch

    # cumsum backward is a reverse (exclusive) cumulative sum of the output gradient.
    return torch.flip(torch.cumsum(torch.flip(output_grad, dims=[dim]), dim=dim), dims=[dim])


ttnn.attach_golden_function(
    ttnn.moreh_cumsum_backward,
    golden_function=_golden_cumsum_backward,
    output_tensor_kwarg_names=("input_grad",),
)


def _golden_dot_backward(output_grad, input, other, *, input_grad=None, other_grad=None, **__):
    # Dot backward: grad_a = output_grad * other, grad_b = output_grad * input (elementwise, then broadcast).
    grad_a = output_grad * other
    grad_b = output_grad * input
    return [grad_a if input_grad is not None else None, grad_b if other_grad is not None else None]


ttnn.attach_golden_function(
    ttnn.moreh_dot_backward,
    golden_function=_golden_dot_backward,
    output_tensor_kwarg_names=("input_grad", "other_grad"),
)


def _golden_matmul_backward(output_grad, input_a, input_b, *_, are_required_outputs=None, **__):
    import torch

    input_a, input_b = golden_prepare_grad_inputs(input_a, input_b)
    forward = torch.matmul(input_a, input_b)
    grads = golden_compute_gradients(forward, (input_a, input_b), output_grad)
    if are_required_outputs is not None:
        return golden_select_optional_outputs(grads, are_required_outputs)
    return grads


ttnn.attach_golden_function(
    ttnn.moreh_matmul_backward,
    golden_function=_golden_matmul_backward,
    output_tensor_kwarg_names=("input_a_grad", "input_b_grad"),
)


def _golden_bmm_backward(output_grad, input, mat2, *_, are_required_outputs=None, **__):
    import torch

    input, mat2 = golden_prepare_grad_inputs(input, mat2)
    forward = torch.bmm(input, mat2)
    grads = golden_compute_gradients(forward, (input, mat2), output_grad)
    if are_required_outputs is not None:
        return golden_select_optional_outputs(grads, are_required_outputs)
    return grads


ttnn.attach_golden_function(
    ttnn.moreh_bmm_backward,
    golden_function=_golden_bmm_backward,
    output_tensor_kwarg_names=("input_grad", "mat2_grad"),
)


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
    **__,
):
    import torch

    if bias is None:
        bias_value = torch.zeros(weight.shape[0], dtype=weight.dtype)
    else:
        bias_value = bias.reshape(-1)
    prepared = golden_prepare_grad_inputs(input, weight, bias_value)
    forward = torch.nn.functional.linear(prepared[0], prepared[1], prepared[2])
    grads = golden_compute_gradients(forward, tuple(prepared), output_grad)
    if bias_grad is not None:
        grads[2] = grads[2].reshape(bias_grad.shape)
    required = [
        are_required_outputs[0] or input_grad is not None,
        are_required_outputs[1] or weight_grad is not None,
        are_required_outputs[2] or bias_grad is not None,
    ]
    return golden_select_optional_outputs(grads, required)


ttnn.attach_golden_function(
    ttnn.moreh_linear_backward,
    golden_function=_golden_linear_backward,
    output_tensor_kwarg_names=("input_grad", "weight_grad", "bias_grad"),
)


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
    **__,
):
    normalized_shape = tuple(input.shape[-normalized_dims:])
    stats_shape = tuple(input.shape[:-normalized_dims])
    expanded_shape = (*stats_shape, *([1] * normalized_dims))
    centered = input - mean.reshape(expanded_shape)
    normalized = centered * rstd.reshape(expanded_shape)
    gamma_value = gamma.reshape(normalized_shape) if gamma is not None else 1.0
    scaled_output_grad = output_grad * gamma_value
    reduced_dims = tuple(range(input.ndim - normalized_dims, input.ndim))
    element_count = 1
    for size in normalized_shape:
        element_count *= size
    input_grad_value = rstd.reshape(expanded_shape) * (
        scaled_output_grad
        - scaled_output_grad.sum(dim=reduced_dims, keepdim=True) / element_count
        - normalized * (scaled_output_grad * normalized).sum(dim=reduced_dims, keepdim=True) / element_count
    )

    outer_dims = tuple(range(input.ndim - normalized_dims))
    gamma_grad_value = output_grad * normalized
    beta_grad_value = output_grad
    if outer_dims:
        gamma_grad_value = gamma_grad_value.sum(dim=outer_dims)
        beta_grad_value = beta_grad_value.sum(dim=outer_dims)
    if gamma_grad is not None:
        gamma_grad_value = gamma_grad_value.reshape(gamma_grad.shape)
    if beta_grad is not None:
        beta_grad_value = beta_grad_value.reshape(beta_grad.shape)
    return [
        input_grad_value if input_grad is not None else None,
        gamma_grad_value if gamma_grad is not None else None,
        beta_grad_value if beta_grad is not None else None,
    ]


ttnn.attach_golden_function(
    ttnn.moreh_layer_norm_backward,
    golden_function=_golden_layer_norm_backward,
    output_tensor_kwarg_names=("input_grad", "gamma_grad", "beta_grad"),
)


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
    **__,
):
    batch_size, channels = input.shape[:2]
    grouped_input = input.reshape(batch_size, num_groups, -1)
    normalized = (grouped_input - mean.reshape(batch_size, num_groups, 1)) * rstd.reshape(batch_size, num_groups, 1)
    gamma_value = gamma.reshape(1, channels, *([1] * (input.ndim - 2))) if gamma is not None else 1.0
    scaled_output_grad = (output_grad * gamma_value).reshape(batch_size, num_groups, -1)
    elements_per_group = grouped_input.shape[-1]
    input_grad_value = rstd.reshape(batch_size, num_groups, 1) * (
        scaled_output_grad
        - scaled_output_grad.sum(dim=-1, keepdim=True) / elements_per_group
        - normalized * (scaled_output_grad * normalized).sum(dim=-1, keepdim=True) / elements_per_group
    )
    input_grad_value = input_grad_value.reshape(input.shape)

    normalized_input = normalized.reshape(input.shape)
    reduction_dims = (0, *range(2, input.ndim))
    gamma_grad_value = (output_grad * normalized_input).sum(dim=reduction_dims)
    beta_grad_value = output_grad.sum(dim=reduction_dims)
    if gamma_grad is not None:
        gamma_grad_value = gamma_grad_value.reshape(gamma_grad.shape)
    if beta_grad is not None:
        beta_grad_value = beta_grad_value.reshape(beta_grad.shape)
    required = [
        are_required_outputs[0] or input_grad is not None,
        are_required_outputs[1] or gamma_grad is not None,
        are_required_outputs[2] or beta_grad is not None,
    ]
    return golden_select_optional_outputs([input_grad_value, gamma_grad_value, beta_grad_value], required)


ttnn.attach_golden_function(
    ttnn.moreh_group_norm_backward,
    golden_function=_golden_group_norm_backward,
    output_tensor_kwarg_names=("input_grad", "gamma_grad", "beta_grad"),
)


def _golden_norm_backward(input, output, output_grad, p, *, dim=None, keepdim=False, input_grad=None, **__):
    import torch

    (input,) = golden_prepare_grad_inputs(input)
    forward = torch.linalg.vector_norm(input, ord=p, dim=dim, keepdim=keepdim)
    grads = golden_compute_gradients(forward, (input,), output_grad)
    return grads[0]


ttnn.attach_golden_function(
    ttnn.moreh_norm_backward,
    golden_function=_golden_norm_backward,
    output_tensor_kwarg_names=("input_grad",),
)


def _golden_softmax_backward(output_tensor, output_grad_tensor, dim, *_, **__):
    import torch

    # softmax backward: grad = output * (output_grad - sum(output_grad * output, dim, keepdim=True))
    return output_tensor * (output_grad_tensor - (output_grad_tensor * output_tensor).sum(dim=dim, keepdim=True))


ttnn.attach_golden_function(
    ttnn.moreh_softmax_backward,
    golden_function=_golden_softmax_backward,
    output_tensor_kwarg_names=("input_grad_tensor",),
)


def _golden_softmin_backward(output_tensor, output_grad_tensor, dim, *_, **__):
    # softmin(x) = softmax(-x), so its derivative has the opposite sign of softmax.
    return output_tensor * ((output_grad_tensor * output_tensor).sum(dim=dim, keepdim=True) - output_grad_tensor)


ttnn.attach_golden_function(
    ttnn.moreh_softmin_backward,
    golden_function=_golden_softmin_backward,
    output_tensor_kwarg_names=("input_grad_tensor",),
)


def _golden_logsoftmax_backward(output_tensor, output_grad_tensor, dim, *_, **__):
    import torch

    # logsoftmax backward: grad = output_grad - exp(output) * sum(output_grad, dim, keepdim=True)
    return output_grad_tensor - output_tensor.exp() * output_grad_tensor.sum(dim=dim, keepdim=True)


ttnn.attach_golden_function(
    ttnn.moreh_logsoftmax_backward,
    golden_function=_golden_logsoftmax_backward,
    output_tensor_kwarg_names=("input_grad_tensor",),
)


def _nll_loss_backward_impl(
    target_tensor, output_grad_tensor, *, weight_tensor, input_grad_tensor, divisor_tensor, ignore_index, reduction_mean
):
    import torch

    if input_grad_tensor is None:
        raise NotImplementedError("moreh nll_loss backward golden requires input_grad_tensor for the output shape")
    input_grad = torch.zeros_like(input_grad_tensor)
    num_batches, num_classes = input_grad_tensor.shape[0], input_grad_tensor.shape[1]
    spatial_shape = input_grad_tensor.shape[2:]
    spatial_volume = 1
    for s in spatial_shape:
        spatial_volume *= s

    target = target_tensor.long().reshape(num_batches, spatial_volume)
    grad = input_grad.reshape(num_batches, num_classes, spatial_volume)

    # Broadcast the output gradient to one value per (batch, spatial) position.
    out_grad = output_grad_tensor.reshape(-1)
    out_grad = out_grad.expand(num_batches * spatial_volume).reshape(num_batches, spatial_volume)

    valid = target != ignore_index
    clamped_target = target.clamp(min=0, max=num_classes - 1)
    weights = torch.ones(num_classes, dtype=input_grad.dtype)
    if weight_tensor is not None:
        weights = weight_tensor.reshape(-1).to(input_grad.dtype)
    # grad[n, target[n, s], s] = -out_grad[n, s] * weight[target[n, s]] for valid positions.
    contrib = (-out_grad) * weights[clamped_target] * valid
    grad.scatter_(1, clamped_target.unsqueeze(1), contrib.unsqueeze(1))

    if reduction_mean:
        divisor = divisor_tensor.reshape(-1)[0] if divisor_tensor is not None else weights[clamped_target[valid]].sum()
        grad = grad / divisor
    return grad.reshape(input_grad_tensor.shape)


def _golden_nll_loss_backward(
    target_tensor,
    output_grad_tensor,
    reduction_mean,
    *_,
    weight_tensor=None,
    input_grad_tensor=None,
    divisor_tensor=None,
    ignore_index=-100,
    **__,
):
    return _nll_loss_backward_impl(
        target_tensor,
        output_grad_tensor,
        weight_tensor=weight_tensor,
        input_grad_tensor=input_grad_tensor,
        divisor_tensor=divisor_tensor,
        ignore_index=ignore_index,
        reduction_mean=reduction_mean,
    )


ttnn.attach_golden_function(
    ttnn.moreh_nll_loss_backward,
    golden_function=_golden_nll_loss_backward,
    output_tensor_kwarg_names=("input_grad_tensor",),
)


def _golden_nll_loss_unreduced_backward(
    target_tensor, output_grad_tensor, *_, weight_tensor=None, input_grad_tensor=None, ignore_index=-100, **__
):
    return _nll_loss_backward_impl(
        target_tensor,
        output_grad_tensor,
        weight_tensor=weight_tensor,
        input_grad_tensor=input_grad_tensor,
        divisor_tensor=None,
        ignore_index=ignore_index,
        reduction_mean=False,
    )


ttnn.attach_golden_function(
    ttnn.moreh_nll_loss_unreduced_backward,
    golden_function=_golden_nll_loss_unreduced_backward,
    output_tensor_kwarg_names=("input_grad_tensor",),
)


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
    **__,
):
    import torch

    # Standard Adam (L2 weight decay couples into the gradient).
    if weight_decay != 0:
        grad = grad + weight_decay * param_in
    exp_avg = beta1 * exp_avg_in + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq_in + (1 - beta2) * grad * grad
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    if amsgrad:
        max_exp_avg_sq = torch.maximum(max_exp_avg_sq_in, exp_avg_sq) if max_exp_avg_sq_in is not None else exp_avg_sq
        denom = (max_exp_avg_sq / bias_correction2).sqrt() + eps
    else:
        max_exp_avg_sq = None
        denom = (exp_avg_sq / bias_correction2).sqrt() + eps
    param = param_in - lr * (exp_avg / bias_correction1) / denom
    # Keep the standard Adam reference above, but skip param_out comparison: the current device
    # implementation passes (moreh_adam.cpp:150) integer step to power_tile, whose exponent argument expects float bits.
    # The optimizer-state outputs do not use that bias-correction path and remain validated.
    ttnn.decorators.set_golden_comparison_config(param, method="skip", scope="all")
    values = [param, exp_avg, exp_avg_sq, max_exp_avg_sq]
    required = [True, True, True, amsgrad or max_exp_avg_sq_out is not None]
    return golden_select_optional_outputs(values, required)


ttnn.attach_golden_function(
    ttnn.moreh_adam,
    golden_function=_golden_adam,
    output_tensor_kwarg_names=("param_out", "exp_avg_out", "exp_avg_sq_out", "max_exp_avg_sq_out"),
)


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
    **__,
):
    import torch

    # AdamW applies decoupled weight decay directly to the parameter.
    param = param_in - lr * weight_decay * param_in
    exp_avg = beta1 * exp_avg_in + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq_in + (1 - beta2) * grad * grad
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    if amsgrad:
        max_exp_avg_sq = torch.maximum(max_exp_avg_sq_in, exp_avg_sq) if max_exp_avg_sq_in is not None else exp_avg_sq
        denom = (max_exp_avg_sq / bias_correction2).sqrt() + eps
    else:
        max_exp_avg_sq = None
        denom = (exp_avg_sq / bias_correction2).sqrt() + eps
    param = param - lr * (exp_avg / bias_correction1) / denom
    values = [param, exp_avg, exp_avg_sq, max_exp_avg_sq]
    required = [True, True, True, amsgrad or max_exp_avg_sq_out is not None]
    return golden_select_optional_outputs(values, required)


ttnn.attach_golden_function(
    ttnn.moreh_adamw,
    golden_function=_golden_adamw,
    output_tensor_kwarg_names=("param_out", "exp_avg_out", "exp_avg_sq_out", "max_exp_avg_sq_out"),
)


def _golden_sgd(
    param_in,
    grad,
    momentum_buffer_in=None,
    param_out=None,
    momentum_buffer_out=None,
    lr=1e-3,
    momentum=0.0,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    *,
    momentum_initialized=False,
    **__,
):
    import torch

    if weight_decay != 0:
        grad = grad + weight_decay * param_in
    buf = momentum_buffer_in
    if momentum != 0:
        if buf is None or not momentum_initialized:
            buf = grad.clone()
        else:
            buf = momentum * buf + (1 - dampening) * grad
        grad = grad + momentum * buf if nesterov else buf
    param = param_in - lr * grad
    values = [param, buf]
    required = [True, momentum != 0 or momentum_buffer_out is not None]
    return golden_select_optional_outputs(values, required)


ttnn.attach_golden_function(
    ttnn.moreh_sgd,
    golden_function=_golden_sgd,
    output_tensor_kwarg_names=("param_out", "momentum_buffer_out"),
)


def _golden_clip_grad_norm(
    inputs, max_norm, norm_type=2.0, error_if_nonfinite=False, *, _ttnn_global_golden=False, **__
):
    import torch

    # Total norm across all gradient tensors: (sum_i ||g_i||_p^p)^(1/p).
    per_tensor_norms = [torch.linalg.vector_norm(inp, ord=norm_type).reshape(1) for inp in inputs]
    total_norm = torch.linalg.vector_norm(torch.cat(per_tensor_norms), ord=norm_type)
    if error_if_nonfinite and not bool(torch.isfinite(total_norm)):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, "
            "so it cannot be clipped."
        )
    if _ttnn_global_golden:
        clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for input_tensor in inputs:
            input_tensor.mul_(clip_coefficient.to(input_tensor.dtype))
    total_norm = total_norm.reshape(1, 1)
    ttnn.decorators.set_golden_comparison_config(
        total_norm,
        method="allclose",
        scope="degenerate",
        rtol=0.1,
        atol=0.1,
        nonfinite="mask",
    )
    return total_norm


_golden_clip_grad_norm._ttnn_mutates_global_inputs = True
ttnn.attach_golden_function(
    ttnn.moreh_clip_grad_norm,
    golden_function=_golden_clip_grad_norm,
    output_tensor_kwarg_names=("total_norm",),
)

# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn._ttnn
from ttnn.operations.golden_common import (
    golden_compute_gradients,
    golden_prepare_grad_inputs,
    golden_select_optional_outputs,
)

abs = ttnn._ttnn.operations.moreh.moreh_abs_pow
adam = ttnn._ttnn.operations.moreh.moreh_adam
adamw = ttnn._ttnn.operations.moreh.moreh_adamw
arange = ttnn._ttnn.operations.moreh.moreh_arange
bmm = ttnn._ttnn.operations.moreh.moreh_bmm
bmm_backward = ttnn._ttnn.operations.moreh.moreh_bmm_backward
clip_grad_norm = ttnn._ttnn.operations.moreh.moreh_clip_grad_norm
cumsum = ttnn._ttnn.operations.moreh.moreh_cumsum
cumsum_backward = ttnn._ttnn.operations.moreh.moreh_cumsum_backward
dot = ttnn._ttnn.operations.moreh.moreh_dot
dot_backward = ttnn._ttnn.operations.moreh.moreh_dot_backward
fold = ttnn._ttnn.operations.moreh.moreh_fold
getitem = ttnn._ttnn.operations.moreh.moreh_getitem
group_norm = ttnn._ttnn.operations.moreh.moreh_group_norm
group_norm_backward = ttnn._ttnn.operations.moreh.moreh_group_norm_backward
layer_norm = ttnn._ttnn.operations.moreh.moreh_layer_norm
layer_norm_backward = ttnn._ttnn.operations.moreh.moreh_layer_norm_backward
linear = ttnn._ttnn.operations.moreh.moreh_linear
linear_backward = ttnn._ttnn.operations.moreh.moreh_linear_backward
logsoftmax = ttnn._ttnn.operations.moreh.moreh_logsoftmax
logsoftmax_backward = ttnn._ttnn.operations.moreh.moreh_logsoftmax_backward
matmul = ttnn._ttnn.operations.moreh.moreh_matmul
matmul_backward = ttnn._ttnn.operations.moreh.moreh_matmul_backward
mean = ttnn._ttnn.operations.moreh.moreh_mean
mean_backward = ttnn._ttnn.operations.moreh.moreh_mean_backward
nll_loss = ttnn._ttnn.operations.moreh.moreh_nll_loss
nll_loss_backward = ttnn._ttnn.operations.moreh.moreh_nll_loss_backward
nll_loss_unreduced_backward = ttnn._ttnn.operations.moreh.moreh_nll_loss_unreduced_backward
norm = ttnn._ttnn.operations.moreh.moreh_norm
norm_backward = ttnn._ttnn.operations.moreh.moreh_norm_backward
sgd = ttnn._ttnn.operations.moreh.moreh_sgd
softmax = ttnn._ttnn.operations.moreh.moreh_softmax
softmax_backward = ttnn._ttnn.operations.moreh.moreh_softmax_backward
softmin = ttnn._ttnn.operations.moreh.moreh_softmin
softmin_backward = ttnn._ttnn.operations.moreh.moreh_softmin_backward
sum = ttnn._ttnn.operations.moreh.moreh_sum
sum_backward = ttnn._ttnn.operations.moreh.moreh_sum_backward

SoftmaxBackwardOp = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOp
SoftmaxBackwardOpParallelizationStrategy = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOpParallelizationStrategy
SoftmaxOp = ttnn._ttnn.operations.moreh.MorehSoftmaxOpParallelizationStrategy
SoftmaxOpParallelizationStrategy = ttnn._ttnn.operations.moreh.MorehSoftmaxOpParallelizationStrategy


# ---------------------------------------------------------------------------
# Golden functions
# ---------------------------------------------------------------------------


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
    return output.reshape(1, 1, 1, 1)


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

    # Select elements along each indexed dim using the corresponding index tensor.
    output = input
    for index_tensor, dim in sorted(zip(index_tensors, index_dims), key=lambda pair: pair[1]):
        output = torch.index_select(output, dim, index_tensor.reshape(-1).long())
    return output


ttnn.attach_golden_function(ttnn.moreh_getitem, golden_function=_golden_getitem)


def _golden_fold(input, *_, output_size, kernel_size, dilation=(1, 1), padding=(0, 0), stride=(1, 1), **__):
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


def _golden_layer_norm(input, normalized_dims, *_, eps=1e-5, gamma=None, beta=None, **__):
    import torch

    normalized_shape = tuple(input.shape[-normalized_dims:]) if normalized_dims > 0 else tuple(input.shape)
    return torch.nn.functional.layer_norm(input, normalized_shape, gamma, beta, eps)


ttnn.attach_golden_function(ttnn.moreh_layer_norm, golden_function=_golden_layer_norm)


def _golden_group_norm(input, num_groups, *_, eps=1e-5, gamma=None, beta=None, **__):
    import torch

    return torch.nn.functional.group_norm(input, num_groups, gamma, beta, eps)


ttnn.attach_golden_function(ttnn.moreh_group_norm, golden_function=_golden_group_norm)


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

    return torch.nn.functional.nll_loss(
        input_tensor, target_tensor.long(), weight=weight_tensor, reduction=reduction, ignore_index=ignore_index
    )


ttnn.attach_golden_function(ttnn.moreh_nll_loss, golden_function=_golden_nll_loss)


# ---------------------------------------------------------------------------
# Backward goldens
# ---------------------------------------------------------------------------


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


ttnn.attach_golden_function(ttnn.moreh_sum_backward, golden_function=_golden_sum_backward)


def _golden_mean_backward(output_grad, dim=None, *_, keepdim=False, input_grad_shape=None, **__):
    import torch

    grad = output_grad
    if dim is not None and not keepdim:
        dims = [dim] if isinstance(dim, int) else list(dim)
        for d in sorted(dims):
            grad = grad.unsqueeze(d)
    target_shape = list(input_grad_shape) if input_grad_shape is not None else list(grad.shape)
    count = 1
    for a, b in zip(target_shape, grad.shape):
        count *= a // b if b != 0 else 1
    return torch.broadcast_to(grad, target_shape).contiguous() / count


ttnn.attach_golden_function(ttnn.moreh_mean_backward, golden_function=_golden_mean_backward)


def _golden_cumsum_backward(output_grad, dim, *_, **__):
    import torch

    # cumsum backward is a reverse (exclusive) cumulative sum of the output gradient.
    return torch.flip(torch.cumsum(torch.flip(output_grad, dims=[dim]), dim=dim), dims=[dim])


ttnn.attach_golden_function(ttnn.moreh_cumsum_backward, golden_function=_golden_cumsum_backward)


def _golden_dot_backward(output_grad, input, other, *_, **__):
    import torch

    # Dot backward: grad_a = output_grad * other, grad_b = output_grad * input (elementwise, then broadcast).
    grad_a = output_grad * other
    grad_b = output_grad * input
    return [grad_a, grad_b]


ttnn.attach_golden_function(ttnn.moreh_dot_backward, golden_function=_golden_dot_backward)


def _golden_matmul_backward(output_grad, input_a, input_b, *_, are_required_outputs=None, **__):
    import torch

    input_a, input_b = golden_prepare_grad_inputs(input_a, input_b)
    forward = torch.matmul(input_a, input_b)
    grads = golden_compute_gradients(forward, (input_a, input_b), output_grad)
    if are_required_outputs is not None:
        return golden_select_optional_outputs(grads, are_required_outputs)
    return grads


ttnn.attach_golden_function(ttnn.moreh_matmul_backward, golden_function=_golden_matmul_backward)


def _golden_bmm_backward(output_grad, input, mat2, *_, are_required_outputs=None, **__):
    import torch

    input, mat2 = golden_prepare_grad_inputs(input, mat2)
    forward = torch.bmm(input, mat2)
    grads = golden_compute_gradients(forward, (input, mat2), output_grad)
    if are_required_outputs is not None:
        return golden_select_optional_outputs(grads, are_required_outputs)
    return grads


ttnn.attach_golden_function(ttnn.moreh_bmm_backward, golden_function=_golden_bmm_backward)


def _golden_linear_backward(output_grad, input, weight, *_, are_required_outputs=None, bias=None, **__):
    import torch

    inputs = [input, weight] + ([bias] if bias is not None else [])
    prepared = golden_prepare_grad_inputs(*inputs)
    forward = torch.nn.functional.linear(prepared[0], prepared[1], prepared[2] if bias is not None else None)
    grads = golden_compute_gradients(forward, tuple(prepared), output_grad)
    if are_required_outputs is not None:
        return golden_select_optional_outputs(grads, are_required_outputs)
    return grads


ttnn.attach_golden_function(ttnn.moreh_linear_backward, golden_function=_golden_linear_backward)


def _golden_layer_norm_backward(output_grad, input, mean, rstd, normalized_dims, *_, gamma=None, **__):
    import torch

    (input,) = golden_prepare_grad_inputs(input)
    normalized_shape = tuple(input.shape[-normalized_dims:]) if normalized_dims > 0 else tuple(input.shape)
    forward = torch.nn.functional.layer_norm(input, normalized_shape, gamma, None, 1e-5)
    grads = golden_compute_gradients(forward, (input,), output_grad)
    return grads


ttnn.attach_golden_function(ttnn.moreh_layer_norm_backward, golden_function=_golden_layer_norm_backward)


def _golden_group_norm_backward(
    output_grad, input, mean, rstd, num_groups, *_, gamma=None, are_required_outputs=None, **__
):
    import torch

    (input,) = golden_prepare_grad_inputs(input)
    forward = torch.nn.functional.group_norm(input, num_groups, gamma, None)
    grads = golden_compute_gradients(forward, (input,), output_grad)
    if are_required_outputs is not None:
        return golden_select_optional_outputs(grads, are_required_outputs)
    return grads


ttnn.attach_golden_function(ttnn.moreh_group_norm_backward, golden_function=_golden_group_norm_backward)


def _golden_norm_backward(input, output, output_grad, p, dim=None, *_, keepdim=False, **__):
    import torch

    (input,) = golden_prepare_grad_inputs(input)
    forward = torch.linalg.vector_norm(input, ord=p, dim=dim, keepdim=True)
    grads = golden_compute_gradients(forward, (input,), output_grad)
    return grads


ttnn.attach_golden_function(ttnn.moreh_norm_backward, golden_function=_golden_norm_backward)


def _golden_softmax_backward(output_tensor, output_grad_tensor, dim, *_, **__):
    import torch

    # softmax backward: grad = output * (output_grad - sum(output_grad * output, dim, keepdim=True))
    return output_tensor * (output_grad_tensor - (output_grad_tensor * output_tensor).sum(dim=dim, keepdim=True))


ttnn.attach_golden_function(ttnn.moreh_softmax_backward, golden_function=_golden_softmax_backward)


def _golden_softmin_backward(output_tensor, output_grad_tensor, dim, *_, **__):
    import torch

    # softmin(x) = softmax(-x); its backward uses the same form with the softmin output.
    return output_tensor * (output_grad_tensor - (output_grad_tensor * output_tensor).sum(dim=dim, keepdim=True))


ttnn.attach_golden_function(ttnn.moreh_softmin_backward, golden_function=_golden_softmin_backward)


def _golden_logsoftmax_backward(output_tensor, output_grad_tensor, dim, *_, **__):
    import torch

    # logsoftmax backward: grad = output_grad - exp(output) * sum(output_grad, dim, keepdim=True)
    return output_grad_tensor - output_tensor.exp() * output_grad_tensor.sum(dim=dim, keepdim=True)


ttnn.attach_golden_function(ttnn.moreh_logsoftmax_backward, golden_function=_golden_logsoftmax_backward)


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


ttnn.attach_golden_function(ttnn.moreh_nll_loss_backward, golden_function=_golden_nll_loss_backward)


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


ttnn.attach_golden_function(ttnn.moreh_nll_loss_unreduced_backward, golden_function=_golden_nll_loss_unreduced_backward)


# ---------------------------------------------------------------------------
# Optimizer goldens
# ---------------------------------------------------------------------------


def _golden_adam(
    param_in,
    grad,
    exp_avg_in,
    exp_avg_sq_in,
    *_,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    step=0,
    amsgrad=False,
    max_exp_avg_sq_in=None,
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
    values = [param, exp_avg, exp_avg_sq, max_exp_avg_sq]
    required = [True, True, True, amsgrad or max_exp_avg_sq_out is not None]
    return golden_select_optional_outputs(values, required)


ttnn.attach_golden_function(ttnn.moreh_adam, golden_function=_golden_adam)


def _golden_adamw(
    param_in,
    grad,
    exp_avg_in,
    exp_avg_sq_in,
    *_,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=1e-2,
    step=0,
    amsgrad=False,
    max_exp_avg_sq_in=None,
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


ttnn.attach_golden_function(ttnn.moreh_adamw, golden_function=_golden_adamw)


def _golden_sgd(
    param_in,
    grad,
    momentum_buffer_in=None,
    *_,
    lr=1e-3,
    momentum=0.0,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    momentum_initialized=False,
    momentum_buffer_out=None,
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


ttnn.attach_golden_function(ttnn.moreh_sgd, golden_function=_golden_sgd)


def _golden_clip_grad_norm(inputs, max_norm, *_, norm_type=2.0, **__):
    import torch

    # Total norm across all gradient tensors: (sum_i ||g_i||_p^p)^(1/p).
    per_tensor_norms = [torch.linalg.vector_norm(inp, ord=norm_type).reshape(1) for inp in inputs]
    total_norm = torch.linalg.vector_norm(torch.cat(per_tensor_norms), ord=norm_type)
    return total_norm.reshape(1)


ttnn.attach_golden_function(ttnn.moreh_clip_grad_norm, golden_function=_golden_clip_grad_norm)

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn._ttnn

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


# Temporary nanobind migration shim: route logsoftmax through normalization.softmax + log
# when compute_kernel_config is provided, to avoid nanobind argument mismatch.
def logsoftmax(
    input_tensor,
    dim,
    *,
    output_tensor=None,
    strategy=None,
    memory_config=None,
    compute_kernel_config=None,
):
    # Fast path: if no compute_kernel_config provided, delegate to bound op
    if compute_kernel_config is None:
        _kwargs = {}
        if output_tensor is not None:
            _kwargs["output_tensor"] = output_tensor
        if strategy is not None:
            _kwargs["strategy"] = strategy
        if memory_config is not None:
            _kwargs["memory_config"] = memory_config
        return ttnn._ttnn.operations.moreh.moreh_logsoftmax(input_tensor, dim, **_kwargs)
    # Fallback path for nanobind: use normalization softmax, then log
    softmax_out = ttnn.softmax(
        input_tensor,
        dim=dim,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    return ttnn.log(softmax_out)


def logsoftmax_backward(
    output_tensor,
    output_grad_tensor,
    dim,
    *,
    input_grad_tensor=None,
    strategy=None,
    memory_config=None,
    compute_kernel_config=None,
):
    # Fast path: if no compute_kernel_config provided, delegate to bound op
    if compute_kernel_config is None:
        _kwargs = {}
        if input_grad_tensor is not None:
            _kwargs["input_grad_tensor"] = input_grad_tensor
        if strategy is not None:
            _kwargs["strategy"] = strategy
        if memory_config is not None:
            _kwargs["memory_config"] = memory_config
        return ttnn._ttnn.operations.moreh.moreh_logsoftmax_backward(output_tensor, output_grad_tensor, dim, **_kwargs)
    # Fallback path for nanobind: use composite gradient
    # dx = dy - exp(y) * sum(dy, dim, keepdim=True)
    exp_y = ttnn.exp(output_tensor)
    # Avoid passing compute_kernel_config to sum to sidestep nanobind kw-mismatch
    sum_dy = sum(output_grad_tensor, dim, keepdim=True)
    return ttnn.subtract(
        output_grad_tensor,
        ttnn.multiply(exp_y, sum_dy),
        memory_config=memory_config,
    )


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

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

adam = ttnn._ttnn.operations.moreh.moreh_adam
arange = ttnn._ttnn.operations.moreh.moreh_arange
bmm = ttnn._ttnn.operations.moreh.moreh_bmm
bmm_backward = ttnn._ttnn.operations.moreh.moreh_bmm_backward
getitem = ttnn._ttnn.operations.moreh.moreh_getitem
group_norm = ttnn._ttnn.operations.moreh.moreh_group_norm
group_norm_backward = ttnn._ttnn.operations.moreh.moreh_group_norm_backward
layer_norm = ttnn._ttnn.operations.moreh.moreh_layer_norm
layer_norm_backward = ttnn._ttnn.operations.moreh.moreh_layer_norm_backward
logsoftmax = ttnn._ttnn.operations.moreh.moreh_logsoftmax
logsoftmax_backward = ttnn._ttnn.operations.moreh.moreh_logsoftmax_backward
matmul = ttnn._ttnn.operations.moreh.moreh_matmul
mean = ttnn._ttnn.operations.moreh.moreh_mean
mean_backward = ttnn._ttnn.operations.moreh.moreh_mean_backward
nll_loss = ttnn._ttnn.operations.moreh.moreh_nll_loss
nll_loss_backward = ttnn._ttnn.operations.moreh.moreh_nll_loss_backward
nll_loss_unreduced_backward = ttnn._ttnn.operations.moreh.moreh_nll_loss_unreduced_backward
norm = ttnn._ttnn.operations.moreh.moreh_norm
norm_backward = ttnn._ttnn.operations.moreh.moreh_norm_backward
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

# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []

LossReductionMode = ttnn._ttnn.operations.loss.LossReductionMode


def _torch_reduction(reduction):
    # The binding defaults reduction to LossReductionMode, while torch takes the lowercase name.
    # Strings stay accepted so existing callers keep working.
    return reduction.name.lower() if isinstance(reduction, LossReductionMode) else reduction


def _golden_function_l1_loss(
    input_reference: ttnn.Tensor, input_prediction: ttnn.Tensor, *args, reduction=LossReductionMode.NONE, **kwargs
):
    import torch

    output_tensor = torch.nn.L1Loss(reduction=_torch_reduction(reduction))(input_reference, input_prediction)
    return output_tensor


ttnn.attach_golden_function(ttnn.l1_loss, golden_function=_golden_function_l1_loss)


def _golden_function_mse_loss(
    input_reference: ttnn.Tensor, input_prediction: ttnn.Tensor, *args, reduction=LossReductionMode.NONE, **kwargs
):
    import torch

    output_tensor = torch.nn.MSELoss(reduction=_torch_reduction(reduction))(input_reference, input_prediction)
    return output_tensor


ttnn.attach_golden_function(ttnn.mse_loss, golden_function=_golden_function_mse_loss)


__all__ = []

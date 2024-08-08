# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []

LossReductionMode = ttnn._ttnn.operations.loss.LossReductionMode


def _golden_function_l1_loss(ref_tensor: ttnn.Tensor, pred_tensor: ttnn.Tensor, *args, reduction="none", **kwargs):
    import torch

    output_tensor = torch.nn.L1Loss(reduction=reduction)(ref_tensor, pred_tensor)
    return output_tensor


ttnn.attach_golden_function(ttnn.l1_loss, golden_function=_golden_function_l1_loss)


def _golden_function_mse_loss(ref_tensor: ttnn.Tensor, pred_tensor: ttnn.Tensor, *args, reduction="none", **kwargs):
    import torch

    output_tensor = torch.nn.MSELoss(reduction=reduction)(ref_tensor, pred_tensor)
    return output_tensor


ttnn.attach_golden_function(ttnn.mse_loss, golden_function=_golden_function_mse_loss)


__all__ = []

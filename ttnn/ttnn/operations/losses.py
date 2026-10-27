# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []

LossReductionMode = ttnn._ttnn.operations.loss.LossReductionMode


def _torch_reduction(reduction):
    """Map a LossReductionMode to the reduction string Torch losses require; strings pass through."""

    if isinstance(reduction, LossReductionMode):
        return {
            LossReductionMode.NONE: "none",
            LossReductionMode.MEAN: "mean",
            LossReductionMode.SUM: "sum",
        }[reduction]
    return reduction


def _reduced_loss(loss_module, ref_tensor, pred_tensor, reduction):
    """Evaluate a loss, giving scalar reductions the device's float32-reference ULP contract."""

    import torch

    reduction = _torch_reduction(reduction)
    if reduction == "none":
        return loss_module(reduction=reduction)(ref_tensor, pred_tensor)
    # A reduced loss is a single value, so PCC is undefined; the device accumulates in float32
    # and is specified within three ULP of that reference.
    output_tensor = loss_module(reduction=reduction)(ref_tensor.float(), pred_tensor.float()).to(ref_tensor.dtype)
    return ttnn.decorators.set_golden_comparison_config(
        output_tensor, method="ulp", scope="degenerate", ulp_threshold=3
    )


def _golden_function_l1_loss(ref_tensor: ttnn.Tensor, pred_tensor: ttnn.Tensor, *args, reduction="none", **kwargs):
    import torch

    return _reduced_loss(torch.nn.L1Loss, ref_tensor, pred_tensor, reduction)


ttnn.attach_golden_function(ttnn.l1_loss, golden_function=_golden_function_l1_loss)


def _golden_function_mse_loss(ref_tensor: ttnn.Tensor, pred_tensor: ttnn.Tensor, *args, reduction="none", **kwargs):
    import torch

    return _reduced_loss(torch.nn.MSELoss, ref_tensor, pred_tensor, reduction)


ttnn.attach_golden_function(ttnn.mse_loss, golden_function=_golden_function_mse_loss)


__all__ = []

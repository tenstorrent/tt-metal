# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of ttnn.bringup.rms_norm (ai-generated drop-in for ttnn.rms_norm), for the fork's model-case tests.

y = t / sqrt(mean(t^2, -1) + eps) * weight + bias, with t = x + residual (both optional). Source: the op's docstring
(rms_norm_ttnn.py) and its acceptance test (tests/unit/test_rms_norm_ttnn.py)."""

from __future__ import annotations

import torch


def rms_norm(x, *, epsilon, weight=None, bias=None, residual=None):
    t = x.to(torch.float64)
    if residual is not None:
        t = t + residual.to(torch.float64)
    y = t / torch.sqrt(torch.mean(t * t, dim=-1, keepdim=True) + epsilon)
    if weight is not None:
        y = y * weight.to(torch.float64).reshape(-1)
    if bias is not None:
        y = y + bias.to(torch.float64).reshape(-1)
    return y

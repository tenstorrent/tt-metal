# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []


def _golden_function_fft(input_tensor, dim=-1, *args, **kwargs):
    import torch

    return torch.fft.fft(input_tensor, dim=dim)


ttnn.attach_golden_function(ttnn.fft.fft, golden_function=_golden_function_fft)


def _golden_function_ifft(input_tensor, dim=-1, *args, **kwargs):
    import torch

    return torch.fft.ifft(input_tensor, dim=dim)


ttnn.attach_golden_function(ttnn.fft.ifft, golden_function=_golden_function_ifft)

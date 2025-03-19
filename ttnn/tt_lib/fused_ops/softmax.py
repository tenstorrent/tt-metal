# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from tt_lib.utils import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax


def softmax(x: ttnn.Tensor, stable=False):
    """
    Performs Softmax on a ``ttnn.Tensor``.
    """

    if stable:
        sumsW = ttnn.max(x, 3)
        z = ttnn.subtract(x, sumsW)  # x-max(x)
    else:
        z = x
    numerator = ttnn.exp(z)  # exp(z)
    denom1 = ttnn.sum(numerator, 3)  # torch.sum(x, 3)
    denom = ttnn.reciprocal(denom1)
    output = ttnn.multiply(numerator, denom)

    return output


def ref_stable_softmax(x):
    """
    z = x - torch.max(x, dim=3, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, 3)
    denom1 = torch.reciprocal(denominator)
    softmax = numerator*denom1
    """
    softmax = torch.nn.Softmax(3)(x)

    return softmax


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    H, W = 64, 96
    torch.manual_seed(123)

    x = torch.randn((1, 1, H, W))
    ref_sm = ref_stable_softmax(x)

    x_t = tilize_to_list(x)
    t0 = ttnn.Tensor(x_t, [1, 1, H, W], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    func = softmax
    t1 = func(t0)

    tt_got_back = t1.cpu().to_torch()
    tt_got_back = untilize(tt_got_back)

    print("Max diff=")
    print_diff_argmax(tt_got_back, ref_sm)

    ttnn.close_device(device)

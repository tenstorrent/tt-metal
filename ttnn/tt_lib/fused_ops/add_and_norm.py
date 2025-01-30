# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from .layernorm import Layernorm


def AddAndNorm(gamma: ttnn.Tensor, beta: ttnn.Tensor, epsilon, H, W, device):
    """
    Returns a function that performs Eltwise-binary add two
    ``ttnn.Tensor`` s and then LayerNorm the result.
    """

    layernorm = Layernorm(gamma, beta, epsilon, H, W, device, 1)

    def add_and_norm_(activationa, activationb):
        a_plus_b = ttnn.add(activationa, activationb)
        H = activationa.padded_shape[2]
        lnorm_a_plus_b = layernorm(a_plus_b, overrideH=H)
        return lnorm_a_plus_b

    return add_and_norm_

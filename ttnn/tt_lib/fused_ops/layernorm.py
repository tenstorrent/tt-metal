# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from tt_lib.utils import (
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
    tt2torch as t2t,
    tt2torch_rm as t2trm,
    roundup32,
    float_to_bits,
)


# This ref implementation is only here for debugging
def ref_ln(x, gamma, beta=None, epsilon=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + epsilon).sqrt()
    invstd = 1.0 / std
    y1 = (x - mean) * invstd
    y = y1.clone()
    if gamma is not None:
        y *= gamma
    if beta is not None:
        y += beta
    return y, mean, var, std, invstd, y1


# TODO(AP): refactor to support any num_dims
def Layernorm(gamma: float, beta: float, epsilon: float, H, W, device, num_dims=2):
    """
    Returns a function that performs LayerNorm with parameters.

    H, W correspond to normalized_shape in pytorch Layernorm spec

    *Note*: Note that the only ``num_dims`` supported at the moment is ``2``.
    """

    num_dims_ = num_dims
    assert num_dims == 1

    # gamma, beta, epsilon should be tt::tensors of size 32*W
    # with a single populated top row
    # H, W need to be from the "true" shape (unpadded)
    assert gamma is None or len(gamma) == W * 32  # single H-tile
    assert beta is None or len(beta) == W * 32  # single H-tile

    H_ = H
    W_ = W
    padded_h = roundup32(H)
    if num_dims == 1:
        padded_h = 32
    padded_w = roundup32(W)
    gamma_ = ttnn.Tensor(gamma, [1, 1, padded_h, padded_w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    beta_ = None
    if beta is not None:
        beta_ = ttnn.Tensor(beta, [1, 1, padded_h, padded_w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    epsilon_ = ttnn.Tensor(
        [epsilon] + [0.0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    if num_dims == 2:
        var_scaler_ = ttnn.Tensor(
            [1 / (H * W)] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
    else:
        # For num_dims==1 var_scaler_ is implemented using dynamic mask
        assert num_dims == 1

    # 1D variant
    # TODO(AP): merge with 2d? refactor.
    def layernorm_1d_(x, overrideH=None, refx=None, refgamma=None, refbeta=None):
        N = x.padded_shape[0]
        C = x.padded_shape[1]
        H = x.padded_shape[2]
        W = x.padded_shape[3]

        H_ = 1
        if overrideH is not None:
            H_ = overrideH

        # first compute the mean (m)
        means = ttnn.sum(x, 3, scalar=1.0 / W)  # -> NCH1
        x_minus_mean = ttnn.subtract(x, means, BCSUB, BCW)  # need to blank out the H for non-multiple of 32
        if False and refx is not None:
            ry, rmean, rvar, rstd, rinvstd, ry1 = ref_ln(refx, refgamma, refbeta)

        var = ttnn.multiply(x_minus_mean, x_minus_mean)  # (x-m)^2
        var_redW = ttnn.sum(var, 3)  # sum[(x-m)^2]

        scaler = 1 / W
        var_scaler_ = ttnn.fill_rm(1, 1, roundup32(H), 32, H_, 1, epsilon_, scaler, 0)
        var_scaler_ = ttnn.tilize(var_scaler_)

        var_div_n1 = ttnn.multiply(var_redW, var_scaler_)
        var_plus_eps = ttnn.add(var_div_n1, epsilon_)

        var_sqrt = ttnn.sqrt(var_plus_eps)
        inv_sqrt = ttnn.reciprocal(var_sqrt)
        if False and refx is not None:
            qq = t2t(inv_sqrt)[0, 0, 0:9, 0]

        x_div_sqrt = ttnn.multiply(x_minus_mean, inv_sqrt)

        if False and refx is not None:
            qq1 = t2t(x_div_sqrt)[0, 0, 0:9, :]

        x_gamma = ttnn.multiply(x_div_sqrt, gamma_)
        if beta_ is not None:
            x_beta = ttnn.add(x_gamma, beta_)
            return x_beta
        else:
            return x_gamma

    def layernorm_2d_(x):
        N = x.padded_shape[0]
        C = x.padded_shape[1]
        H = x.padded_shape[2]
        W = x.padded_shape[3]

        # first compute the mean (m)
        redW = ttnn.sum(x, 3, scalar=1.0 / W)  # -> NCH1
        mean = ttnn.sum(redW, 2)  # -> NC11 (HW reduce doesn't behave well with small scaler)
        x_minus_mean0 = ttnn.subtract(x, mean)  # need to blank out the H for non-multiple of 32
        hmasku = ttnn.fill_ones_rm(N, C, H, 32, 1, 1, x)  # generate a H-mask with mask[h, w] = 1.0 where h,w < 1
        hmaskt = ttnn.tilize(hmasku)  # tilize the mask
        x_minus_mean = ttnn.multiply(x_minus_mean0, hmaskt)  # zero out (x-m) for h>=H_, h<H

        var = ttnn.multiply(x_minus_mean, x_minus_mean)  # (x-m)^2
        var_redW = ttnn.sum(var, 3)  # sum[(x-m)^2]
        var_redHW = ttnn.sum(var_redW, 2)  # sum[(x-m)^2]
        var_div_n1 = ttnn.multiply(var_redHW, var_scaler_)  # *= 1/(everything not batch)
        var_plus_eps = ttnn.add(var_div_n1, epsilon_)

        var_sqrt = ttnn.sqrt(var_plus_eps)
        inv_sqrt = ttnn.reciprocal(var_sqrt)

        x_div_sqrt = ttnn.multiply(x_minus_mean, inv_sqrt)
        x_gamma = ttnn.multiply(x_div_sqrt, gamma_)
        if beta_ is not None:
            x_beta = ttnn.add(x_gamma, beta_)
            return x_beta
        else:
            return x_gamma

    # unbiased_var = [(x-m)^2]/(n-1)
    # m = E[x]
    # var = E[(x-m)^2]
    # result = (x - E[x])/sqrt(var+epsilon)*gamma+beta
    def layernorm_(x, overrideH=None, refx=None, refgamma=None):
        if num_dims_ == 1:
            return layernorm_1d_(x, overrideH, refx, refgamma)

        assert num_dims_ == 2  # Only 1d and 2d are supported at the moment
        return layernorm_2d_(x)

    return layernorm_


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)


if __name__ == "__main__":
    # Initialize the device
    device = ttnn.CreateDevice(0)

    H = 64
    W = 96
    epsf = 1e-4
    betaf = 0.345
    gammaf = 0.123
    torch.manual_seed(123)
    x = torch.randn((1, 1, H, W))
    ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)

    gamma = pad_weight(torch.full((1, 1, 1, W), gammaf))
    beta = pad_weight(torch.full((1, 1, 1, W), betaf))

    t0 = ttnn.Tensor(tilize_to_list(x), [1, 1, H, W], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    ttgamma = tilize_to_list(gamma)
    ttbeta = tilize_to_list(beta)
    func = Layernorm(ttgamma, ttbeta, epsf, 1, W, device, num_dims=1)

    t1 = func(t0, overrideH=H)

    tt_got_back = t1.cpu().to_torch()
    tt_got_back = untilize(tt_got_back)

    print("Layernorm max absdiff=")
    print_diff_argmax(tt_got_back, ref_lnorm)

    ttnn.CloseDevice(device)

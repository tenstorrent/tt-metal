# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post in torch (float64), from the op's docstring: comb applied transposed."""

import torch


def mhc_post(f, x, post, comb, comb_transposed=True):
    """X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, i*n + j] * X[t, i*C:(i+1)*C];
    comb_transposed=False: sum_i comb[t, j*n + i] * X[t, i*C:(i+1)*C]."""
    f, x, post, comb = f.double(), x.double(), post.double(), comb.double()
    n = post.shape[-1]
    C = f.shape[-1]
    xs = x.reshape(*x.shape[:-1], n, C)
    m = comb.reshape(*comb.shape[:-1], n, n)
    if not comb_transposed:
        m = m.transpose(-1, -2)
    out = post.unsqueeze(-1) * f.unsqueeze(-2) + torch.einsum("...ij,...ic->...jc", m, xs)
    return out.reshape(*x.shape)


def mhc_post_sumsq(f, x, post, comb):
    """stats_weight's column n (n + 2): sum over the row of X'^2, X' the unrounded mix (comb applied transposed, as
    mhc_post), computed in the inputs' dtype (float64 for the reference, float32 for noise_limit's formats run)."""
    n = post.shape[-1]
    C = f.shape[-1]
    xs = x.reshape(*x.shape[:-1], n, C)
    m = comb.reshape(*comb.shape[:-1], n, n)
    mix = post.unsqueeze(-1) * f.unsqueeze(-2) + torch.einsum("...ij,...ic->...jc", m, xs)
    return mix.reshape(*x.shape).square().sum(-1)


# FPU matmul operand precision per fidelity on Blackhole (deepseek_v41_flash_d_p O.1 probe mhc_post_ds41, DS41-42):
# the FPU multiplies in0 (srcB) and in1 (srcA) with these explicit mantissa bits, the rest truncated; products and
# sums are fp32 (the in-tile partial dot ~TF32, DS41-16: ~3e-4 of the device vs this model on a 10240-long dot).
FPU_BITS = {"LoFi": (6, 4), "HiFi2": (6, 9), "HiFi3": (10, 9), "HiFi4": (10, 9)}


def _truncate(t, bits):
    i = t.float().contiguous().view(torch.int32)
    sh = 23 - bits
    return ((i >> sh) << sh).view(torch.float32)


def fpu_operands(a, b, fidelity):
    """(a, b) of a ttnn matmul a @ b as the FPU multiplies them at ``fidelity`` (float32, truncated mantissas). Only
    for the formats' run of noise_limit: the float64 reference takes the operands as they are."""
    ka, kb = FPU_BITS[fidelity]
    return _truncate(a, ka), _truncate(b, kb)

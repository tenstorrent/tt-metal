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

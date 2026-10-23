# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of the flat routed expert (ttnn.bringup.flat_routed_expert / flat_routed_expert_with_program), for
the fork's model-case tests: per flat row, the gated FFN of its expert, y = act(x W1^T, x W3^T) W2^T (weights as
nn.Linear stores them, [N, K], weights_nk), act = flat_expert.act_ref.

``formats``: the op's real formats for noise_limit's formats run (float32): x and h as the tiles the op multiplies
(bfp8 unless x_bf16 / h_bf16; fork_cases.round_to's block-float model, groups of 16 along K), the FPU's operand
mantissa bits at the call's fidelity (x / h in0, the weights in1: FPU_BITS, truncated; DS41-42, DS41-108: on this op
at LoFi with bf16 x / h tiles the device lands 2.8e-3 from this model, 5.5e-2 from float64), every product and sum
fp32 (fp32 DEST), y rounded to bf16. The bfp8 packer's own rounding (ties away, DS41-103) is not modelled bit for bit:
the device is ~1.8e-2 from this model there, inside its distance from float64 (deepseek_v41_flash_d_p O.1 probes)."""

from __future__ import annotations

import torch

from models.demos.common.bringup.testing.fork_cases import round_to

FPU_BITS = {"LoFi": (6, 4), "HiFi2": (6, 9), "HiFi3": (10, 9), "HiFi4": (10, 9)}
FIDELITY = {0: "LoFi", 2: "HiFi2", 3: "HiFi3", 4: "HiFi4"}  # ttnn.MathFidelity values


def act_ref(g, u, act):
    if act == "clamped_silu":  # DeepSeek V4: limit 10 (flat_expert.act_ref)
        return torch.nn.functional.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0)
    raise ValueError(act)


def truncate(t, bits):
    """float32 t with only ``bits`` explicit mantissa bits (the rest dropped toward zero)."""
    i = t.float().contiguous().view(torch.int32)
    sh = 23 - bits
    return ((i >> sh) << sh).view(torch.float32)


def bf16_away(t):
    """float32 -> bf16 values, to nearest with ties away from zero (the packer)."""
    b = t.float().contiguous().view(torch.int32)
    sign = b & torch.tensor(-0x80000000, dtype=torch.int32)
    mag = ((b & 0x7FFFFFFF) + 0x8000) & ~0xFFFF
    return (sign | mag).view(torch.float32)


def block_round(t, mant, ties="away", clamp=False):
    """t as a block-float tile holds it: per 16 values of a row (the last dim), one shared exponent E (frexp of the
    group's largest magnitude), each value an integer multiple of 2^(E - mant), |q| < 2^mant (bfp8: mant 7, bfp4: 3).
    ties: "away" (the packer) or "even". clamp: keep |q| <= 2^mant - 1 (a rounding up of the largest value would need
    the next exponent: values a host packs exactly)."""
    x = t.float()
    n = x.shape[-1]
    assert n % 16 == 0, n
    g = x.reshape(*x.shape[:-1], n // 16, 16)
    _, e = torch.frexp(g.abs().amax(dim=-1, keepdim=True))
    q = torch.ldexp(g, mant - e)
    q = torch.sign(q) * torch.floor(q.abs() + 0.5) if ties == "away" else torch.round(q)
    if clamp:
        q = q.clamp(-(2**mant - 1), 2**mant - 1)
    return torch.ldexp(q, e - mant).reshape(x.shape)


def ffn(x, w1, w3, w2, act, formats=None):
    """y [rows, H] of one expert on rows x [rows, H] in x's dtype. ``formats`` (float32 only): dict(x_bf16, h_bf16,
    fidelity) -> the op's arithmetic (module docstring)."""
    if formats is None or x.dtype != torch.float32:
        return act_ref(x @ w1.T, x @ w3.T, act) @ w2.T
    b0, b1 = FPU_BITS[formats["fidelity"]]
    xq = truncate(round_to(x, "bf16" if formats["x_bf16"] else "bfp8"), b0)
    w1, w3, w2 = truncate(w1, b1), truncate(w3, b1), truncate(w2, b1)
    h = act_ref(xq @ w1.T, xq @ w3.T, act)
    h = round_to(h, "bf16" if formats["h_bf16"] else "bfp8")
    return round_to(truncate(h, b0) @ w2.T, "bf16")

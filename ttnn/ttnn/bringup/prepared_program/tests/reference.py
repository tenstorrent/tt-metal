# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of the programs the model cases run through the prepared path (cases.py), for their reference
checks. Given float64 tensors: the exact math. Given float32 tensors (fork_cases.noise_limit's formats run, inputs
already rounded to their stored formats): the device's arithmetic - the FPU's operand mantissa bits at the matmul's
fidelity (truncated: activations in0, weights in1; as the flat routed expert's reference, DS41-42), each tile's 32-long
partial dot rounded to ~TF32 (10 mantissa bits; known issue K-30, DS41-16) and fp32 sums of those, the intermediate
tiles' format, the output rounded by noise_limit.

Fused MoE head (models/demos/moe_block/tt/fused_head.py, DeepSeek-V4.1):
- shared expert: silu(min(x W1^T, limit)) clamp(x W3^T, +-limit) W2^T; weights bfp8 at the case's fidelity, h bf16 tiles;
- router: scores sqrt(softplus(x W_r^T)) (logits at HiFi3: bf16 x and W_r exact in the FPU, fp32 sums; softplus and
  sqrt in fp32), top-K by score + bias as a top-K key - the fp32 key with its low mantissa bits holding the expert id
  (router_stream.py rt_key / em_of: 8 bits up to 256 experts), so a key keeps 15 mantissa bits; weights = the picks'
  scores / their sum x route_scale, bf16."""

from __future__ import annotations

import torch

FPU_BITS = {"LoFi": (6, 4), "HiFi2": (6, 9), "HiFi3": (10, 9), "HiFi4": (10, 9)}  # (in0, in1) mantissa bits


def truncate(t, bits):
    """float32 t with only ``bits`` explicit mantissa bits (the rest dropped toward zero)."""
    i = t.float().contiguous().view(torch.int32)
    sh = 23 - bits
    return ((i >> sh) << sh).view(torch.float32)


def tf32(t):
    """float32 t rounded to 10 explicit mantissa bits, to nearest even."""
    i = t.float().contiguous().view(torch.int32)
    sh = 13
    i = (i + (1 << (sh - 1)) - 1 + ((i >> sh) & 1)) >> sh << sh
    return i.view(torch.float32)


def mm(a, b):
    """a @ b as the FPU sums it (float32 operands as already truncated): per 32-long K block a ~TF32 partial dot, fp32
    sums of the blocks."""
    acc = None
    for k in range(0, a.shape[-1], 32):
        p_ = tf32(a[..., k : k + 32] @ b[k : k + 32])
        acc = p_ if acc is None else acc + p_
    return acc


def bf16(t):
    return t.to(torch.bfloat16).float()


def shared_expert(x, gT, uT, dT, limit, fidelity):
    """x [T, H]; gT / uT [H, I], dT [I, H] (the engine's [K, N] weights) -> [T, H]."""
    if x.dtype != torch.float32:
        return torch.nn.functional.silu((x @ gT).clamp(max=limit)) * (x @ uT).clamp(-limit, limit) @ dT
    b0, b1 = FPU_BITS[fidelity]
    xq = truncate(x, b0)
    g, u = mm(xq, truncate(gT, b1)), mm(xq, truncate(uT, b1))
    h = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
    return mm(truncate(bf16(h), b0), truncate(dT, b1))


def router_scores(x, wr):
    """x [T, H], W_r [E, H] -> sqrt(softplus(x W_r^T)) [T, E] (float32: HiFi3 logits, fp32 math)."""
    if x.dtype != torch.float32:
        return torch.sqrt(torch.nn.functional.softplus(x @ wr.T))
    b0, b1 = FPU_BITS["HiFi3"]
    return torch.sqrt(torch.nn.functional.softplus(mm(truncate(x, b0), truncate(wr, b1).T.contiguous())))


def topk_keys(scores, bias, key_bits):
    """The top-K keys: score + bias (float32: as the fp32 key holds it, key_bits mantissa bits)."""
    k = scores + bias
    return truncate(k, key_bits) if scores.dtype == torch.float32 else k


def route_weights(x, wr, ids, scale):
    """The weights of the given picks ids [T, K]: their scores / their sum x scale."""
    s = router_scores(x, wr).gather(1, ids)
    return s / s.sum(-1, keepdim=True) * scale

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ckv_bfp4 in torch: tt-metal's host bfloat4_b rule (blockfloat_common.cpp convert_u32_to_bfp) per 16-value group of
a row, as a 288-byte row record (32 exponents, then nibbles, value 2i in the low nibble of byte i)."""

import torch


def encode_rows(x: torch.Tensor, exp_twice: bool = True) -> torch.Tensor:
    """[n, 512] (any float) -> [n, 320] uint8 (the 32 exponents twice, then the nibbles); exp_twice=False: [n, 288]
    (tt/ckv_codec.py format B)."""
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp, sign = (u >> 23) & 0xFF, u >> 31
    man = (u & 0x7FFFFF) | (1 << 23)
    eg = exp.view(-1, 32, 16)
    shared = eg.amax(-1, keepdim=True)
    m = man.view(-1, 32, 16) >> (shared - eg).clamp(max=62)
    q, rv = m >> 21, m & ((1 << 21) - 1)
    q = (q + ((rv > (1 << 20)) | ((rv == (1 << 20)) & ((q & 1) == 1))).long()).clamp(max=7)
    q = torch.where(eg == 0, torch.zeros_like(q), q)
    s = torch.where(q == 0, torch.zeros_like(q), sign.view(-1, 32, 16))
    c = ((s << 3) | q).view(-1, 256, 2)
    e = shared.squeeze(-1)
    parts = [e, e] if exp_twice else [e]
    return torch.cat(parts + [c[..., 0] | (c[..., 1] << 4)], -1).to(torch.uint8)


def decode_rows(b: torch.Tensor) -> torch.Tensor:
    """[n, 320] (or [n, 288]) uint8 -> [n, 512] fp32."""
    e = b[:, :32].long()
    nb = b[:, b.shape[1] - 256 :].long()
    c = torch.stack([nb & 15, nb >> 4], -1).view(-1, 32, 16)
    v = (c & 7).double() * torch.pow(2.0, (e - 129).double()).unsqueeze(-1)
    return torch.where((c >> 3) == 1, -v, v).view(-1, 512).float()

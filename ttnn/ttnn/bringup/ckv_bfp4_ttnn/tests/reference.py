# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ckv_bfp4 / bfp_round in torch: tt-metal's host bfloat4_b / bfloat8_b rule (blockfloat_common.cpp convert_u32_to_bfp)
per 16-value group of a row, as a row record (32 exponents (twice), then nibbles, value 2i in the low nibble of byte
i; bfloat8_b: one sign-magnitude byte a value) or as values."""

import torch


def _codes(x: torch.Tensor, mbits: int):
    """x [.., 16 k] -> (shared exponents [.., k], sign-magnitude codes [.., 16 k]) by the host rule."""
    shape = x.shape
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp, sign = (u >> 23) & 0xFF, u >> 31
    man = (u & 0x7FFFFF) | (1 << 23)
    eg = exp.view(-1, 16)
    shared = eg.amax(-1, keepdim=True)
    m = man.view(-1, 16) >> (shared - eg).clamp(max=62)
    sh = 24 - mbits
    q, rv, half = m >> sh, m & ((1 << sh) - 1), 1 << (sh - 1)
    q = (q + ((rv > half) | ((rv == half) & ((q & 1) == 1))).long()).clamp(max=(1 << mbits) - 1)
    q = torch.where(eg == 0, torch.zeros_like(q), q)
    s = torch.where(q == 0, torch.zeros_like(q), sign.view(-1, 16))
    return shared.view(*shape[:-1], -1), ((s << mbits) | q).view(shape)


def round_values(x: torch.Tensor, mbits: int = 3) -> torch.Tensor:
    """x [.., 16 k] -> fp32 values on the grid (groups of 16 along the last dim)."""
    e, c = _codes(x, mbits)
    v = (c & ((1 << mbits) - 1)).double().view(*c.shape[:-1], -1, 16)
    v = v * torch.pow(2.0, (e - 127 - (mbits - 1)).double()).unsqueeze(-1)
    return torch.where((c >> mbits).view(v.shape) == 1, -v, v).view(x.shape).float()


def encode_rows(x: torch.Tensor, exp_twice: bool = True, mbits: int = 3) -> torch.Tensor:
    """[n, 512] (any float) -> [n, 320] uint8 (the 32 exponents twice, then the nibbles); exp_twice=False: [n, 288]
    (tt/ckv_codec.py format B); mbits=7: bfloat8_b, [n, 576] (one byte a value)."""
    e, c = _codes(x.reshape(-1, 512), mbits)
    if mbits == 3:
        c = c.view(-1, 256, 2)
        c = c[..., 0] | (c[..., 1] << 4)
    parts = [e, e] if exp_twice else [e]
    return torch.cat(parts + [c], -1).to(torch.uint8)


def decode_rows(b: torch.Tensor) -> torch.Tensor:
    """[n, 320] (or [n, 288]; bfloat8_b [n, 576]) uint8 -> [n, 512] fp32."""
    e = b[:, :32].long()
    if b.shape[1] == 576:
        c = b[:, 64:].long().view(-1, 32, 16)
        v = (c & 127).double() * torch.pow(2.0, (e - 133).double()).unsqueeze(-1)
        return torch.where((c >> 7) == 1, -v, v).view(-1, 512).float()
    nb = b[:, b.shape[1] - 256 :].long()
    c = torch.stack([nb & 15, nb >> 4], -1).view(-1, 32, 16)
    v = (c & 7).double() * torch.pow(2.0, (e - 129).double()).unsqueeze(-1)
    return torch.where((c >> 3) == 1, -v, v).view(-1, 512).float()

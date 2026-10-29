# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""topk_large_indices in torch: per row, the k largest of the first n (valid_length, else the width) values,
descending; ids uint32, a slot past the row's finite values is the sentinel 0xFFFFFFFF (-inf never selected)."""

import torch

SENTINEL = 0xFFFFFFFF


def topk_values(x: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """[rows, width] -> [rows, k] float32: the k largest of x[:, :n], descending, -inf where a row has fewer."""
    return x[:, :n].float().topk(k, dim=-1).values


def check(x: torch.Tensor, ids: torch.Tensor, k: int, n: int) -> str | None:
    """None if ids ([rows, k] int64) is a valid answer for x ([rows, width]); else what is wrong. Ties may take any of
    the tied ids, so the values the ids point at are compared, position by position, with topk_values."""
    want = topk_values(x, k, n)
    finite = torch.isfinite(want)
    sent = ids == SENTINEL
    if not torch.equal(sent, ~finite):
        return f"sentinel slots differ on {int((sent != ~finite).any(-1).sum())} rows"
    idc = ids.clamp(max=n - 1)
    if (~sent).any() and int(ids[~sent].max()) >= n:
        return "an id past valid_length"
    for r in range(ids.shape[0]):
        row = ids[r][~sent[r]]
        if row.unique().numel() != row.numel():
            return f"row {r}: an id twice"
    got = torch.gather(x[:, :n].float(), -1, idc).masked_fill(sent, float("-inf"))
    bad = (got != want).any(-1).nonzero().flatten()
    return None if bad.numel() == 0 else f"{bad.numel()} rows pick other values (first {bad[:6].tolist()})"

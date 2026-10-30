# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.topk_large_indices against the stock ttnn.experimental.topk_large_indices on the same input: the same
selection (position by position the same values, sentinels in the same slots; ids differ only between keys of equal
bf16 score, whose order neither op defines; printed with each op's tie statistics), at DeepSeek-V4.1's per-chip shapes on the Galaxy
8x4 (160 query rows a chip after P.8 / P.10): the index top-k 512 over the ratio-2 keys (28160 at the ladder's last
chunk, 522240 at 1M) and the ratio-1 keys (56320 / 1044480), the candidate blocks' top 2048 (7040 / 130560), and
the 2x2's 512-row shapes. Inputs: plain bf16 randn (bf16 ties), heavily tied rows (16 levels), the causal score
layout (each row's future keys -inf, the score op's), and the re-index layout (whole blocks of 8 keys -inf outside
2048 candidate blocks). Every result is also checked against torch's selection (reference.py)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_ref_spec = importlib.util.spec_from_file_location(
    "bringup_topk_large_indices_tests_reference_unit", Path(__file__).resolve().parents[1] / "reference.py"
)
ref = importlib.util.module_from_spec(_ref_spec)
_ref_spec.loader.exec_module(ref)


def _scores(kind: str, rows: int, width: int, valid: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=g)
    if kind == "tied":  # 16 levels: most of a row ties at the cutoff
        x = (x * 4).round().clamp(-8, 7)
    elif kind == "causal":  # row i sees the first valid - rows + i + 1 keys (the newest chunk's queries), the rest -inf
        vis = valid - rows + 1 + torch.arange(rows).view(rows, 1)
        x = x.masked_fill(torch.arange(width).view(1, width) >= vis, float("-inf"))
    elif kind == "blocks":  # the re-index mask: 2048 candidate blocks of 8 keys a row stay, every other key -inf
        nb = valid // 8
        keep = torch.zeros(rows, nb, dtype=torch.bool)
        pick = torch.rand(rows, nb, generator=g).argsort(-1)[:, : min(2048, nb)]
        keep.scatter_(1, pick, True)
        x[:, :valid] = x[:, :valid].masked_fill(~keep.repeat_interleave(8, -1), float("-inf"))
    return x.bfloat16().reshape(1, 1, rows, width)


def _ties(x: torch.Tensor, ids: torch.Tensor) -> str:
    """How an answer breaks ties: of the rows whose cutoff value has more holders than slots, how many keep the lowest
    / the highest ids of it; of the runs of equal values in the output, how many list their ids ascending."""
    rows, k = ids.shape
    over = low = high = runs = asc = 0
    for r in range(rows):
        sel = ids[r][ids[r] != ref.SENTINEL]
        if sel.numel() == 0:
            continue
        v = x[r].float()[sel]
        cut = v[-1]
        holders = (x[r].float() == cut).nonzero().flatten()
        chosen = sel[v == cut].sort().values
        if holders.numel() > chosen.numel():
            over += 1
            low += int(torch.equal(chosen, holders[: chosen.numel()]))
            high += int(torch.equal(chosen, holders[-chosen.numel() :]))
        i = 0
        while i < sel.numel():
            j = i
            while j + 1 < sel.numel() and v[j + 1] == v[i]:
                j += 1
            if j > i:
                runs += 1
                asc += int(bool((sel[i + 1 : j + 1] > sel[i:j]).all()))
            i = j + 1
    return f"cutoff-tie rows {over}: lowest ids {low}, highest {high}; tied runs {runs}: ascending {asc}"


SHAPES = [
    # (id, rows, width, valid, k)
    ("8x4-r2-k512-28160", 160, 28160, 28160, 512),
    ("8x4-r1-k512-56320", 160, 56320, 56320, 512),
    ("8x4-cand-k2048-7040", 160, 7040, 7040, 2048),
    ("8x4-1M-r2-k512-522240", 160, 522240, 522240, 512),
    ("8x4-1M-r2-k512-522240-v100000", 160, 522240, 100000, 512),
    ("8x4-1M-r1-k512-1044480", 160, 1044480, 1044480, 512),
    ("8x4-1M-cand-k2048-130560", 160, 130560, 130560, 2048),
    ("2x2-r2-k512-27648", 512, 27648, 27648, 512),
    ("2x2-cand-k2048-6912", 512, 6912, 6912, 2048),
]
KINDS = {
    512: ["randn", "tied", "causal", "blocks"],
    2048: ["randn", "tied", "causal"],
}


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "sid, rows, width, valid, k, kind",
    [(s[0], *s[1:], kind) for s in SHAPES for kind in KINDS[s[4]] if not (kind == "blocks" and s[3] // 8 < 4096)],
    ids=[f"{s[0]}-{kind}" for s in SHAPES for kind in KINDS[s[4]] if not (kind == "blocks" and s[3] // 8 < 4096)],
)
def test_fork_selects_as_stock(device, sid, rows, width, valid, k, kind):
    x = _scores(kind, rows, width, valid, seed=rows + width + valid)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    kw = {} if valid == width else {"valid_length": valid}
    fork = ttnn.to_torch(ttnn.bringup.topk_large_indices(t, k=k, **kw)).to(torch.int64) & 0xFFFFFFFF
    stock = ttnn.to_torch(ttnn.experimental.topk_large_indices(t, k=k, **kw)).to(torch.int64) & 0xFFFFFFFF
    fork, stock = fork.reshape(rows, k), stock.reshape(rows, k)
    err = ref.check(x[0, 0], fork, k, valid)
    assert err is None, f"fork vs torch: {err}"
    assert ref.check(x[0, 0], stock, k, valid) is None, "stock vs torch"
    print(f"\n{sid}-{kind}: fork {_ties(x[0, 0, :, :valid], fork)} | stock {_ties(x[0, 0, :, :valid], stock)}")
    # the same selection: position by position the fork's ids point at the stock's values (so every id difference is
    # between keys of equal bf16 score, and sentinels sit at the same slots)
    xs = x[0, 0, :, :valid].float()
    val = lambda ids: torch.gather(xs, -1, ids.clamp(max=valid - 1)).masked_fill(ids == ref.SENTINEL, float("-inf"))
    assert torch.equal(val(fork), val(stock)), "the fork selects other values than the stock op"
    # ids: the tie order is each op's own (no defined rule: neither keeps the lowest or the highest tied ids, nor lists
    # a tied run ascending; the fork splits a row across MATH / PACK and core pairs where the stock op does not), so
    # equal ids are not required; reported for the record
    pos = (fork != stock).any(-1)
    sets = torch.tensor([set(fork[r].tolist()) != set(stock[r].tolist()) for r in range(rows)])
    print(
        f"{sid}-{kind}: rows with other ids than the stock op: by position {int(pos.sum())} / {rows}, as sets {int(sets.sum())}"
    )

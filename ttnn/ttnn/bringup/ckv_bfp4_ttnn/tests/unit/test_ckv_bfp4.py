# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ckv_bfp4 on one device: the row records equal the host bfloat4_b rule byte for byte, rows outside
[row0, row0 + n) are untouched, the rounded bf16 output holds the records' values, and the op is deterministic."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_spec = importlib.util.spec_from_file_location("ckv_bfp4_reference", Path(__file__).parents[1] / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

FILL = 0xAB


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 512, generator=g) * torch.pow(2.0, torch.randint(-12, 6, (n, 1), generator=g).float())
    x[:, :16] *= 50.0  # groups of other scales in the same row
    x[1, 32:48] = 0.0  # an all-zero group
    x[2] = 0.0  # an all-zero row (the masked rows of a chunk)
    return x.bfloat16()


@pytest.mark.parametrize("n, row0, table_rows", [(32, 0, 32), (64, 96, 256), (2560, 128, 2816), (5120, 0, 5120)])
def test_ckv_bfp4_rows(device, n, row0, table_rows):
    x = _inputs(n, n + row0)
    ckv = ttnn.from_torch(x.reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rows = ttnn.from_torch(
        torch.full((1, 1, table_rows, 320), FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=device
    )
    rounded = ttnn.bringup.ckv_bfp4(ckv, rows, row0, rounded=True)
    got = ttnn.to_torch(rows).view(torch.uint8).reshape(table_rows, 320)
    exp = ref.encode_rows(x)
    bad = (got[row0 : row0 + n] != exp).any(-1).nonzero().flatten()
    assert not len(bad), f"{len(bad)} rows differ, first {bad[:5].tolist()}"
    assert (got[:row0] == FILL).all() and (got[row0 + n :] == FILL).all(), "rows outside [row0, row0 + n) written"
    r = ttnn.to_torch(rounded).reshape(n, 512)
    assert torch.equal(r.float(), ref.decode_rows(exp)), "rounded bf16 != the records' values"

    other = ttnn.from_torch(
        _inputs(n, n + row0 + 1).reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    def run(inp):
        out = ttnn.bringup.ckv_bfp4(inp, rows, row0, rounded=True)
        return [rows, out]

    determinism.assert_deterministic(lambda: run(ckv), lambda: run(other), first=[rows, rounded], label=f"n{n}")


def test_ckv_bfp4_without_rounded_output(device):
    x = _inputs(64, 7)
    ckv = ttnn.from_torch(x.reshape(1, 1, 64, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rows = ttnn.from_torch(torch.zeros(1, 1, 64, 320, dtype=torch.uint8), dtype=ttnn.uint8, device=device)
    assert ttnn.bringup.ckv_bfp4(ckv, rows) is None
    assert torch.equal(ttnn.to_torch(rows).view(torch.uint8).reshape(64, 320), ref.encode_rows(x))


@pytest.mark.parametrize("n, rounded", [(64, True), (2560, True), (64, False)])
def test_ckv_bfp4_prepared(device, n, rounded):
    """prepared=True (P.7): one program per (mesh, n, rounded), row0 a value slot, every address patched. Over calls at
    other row0, other inputs and other tables (new addresses), the records and the rounded output equal the plain
    generic_op path byte for byte, rows outside each call's range untouched, and the program is prepared once."""
    mod = importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")

    table_rows = 4 * n + 64
    tables = {
        p: [
            ttnn.from_torch(
                torch.full((1, 1, table_rows, 320), FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=device
            )
            for _ in range(2)
        ]
        for p in (False, True)
    }
    before = len(mod._PREPARED)
    for i, (t, row0) in enumerate([(0, 0), (0, n), (1, 32), (0, 3 * n + 64), (1, 2 * n)]):
        x = _inputs(n, 100 * n + i)
        ckv = ttnn.from_torch(x.reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        outs = {p: ttnn.bringup.ckv_bfp4(ckv, tables[p][t], row0, rounded=rounded, prepared=p) for p in (False, True)}
        exp = ref.encode_rows(x)
        for p in (False, True):
            got = ttnn.to_torch(tables[p][t]).view(torch.uint8).reshape(table_rows, 320)
            assert torch.equal(got[row0 : row0 + n], exp), f"prepared={p} call {i}: records differ"
        a, b = (ttnn.to_torch(tables[p][t]).view(torch.uint8) for p in (False, True))
        assert torch.equal(a, b), f"call {i}: prepared table != generic_op table"
        if rounded:
            assert torch.equal(ttnn.to_torch(outs[False]), ttnn.to_torch(outs[True])), f"call {i}: rounded differ"
        else:
            assert outs[True] is None
    assert len(mod._PREPARED) == before + 1, "the program was not prepared exactly once"
    assert list(mod._PREPARED.values())[-1][1] is not None, "the program could not be prepared (patch count)"


@pytest.mark.parametrize("n, row0, table_rows", [(32, 0, 32), (64, 96, 256), (2176, 0, 2176)])
def test_ckv_bfp8_rows(device, n, row0, table_rows):
    """dtype=bfloat8_b: 576 B records ([32 exponents | the same 32 | 512 sign-magnitude bytes]), host rule, RNE."""
    x = _inputs(n, n + row0 + 11)
    ckv = ttnn.from_torch(x.reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rows = ttnn.from_torch(
        torch.full((1, 1, table_rows, 576), FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=device
    )
    rounded = ttnn.bringup.ckv_bfp4(ckv, rows, row0, rounded=True, dtype=ttnn.bfloat8_b)
    got = ttnn.to_torch(rows).view(torch.uint8).reshape(table_rows, 576)
    exp = ref.encode_rows(x, mbits=7)
    bad = (got[row0 : row0 + n] != exp).any(-1).nonzero().flatten()
    assert not len(bad), f"{len(bad)} rows differ, first {bad[:5].tolist()}"
    assert (got[:row0] == FILL).all() and (got[row0 + n :] == FILL).all(), "rows outside [row0, row0 + n) written"
    assert torch.equal(ttnn.to_torch(rounded).reshape(n, 512).float(), ref.decode_rows(exp))

    other = ttnn.from_torch(
        _inputs(n, n + 1).reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    def run(inp):
        ttnn.bringup.ckv_bfp4(inp, rows, row0, dtype=ttnn.bfloat8_b)
        return [rows]

    determinism.assert_deterministic(lambda: run(ckv), lambda: run(other), first=[rows], label=f"bfp8 n{n}")


@pytest.mark.parametrize("n, row0, blocks", [(32, 0, 1), (64, 96, 8), (2176, 0, 68)])
def test_ckv_bfp8_blocks(device, n, row0, blocks):
    """blocks=True: 32 rows a page ([exponents | the same, 32 B on | values]), host rule, RNE; other pages untouched."""
    x = _inputs(n, n + row0 + 13)
    ckv = ttnn.from_torch(x.reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    table = ttnn.from_torch(
        torch.full((1, 1, blocks, ref.BLOCK_BYTES), FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=device
    )
    ttnn.bringup.ckv_bfp4(ckv, table, row0, dtype=ttnn.bfloat8_b, blocks=True)
    got = ttnn.to_torch(table).view(torch.uint8).reshape(blocks, ref.BLOCK_BYTES)
    exp, mask = ref.encode_blocks(x)
    b0 = row0 // 32
    mine = got[b0 : b0 + n // 32]
    bad = ((mine != exp) & mask).any(-1).nonzero().flatten()
    assert not len(bad), f"{len(bad)} blocks differ, first {bad[:5].tolist()}"
    assert (got[:b0] == FILL).all() and (got[b0 + n // 32 :] == FILL).all(), "pages outside the rows written"

    other = ttnn.from_torch(
        _inputs(n, n + 1).reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    def run(inp):
        ttnn.bringup.ckv_bfp4(inp, table, row0, dtype=ttnn.bfloat8_b, blocks=True)
        return [table]

    determinism.assert_deterministic(lambda: run(ckv), lambda: run(other), first=[table], label=f"blocks n{n}")


@pytest.mark.parametrize("blocks", [False, True], ids=["bfp8_rows", "bfp8_blocks"])
def test_ckv_bfp8_prepared(device, blocks):
    """prepared=True with dtype=bfloat8_b (records or blocks): byte-equal to the generic_op path over calls at other
    row0 / tables, and a program of its own (dtype and blocks are in the prepared key)."""
    mod = importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")
    n, rows_total = 64, 256
    shape = [1, 1, rows_total // 32, ref.BLOCK_BYTES] if blocks else [1, 1, rows_total, 576]
    tables = {
        p: [
            ttnn.from_torch(torch.full(shape, FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=device)
            for _ in range(2)
        ]
        for p in (False, True)
    }
    before = len(mod._PREPARED)
    for i, (t, row0) in enumerate([(0, 0), (0, 96), (1, 32), (0, 192)]):
        x = _inputs(n, 7 * n + i)
        ckv = ttnn.from_torch(x.reshape(1, 1, n, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for p in (False, True):
            ttnn.bringup.ckv_bfp4(ckv, tables[p][t], row0, dtype=ttnn.bfloat8_b, blocks=blocks, prepared=p)
        a, b = (ttnn.to_torch(tables[p][t]).view(torch.uint8) for p in (False, True))
        assert torch.equal(a, b), f"call {i}: prepared table != generic_op table"
    assert len(mod._PREPARED) == before + 1, "the program was not prepared exactly once"
    assert list(mod._PREPARED.values())[-1][1] is not None, "the program could not be prepared (patch count)"

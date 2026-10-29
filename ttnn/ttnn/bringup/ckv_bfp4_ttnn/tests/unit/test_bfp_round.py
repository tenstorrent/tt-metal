# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.bfp_round on one device: bf16 TILE -> bfloat4_b / bfloat8_b TILE holding exactly the host rule's
round-to-nearest-even values (one exponent per 16 values of a tile row), as a new tensor or in place at a tile row of
a wider table (rows outside untouched), deterministic."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_spec = importlib.util.spec_from_file_location("ckv_bfp4_reference", Path(__file__).parents[1] / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

MBITS = {"bfloat4_b": 3, "bfloat8_b": 7}


def _inputs(shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g) * torch.pow(2.0, torch.randint(-12, 6, (*shape[:-1], 1), generator=g).float())
    x[..., 0, :16] *= 50.0
    x[..., 1, 32:48] = 0.0
    x[..., 2, :] = 0.0
    return x.bfloat16()


@pytest.mark.parametrize("dt", ["bfloat4_b", "bfloat8_b"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (1, 64, 512, 128), (1, 1, 1024, 512), (1, 3, 96, 96)])
def test_bfp_round_new(device, dt, shape):
    dtype = getattr(ttnn, dt)
    x = _inputs(shape, sum(shape))
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.bringup.bfp_round(t, dtype)
    assert out.dtype == dtype and list(out.shape) == list(shape)
    got = ttnn.to_torch(out).float()
    exp = ref.round_values(x, MBITS[dt])
    assert torch.equal(got, exp), f"{int((got != exp).sum())} values differ"
    other = ttnn.from_torch(_inputs(shape, sum(shape) + 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    determinism.assert_deterministic(
        lambda: ttnn.bringup.bfp_round(t, dtype),
        lambda: ttnn.bringup.bfp_round(other, dtype),
        first=out,
        label=f"{dt} {shape}",
    )


@pytest.mark.parametrize("dt", ["bfloat4_b", "bfloat8_b"])
@pytest.mark.parametrize("n, row0, rows", [(32, 0, 64), (320, 640, 2048), (1024, 1024, 2048)])
def test_bfp_round_into_table(device, dt, n, row0, rows):
    dtype = getattr(ttnn, dt)
    x = _inputs((1, 1, n, 128), n + row0)
    fill = torch.full((1, 1, rows, 128), 0.75)
    table = ttnn.from_torch(fill, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert ttnn.bringup.bfp_round(t, out=table, row0=row0) is table
    got = ttnn.to_torch(table).float().reshape(rows, 128)
    assert torch.equal(got[row0 : row0 + n], ref.round_values(x, MBITS[dt]).reshape(n, 128))
    assert (got[:row0] == 0.75).all() and (got[row0 + n :] == 0.75).all(), "rows outside [row0, row0 + n) written"


def test_bfp_round_refusals(device, expect_error):
    t = ttnn.from_torch(torch.zeros(1, 1, 32, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tab = ttnn.from_torch(torch.zeros(1, 1, 64, 128), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(AssertionError, "bfloat4_b or bfloat8_b"):
        ttnn.bringup.bfp_round(t, ttnn.bfloat16)
    with expect_error(AssertionError, "16"):
        ttnn.bringup.bfp_round(t, out=tab, row0=16)
    with expect_error(AssertionError, "64"):
        ttnn.bringup.bfp_round(t, out=tab, row0=64)


@pytest.mark.parametrize("dt", ["bfloat4_b", "bfloat8_b"])
@pytest.mark.parametrize("n", [512, 13824])
def test_bfp_round_prepared(device, dt, n):
    """prepared=True: one program per (mesh, tiles, dtype, memory configs), page0 a value slot, every address patched.
    New outputs and in-place writes at other row0 / tables equal the plain generic_op path byte for byte, rows outside
    each call's range untouched, and the program is prepared once (the same program writes both)."""
    mod = importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")
    dtype = getattr(ttnn, dt)
    rows = 3 * n
    tables = {
        p: [
            ttnn.from_torch(torch.full((1, 1, rows, 128), 0.75), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(2)
        ]
        for p in (False, True)
    }
    before = len(mod._PREPARED)
    for i, (k, row0) in enumerate([(0, 0), (0, n), (1, 2 * n), (0, 2 * n), (1, 0)]):
        x = _inputs((1, 1, n, 128), 7 * n + i)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        new = {p: ttnn.bringup.bfp_round(t, dtype, prepared=p) for p in (False, True)}
        assert torch.equal(ttnn.to_torch(new[False]), ttnn.to_torch(new[True])), f"call {i}: new outputs differ"
        assert torch.equal(ttnn.to_torch(new[True]).float(), ref.round_values(x, MBITS[dt])), f"call {i}: values"
        for p in (False, True):
            ttnn.bringup.bfp_round(t, out=tables[p][k], row0=row0, prepared=p)
        a, b = (ttnn.to_torch(tables[p][k]).float().reshape(rows, 128) for p in (False, True))
        assert torch.equal(a, b), f"call {i}: prepared table != generic_op table"
        assert torch.equal(b[row0 : row0 + n], ref.round_values(x, MBITS[dt]).reshape(n, 128)), f"call {i}: rows"
    assert len(mod._PREPARED) == before + 1, "the program was not prepared exactly once"
    assert list(mod._PREPARED.values())[-1][1] is not None, "the program could not be prepared (patch count)"


@pytest.mark.parametrize("prepared", [False, True], ids=["generic", "prepared"])
@pytest.mark.parametrize("dt", ["bfloat4_b", "bfloat8_b"])
@pytest.mark.parametrize("n", [96, 13824])
def test_bfp_round_compact(device, dt, n, prepared):
    """Option compact (merged core ranges, no per-core runtime args, the kernels split the tiles, page0 frozen in the
    program): new outputs and in-place writes (a reused output, as deepseek_v41_flash_d_p's LOWBIT_HOST) equal the
    default path byte for byte and the host rule, rows outside each call's range untouched."""
    mod = importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")
    dtype = getattr(ttnn, dt)
    rows = 3 * n
    tables = {
        c: ttnn.from_torch(torch.full((1, 1, rows, 128), 0.75), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        for c in (False, True)
    }
    for i, row0 in enumerate([0, n, 2 * n, 0]):
        x = _inputs((1, 1, n, 128), 5 * n + i)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        new = {c: ttnn.bringup.bfp_round(t, dtype, prepared=prepared, compact=c) for c in (False, True)}
        assert torch.equal(ttnn.to_torch(new[False]), ttnn.to_torch(new[True])), f"call {i}: new outputs differ"
        assert torch.equal(ttnn.to_torch(new[True]).float(), ref.round_values(x, MBITS[dt])), f"call {i}: values"
        for c in (False, True):
            ttnn.bringup.bfp_round(t, out=tables[c], row0=row0, prepared=prepared, compact=c)
        a, b = (ttnn.to_torch(tables[c]).float().reshape(rows, 128) for c in (False, True))
        assert torch.equal(a, b), f"call {i}: compact table != default table"
        assert torch.equal(b[row0 : row0 + n], ref.round_values(x, MBITS[dt]).reshape(n, 128)), f"call {i}: rows"
    if prepared:
        assert list(mod._PREPARED.values())[-1][1] is not None, "the program could not be prepared (patch count)"

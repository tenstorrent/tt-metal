# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ckv_bfp4_rows on the bring-up mesh (deepseek_v41_flash_d_p MB.4): rows [src_row0, src_row0 + n) of a
ROW_MAJOR bf16 row tensor (interleaved, or ND-sharded like a served cache slab) -> the host bfloat4_b rule's row records
at [row0, row0 + n), byte for byte (the same records as ckv_bfp4 of the tilized rows), rows outside untouched, every
chip; one prepared program for every (src_row0, n, row0); deterministic.

Runs on the mesh of the spec (the resident runner's held mesh, BATCH_SHARED):

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4_rows.py"
"""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.harness import mesh_parametrize

_spec = importlib.util.spec_from_file_location("ckv_bfp4_reference", Path(__file__).parents[1] / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

BATCH_SHARED = True  # (a mesh-only op test: the resident runner's held mesh)
FILL = 0xAB


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 512, generator=g) * torch.pow(2.0, torch.randint(-12, 6, (n, 1), generator=g).float())
    x[:, :16] *= 50.0
    x[1, 32:48] = 0.0
    x[2] = 0.0
    return x.bfloat16()


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return (min(g.x, 11), min(g.y, 10))  # (a Galaxy's 110-core cap, agent rule 18; smaller grids as they are)


def _nd(mesh, shard_rows=16):
    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    banks = get_num_dram_banks(mesh)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, shard_rows, 512],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def _src(mesh, x, mc):
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _table(mesh, rows):
    return ttnn.from_torch(
        torch.full((1, 1, rows, 320), FILL, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _chips(t, rows):
    devs = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(d).view(torch.uint8).reshape(rows, 320) for d in (devs[0], devs[-1])]


def _why(got, exp, x):
    """A failure's shape: unwritten, exponents vs values, transposed bands, zeros."""
    g, e = got[:32], exp[:32]
    out = [
        f"unwritten rows {int((g == FILL).all(-1).sum())}",
        f"exp bytes equal {bool(torch.equal(g[:, :64], e[:, :64]))}",
    ]
    out.append(f"value bytes equal {bool(torch.equal(g[:, 64:], e[:, 64:]))}")
    dg = ref.decode_rows(g)
    xt = x[:32].float()
    out.append(f"decoded vs x max {float((dg - xt).abs().max()):.3g}, |x| max {float(xt.abs().max()):.3g}")
    out.append(f"decoded zeros {int((dg == 0).sum())}/{dg.numel()}")
    out.append(
        f"got row0 {g[0, :8].tolist()} .. {g[0, 64:72].tolist()}, exp row0 {e[0, :8].tolist()} .. {e[0, 64:72].tolist()}"
    )
    return "; ".join(out)


# (src_row0, n, row0): one band, idle cores; a slab batch's rows at an offset; more bands than cores (several a core);
# a record row past the source row
CALLS = [(0, 32, 0), (2048 + 64, 640, 96), (4096, 8192, 0), (1024, 4096, 4096)]


@mesh_parametrize
@pytest.mark.parametrize("layout", ["interleaved", "nd_sharded"])
def test_ckv_bfp4_rows(mesh_device, layout):
    if mesh_device is None:
        pytest.skip("device test")
    import sys

    mod = sys.modules["ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4"]  # (the module: the package exports the function)

    mesh = mesh_device
    batches, R, table_rows = 4, 4096, 8192 + 1024
    x = _inputs(batches * R, 7 + (layout == "nd_sharded"))
    mc = _nd(mesh) if layout == "nd_sharded" else ttnn.DRAM_MEMORY_CONFIG
    src = _src(mesh, x.reshape(batches, 1, R, 512), mc)
    mg = _grid(mesh)
    n_before = len(mod._ROWS_PREPARED)
    for src_row0, n, row0 in CALLS:
        table = _table(mesh, table_rows)
        ttnn.bringup.ckv_bfp4_rows(src, table, src_row0, n, row0, max_grid=mg)
        exp = ref.encode_rows(x[src_row0 : src_row0 + n])
        for got in _chips(table, table_rows):
            bad = (got[row0 : row0 + n] != exp).any(-1).nonzero().flatten()
            assert not len(
                bad
            ), f"{layout} {(src_row0, n, row0)}: {len(bad)} rows differ, first {bad[:5].tolist()}; " + _why(
                got[row0 : row0 + n], exp, x[src_row0 : src_row0 + n]
            )
            assert (got[:row0] == FILL).all() and (got[row0 + n :] == FILL).all(), "rows outside written"
        ttnn.deallocate(table)
    assert len(mod._ROWS_PREPARED) == n_before + 1, "one prepared program for every (src_row0, n, row0)"

    # the same records as the tile path ckv_bfp4(to_layout(slice)) (the off path of deepseek_v41_flash_d_p's pack)
    src_row0, n, row0 = 2048 + 64, 640, 96
    flat = _src(mesh, x.reshape(1, 1, batches * R, 512), ttnn.DRAM_MEMORY_CONFIG)
    tl = ttnn.to_layout(ttnn.slice(flat, [0, 0, src_row0, 0], [1, 1, src_row0 + n, 512]), ttnn.TILE_LAYOUT)
    a, b = _table(mesh, table_rows), _table(mesh, table_rows)
    ttnn.bringup.ckv_bfp4(tl, a, row0, prepared=True, max_grid=mg)
    ttnn.bringup.ckv_bfp4_rows(src, b, src_row0, n, row0, max_grid=mg)
    for ga, gb in zip(_chips(a, table_rows), _chips(b, table_rows)):
        assert torch.equal(ga, gb), "ckv_bfp4_rows != ckv_bfp4 of the tilized rows"

    other = _src(mesh, _inputs(batches * R, 99).reshape(batches, 1, R, 512), mc)
    out = _table(mesh, table_rows)

    def run(inp):
        ttnn.bringup.ckv_bfp4_rows(inp, out, 4096, 8192, 0, max_grid=mg)
        return [out]

    determinism.assert_deterministic(lambda: run(src), lambda: run(other), first=None, label=f"rows {layout}")


@mesh_parametrize
def test_ckv_bfp4_rows_refusals(mesh_device, expect_error):
    if mesh_device is None:
        pytest.skip("device test")
    src = _src(mesh_device, torch.zeros(1, 1, 64, 512).bfloat16(), ttnn.DRAM_MEMORY_CONFIG)
    table = _table(mesh_device, 64)
    with expect_error(AssertionError, "n must be a multiple of 32"):
        ttnn.bringup.ckv_bfp4_rows(src, table, 0, 16, 0)
    with expect_error(AssertionError, "ckv_bfp4_rows: source rows"):
        ttnn.bringup.ckv_bfp4_rows(src, table, 32, 64, 0)
    with expect_error(AssertionError, "ckv_bfp4_rows: record rows"):
        ttnn.bringup.ckv_bfp4_rows(src, table, 0, 64, 32)

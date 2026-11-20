# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.unpack_fp8_rows (deepseek_v41_flash_d_p P.76, INDEX_K_SOURCE=served): Blaze decode's 192-byte
indexer key rows [e4m3 x 128 | fp32 scale | 60 zero bytes] -> e4m3 x scale on the bfloat8_b grid (nearest even, the
host packer's rule), bfloat8_b TILE, in place at a tile row.

Whole box (FABRIC_2D), per-chip random rows encoded on the host as the stock cast does (scale = amax / 448, values
e4m3 of key / scale; zero rows all zero), uploaded as the served caches hold them (ND-sharded, one 16-row record a
shard, round robin over the DRAM banks, several batches) and DRAM interleaved: the decoded tiles against the host
reference (e4m3 x scale in fp32, then rounded to nearest even onto bfloat8_b in float64; ttnn's host packer differs
just above ties, measured and printed), rows outside the range untouched, a range at
another output row and batch, determinism, the prepared program reused across ranges (one program), refusals."""

import pytest
import torch

import ttnn

ROW = 192


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), min(g.y, 10))  # (a Galaxy's 110-core cap, agent rule 18)


def _encode(k: torch.Tensor) -> torch.Tensor:
    """keys [.., 128] float -> 192-byte rows uint8 [.., 192] (scale = max(amax, 1e-4) / 448; e4m3 of k / scale; an
    all-zero row all zero)."""
    amax = k.abs().amax(-1, keepdim=True).float()
    scale = amax.clamp(min=1e-4) * (1.0 / 448.0)
    e = (k.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = torch.zeros(*k.shape[:-1], ROW, dtype=torch.uint8)
    out[..., :128] = e.view(torch.uint8)
    out[..., 128:132] = scale.float().contiguous().view(torch.uint8)
    zero = amax.squeeze(-1) == 0
    out[zero] = 0
    return out


def _decode(rows: torch.Tensor) -> torch.Tensor:
    """The values the rows hold, fp32 (e4m3 x scale, one fp32 multiply)."""
    v = rows[..., :128].contiguous().view(torch.float8_e4m3fn).float()
    sc = rows[..., 128:132].contiguous().view(torch.float32)
    return v * sc


def _keys(chips, b, r, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(chips, b, r, 128, generator=g)
    k[..., :, 5] *= 11.0  # a large channel
    k[..., 3, :] *= 1e-5  # a tiny row
    k[..., 7, :] = 0.0  # a zero (unwritten) row
    k[..., 9, :64] = 0.0
    return k


def _nd(mesh, shard_rows=16):
    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    banks = get_num_dram_banks(mesh)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(i, 0), ttnn.CoreCoord(i, 0)) for i in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, shard_rows, ROW],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def _rows_dev(mesh, rows, mc):
    """[chips, B, R, 192] uint8 -> device [B, 1, R, 192] uint8 ROW_MAJOR, chip i its own."""
    chips, b, r, _ = rows.shape
    return ttnn.from_torch(
        rows.reshape(chips * b, 1, r, ROW).contiguous(),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _out_dev(mesh, chips, m, fill):
    """[1, 1, m, 128] bfloat8_b TILE, chip i filled with ``fill[i]``."""
    return ttnn.from_torch(
        fill.reshape(chips, 1, m, 128).contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _host(t, chips, m):
    return torch.stack([ttnn.to_torch(x).float().reshape(m, 128) for x in ttnn.get_device_tensors(t)])[:chips]


def _bfp8(x: torch.Tensor) -> torch.Tensor:
    """x [n, 128] fp32 -> exactly rounded to nearest even onto the bfloat8_b grid (float64 arithmetic): per 16 values
    of a row (a tile face row), E = the max exponent, step 2^(E - 6), q = |x| / step rounded half even, at most 127.
    (ttnn's host packer, ttnn.from_torch(dtype=bfloat8_b), rounds an fp32 value just above a tie as a tie (it does not
    keep the bits below the tie): ~1e-4 of the rows here differ from it by one step, the device being the exact one.)"""
    g = x.double().reshape(-1, 16)
    _, e = torch.frexp(g.abs().amax(-1, keepdim=True))
    step = torch.ldexp(torch.ones_like(g[:, :1]), e - 7)  # 2^(E - 6), E = e - 1
    q = torch.round(g.abs() / step).clamp(max=127)  # torch.round: half to even
    out = torch.where(g.abs().amax(-1, keepdim=True) == 0, torch.zeros_like(g), torch.sign(g) * q * step)
    return out.float().reshape(x.shape)


def _host_packer(x: torch.Tensor) -> torch.Tensor:
    t = ttnn.from_torch(x.float().contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float()


@pytest.mark.parametrize("sharded", [True, False], ids=["nd_sharded", "interleaved"])
def test_unpack_fp8_rows(mesh, sharded):
    chips = mesh.get_num_devices()
    B, R, M = 3, 512, 640
    keys = _keys(chips, B, R, seed=7)
    rows = _encode(keys)
    exp_all = _bfp8(_decode(rows).reshape(-1, 128)).reshape(chips, B, R, 128)
    mc = _nd(mesh) if sharded else ttnn.DRAM_MEMORY_CONFIG
    dev = _rows_dev(mesh, rows, mc)
    sentinel = torch.full((chips, M, 128), 3.0)
    out = _out_dev(mesh, chips, M, sentinel)
    grid = _grid(mesh)

    # batch 1, rows [64, 448) -> out rows [128, 512)
    ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=64, row_hi=448, batch=1, out_row0=128, max_grid=grid)
    got = _host(out, chips, M)
    exp = sentinel.clone()
    exp[:, 128:512] = exp_all[:, 1, 64:448]
    diff = (got - exp).abs()
    print(
        f"unpack_fp8_rows ({'nd' if sharded else 'interleaved'}): max |device - host bfp8| {diff.max().item():.3g}, "
        f"values differing {(diff > 0).float().mean().item():.3g}"
    )
    assert torch.equal(got[:, :128], sentinel[:, :128]) and torch.equal(got[:, 512:], sentinel[:, 512:]), "outside"
    assert torch.equal(got, exp), "decoded rows differ from e4m3 x scale on the host bfloat8_b grid"

    # the same program at another range (value slots), in place; determinism
    ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=0, row_hi=128, batch=2, max_grid=grid)
    got2 = _host(out, chips, M)
    exp[:, 0:128] = exp_all[:, 2, 0:128]
    assert torch.equal(got2, exp)
    ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=0, row_hi=128, batch=2, max_grid=grid)
    assert torch.equal(_host(out, chips, M), got2), "not deterministic"
    # a zero row decodes to zero
    assert torch.all(got2[:, 7] == 0)


def test_unpack_fp8_rows_whole_grid(mesh):
    """whole_grid=True (deepseek_v41_flash_d_p MB.2): every range on the whole grid (idle cores past the band count),
    one prepared program for every row count; bytes equal to the default's (a core a band) at 1, 3 and 37 bands."""
    chips = mesh.get_num_devices()
    B, R, M = 2, 2048, 2048
    rows = _encode(_keys(chips, B, R, seed=13))
    dev = _rows_dev(mesh, rows, _nd(mesh))
    grid = _grid(mesh)
    for lo, n in ((0, 32), (96, 96), (512, 37 * 32)):
        a = _out_dev(mesh, chips, M, torch.full((chips, M, 128), 3.0))
        b = _out_dev(mesh, chips, M, torch.full((chips, M, 128), 3.0))
        ttnn.bringup.unpack_fp8_rows(dev, a, row_lo=lo, row_hi=lo + n, batch=1, max_grid=grid)
        ttnn.bringup.unpack_fp8_rows(dev, b, row_lo=lo, row_hi=lo + n, batch=1, max_grid=grid, whole_grid=True)
        assert torch.equal(_host(a, chips, M), _host(b, chips, M)), (lo, n)
    from ttnn.bringup.fp8_rows_ttnn import fp8_rows as F

    gx, gy = grid.x, grid.y
    shape = tuple(int(x) for x in dev.shape)
    mid = dev.device().id()  # (a process may hold programs of an earlier mesh: the precompile pass's)
    progs = [k for k in F._PREPARED if k[:4] == ("unpack", mid, shape, (1, 1, M, 128))]
    assert sum(1 for k in progs if k[6] == gx * gy) == 1, ("one whole-grid program", progs)


def test_unpack_fp8_rows_refusals(mesh, expect_error):
    chips = mesh.get_num_devices()
    rows = _rows_dev(mesh, torch.zeros(chips, 1, 64, ROW, dtype=torch.uint8), ttnn.DRAM_MEMORY_CONFIG)
    out = _out_dev(mesh, chips, 64, torch.zeros(chips, 64, 128))
    with expect_error(AssertionError, "whole tile rows only"):
        ttnn.bringup.unpack_fp8_rows(rows, out, row_lo=16, row_hi=48)
    with expect_error(AssertionError, "outside"):
        ttnn.bringup.unpack_fp8_rows(rows, out, row_lo=0, row_hi=96)
    with expect_error(AssertionError, "out must be bfloat8_b TILE"):
        ttnn.bringup.unpack_fp8_rows(rows, rows, row_lo=0, row_hi=32)


def test_unpack_fp8_rows_many_bands(mesh):
    """65536 rows a chip (the 1M context's ratio-2 index keys on 8x4): every core takes many bands (both CBs
    double-buffered); exact against the host; the host time of a warm call."""
    import time

    chips = mesh.get_num_devices()
    R = 65536
    keys = _keys(chips, 1, R, seed=11)
    rows = _encode(keys)
    exp = _bfp8(_decode(rows).reshape(-1, 128)).reshape(chips, R, 128)
    dev = _rows_dev(mesh, rows, _nd(mesh))
    out = _out_dev(mesh, chips, R, torch.zeros(chips, R, 128))
    grid = _grid(mesh)
    ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=0, row_hi=R, max_grid=grid)
    got = _host(out, chips, R)
    bad = (got != exp).any(-1)  # [chips, R]
    if bad.any():
        rows_bad = bad.nonzero()
        bands = torch.unique(rows_bad[:, 1] // 32)
        print(
            f"many bands: {bad.sum().item()} rows differ on chips {torch.unique(rows_bad[:, 0]).tolist()[:8]}, "
            f"{bands.numel()} bands, first {bands[:12].tolist()}, rows in band {torch.unique(rows_bad[:, 1] % 32).tolist()}"
        )
        for c, r0 in rows_bad[:6].tolist():
            cols = (got[c, r0] != exp[c, r0]).nonzero().flatten()
            sc = rows[c, 0, r0, 128:132].contiguous().view(torch.float32).item()
            print(
                f"chip {c} row {r0}: cols {cols.tolist()[:8]} got {got[c, r0, cols[:4]].tolist()} "
                f"exp {exp[c, r0, cols[:4]].tolist()} exact {_decode(rows[c, 0, r0])[cols[:4]].tolist()} scale {sc}"
            )
    assert torch.equal(got, exp)
    hp = _host_packer(_decode(rows).reshape(-1, 128)).reshape(chips, R, 128)
    print(f"vs ttnn's host packer: {(got != hp).any(-1).float().mean().item():.2e} of rows differ (ties, see _bfp8)")
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(10):
        ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=0, row_hi=R, max_grid=grid)
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    t2 = time.perf_counter()
    print(
        f"unpack_fp8_rows {R} rows a chip: host {(t1 - t0) / 10 * 1e3:.3f} ms a call, "
        f"synced {(t2 - t0) / 10 * 1e3:.3f} ms a call (10 calls)"
    )
    for n in (320, 640, 1280, 2048):
        ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=1024, row_hi=1024 + n, max_grid=grid)  # (built once)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.bringup.unpack_fp8_rows(dev, out, row_lo=1024, row_hi=1024 + n, max_grid=grid)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        t2 = time.perf_counter()
        print(
            f"unpack_fp8_rows {n} rows a chip: host {(t1 - t0) / 20 * 1e3:.3f} ms a call, synced "
            f"{(t2 - t0) / 20 * 1e3:.3f} ms a call (20 calls)"
        )
    assert torch.equal(_host(out, chips, R), exp)
    # reference point: the stock-free pack of the same rows' size (a prepared single-kernel program, common args only)
    e = ttnn.from_torch(
        torch.zeros(chips, 1, 640, 128),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    sc = ttnn.from_torch(
        torch.ones(chips, 1, 640, 1),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    po = ttnn.bringup.pack_fp8_rows(e, sc, max_grid=grid)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.bringup.pack_fp8_rows(e, sc, output_tensor=po, max_grid=grid)
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    print(
        f"pack_fp8_rows 640 rows a chip (reference): host {(t1 - t0) / 20 * 1e3:.3f} ms a call, synced "
        f"{(time.perf_counter() - t0) / 20 * 1e3:.3f}"
    )

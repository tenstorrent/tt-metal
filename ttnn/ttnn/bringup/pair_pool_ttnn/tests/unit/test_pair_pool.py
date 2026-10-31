# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.pair_pool (deepseek_v41_flash_d_p P.24): x [.., R, 2 hd] = [kv | gate] fp32 -> [.., R / 2, hd],
out[t] = kv[2t] w0 + kv[2t + 1] w1, (w0, w1) = softmax(gate[2t], gate[2t + 1]) per column.

Whole box (FABRIC_2D), random per-chip inputs: against the op chain it replaces (tt/compressor.py TtCompressor._pool:
reshape, slices, maximum, subtract, exp, add, divide, multiply, add on the device) to 1 fp32 ulp, against float64
within 1e-6 relative; negative controls (swapped parity, the gate ignored); the prepared program on fresh buffers
(a cache hit), a caller's output buffer, determinism; refusals."""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), g.y)  # (a Galaxy's 110-core cap, agent rule 18)


def _dev(t, mesh):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _inputs(nd, R, hd, seed, spread=3.0):
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(nd, 1, R, hd, generator=g)
    gate = torch.randn(nd, 1, R, hd, generator=g) * spread
    gate[..., 0, :8] = 90.0  # exp underflow of the other row's weight
    gate[..., 2, 8:16] = gate[..., 3, 8:16]  # equal gates
    return torch.cat([kv, gate], dim=-1)


def _want(x, hd):
    """float64 reference."""
    x = x.double()
    p = x.reshape(*x.shape[:-2], x.shape[-2] // 2, 4 * hd)
    kv0, g0, kv1, g1 = p[..., :hd], p[..., hd : 2 * hd], p[..., 2 * hd : 3 * hd], p[..., 3 * hd :]
    w = torch.softmax(torch.stack([g0, g1]), dim=0)
    return kv0 * w[0] + kv1 * w[1]


def _chain(xd, R, hd):
    """The op chain the model ran (tt/compressor.py TtCompressor._pool with TINY_KERNELS=off), on the device."""
    pr = ttnn.reshape(xd, [1, 1, R // 2, 4 * hd])
    col = lambda i: ttnn.slice(pr, [0, 0, 0, i * hd], [1, 1, R // 2, (i + 1) * hd])  # noqa: E731
    kv0, g0, kv1, g1 = col(0), col(1), col(2), col(3)
    m = ttnn.maximum(g0, g1)
    e = [ttnn.exp(ttnn.subtract(g, m, dtype=ttnn.float32)) for g in (g0, g1)]
    den = ttnn.add(e[0], e[1], dtype=ttnn.float32)
    t = [ttnn.multiply(kv, ttnn.divide(ei, den), dtype=ttnn.float32) for kv, ei in zip((kv0, kv1), e)]
    return ttnn.add(t[0], t[1], dtype=ttnn.float32)


def _ulps(a, b):
    ia = a.float().contiguous().view(torch.int32).long()
    ib = b.float().contiguous().view(torch.int32).long()
    ia = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return (ia - ib).abs()


CASES = [(640, 512), (1280, 512), (64, 32), (192, 96)]


@pytest.mark.parametrize("R, hd", CASES, ids=[f"R{c[0]}-hd{c[1]}" for c in CASES])
def test_pair_pool(mesh, R, hd):
    nd = mesh.get_num_devices()
    x = _inputs(nd, R, hd, seed=R + hd)
    xd = _dev(x, mesh)
    out = ttnn.bringup.pair_pool(xd, max_grid=_grid(mesh))
    assert list(out.shape) == [1, 1, R // 2, hd] and out.dtype == ttnn.float32
    got = _host(out, mesh)
    want = _want(x, hd)
    rel = ((got.double() - want).norm() / want.norm()).item()
    worst = ((got.double() - want).abs() / (want.abs() + 1e-30)).max().item()
    chain = _host(_chain(xd, R, hd), mesh)
    ul = _ulps(got, chain)
    print(
        f"R{R} hd{hd}: rel vs float64 {rel:.2e} (worst element {worst:.2e}); vs the op chain: max {ul.max()} ulp, "
        f"{int((ul > 0).sum())} of {ul.numel()} differ"
    )
    assert rel <= 1e-6, rel
    assert int(ul.max()) <= 1, f"{int(ul.max())} ulp from the op chain"
    # negative controls: the weights swapped between the pair's rows; the gate ignored
    xx = x.double().reshape(nd, 1, R // 2, 4 * hd)
    kv0, g0, kv1, g1 = (xx[..., i * hd : (i + 1) * hd] for i in range(4))
    w = torch.softmax(torch.stack([g0, g1]), dim=0)
    bad = kv0 * w[1] + kv1 * w[0]
    assert ((got.double() - bad).norm() / bad.norm()).item() > 1e-2, "swapped weights must fail"
    nog = (kv0 + kv1) / 2
    assert ((got.double() - nog).norm() / nog.norm()).item() > 1e-2, "ignoring the gate must fail"


def test_pair_pool_prepared_and_output(mesh):
    """Fresh buffers each call (the prepared program's patched addresses), a caller's output buffer, determinism."""
    nd = mesh.get_num_devices()
    R, hd = 640, 512
    outs = []
    for seed in (1, 2):
        x = _inputs(nd, R, hd, seed)
        xd = _dev(x, mesh)
        got = _host(ttnn.bringup.pair_pool(xd, max_grid=_grid(mesh)), mesh)
        assert ((got.double() - _want(x, hd)).norm() / _want(x, hd).norm()).item() <= 1e-6
        outs.append(got)
    buf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R // 2, hd]), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    x = _inputs(nd, R, hd, 2)
    xd = _dev(x, mesh)
    r = ttnn.bringup.pair_pool(xd, output_tensor=buf, max_grid=_grid(mesh))
    assert r is buf
    assert torch.equal(_host(buf, mesh), outs[1]), "the output buffer must hold the same bytes"
    # host time a warm call (prepared dispatch, the caller's buffer; printed, never asserted)
    import time

    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.bringup.pair_pool(xd, output_tensor=buf, max_grid=_grid(mesh))
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    print(f"pair_pool host a warm call ({nd} chips, {R}x{2 * hd}): {(t1 - t0) / 20 * 1e6:.1f} us")
    other = _dev(_inputs(nd, R, hd, 3), mesh)
    determinism.assert_deterministic(
        lambda: ttnn.bringup.pair_pool(xd, max_grid=_grid(mesh)),
        lambda: ttnn.bringup.pair_pool(other, max_grid=_grid(mesh)),
        first=r,
        label="pair_pool 640x1024",
    )


def test_pair_pool_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    with expect_error(AssertionError, "% 64 != 0"):
        ttnn.bringup.pair_pool(_dev(torch.zeros(nd, 1, 96, 64), mesh))
    with expect_error(AssertionError, "hd % 32 == 0"):
        ttnn.bringup.pair_pool(_dev(torch.zeros(nd, 1, 64, 96), mesh))
    xb = ttnn.from_torch(
        torch.zeros(1, 1, 64, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    with expect_error(AssertionError, "DataType.BFLOAT16"):
        ttnn.bringup.pair_pool(xb)

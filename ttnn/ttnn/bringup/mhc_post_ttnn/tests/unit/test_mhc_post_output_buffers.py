# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., output_tensor=, stats_output=, stats_scratch=) (deepseek_v41_flash_d_p P.16): the op
writes the caller's buffers for X', the stats row and the partials scratch instead of allocating them. On the whole
box (FABRIC_2D), random per-chip inputs, DeepSeek-V4.1's call (bf16 F and streams, packed mixes, stats_weight):
  * X' and stats are byte-identical on every chip to the same call without the options, and live in the given
    buffers; any subset of the three buffers works;
  * the buffers reused by a second call with other inputs hold the second result (a program-cache hit), their marker
    contents overwritten (negative control); the scratch reused across calls changes nothing;
  * the plain program (no stats_weight) with output_tensor;
  * refusals: a buffer of another spec, an input as the output, stats buffers without stats_weight."""

import pytest
import torch

import ttnn

N = 4
NG = N * (N + 2)


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh, dtype=ttnn.float32):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _bits(t):
    out = []
    for c in map(ttnn.to_torch, ttnn.get_device_tensors(t)):
        out.append(c.view(torch.int16) if c.dtype == torch.bfloat16 else c.view(torch.int32))
    return out


def _same(a, b):
    for x, y in zip(_bits(a), _bits(b)):
        assert torch.equal(x, y)


def scratch_shape(mesh, rows, cols):
    """The op's partials scratch [1, 1, R, 32 blocks] (mhc_post_stats_device_operation.cpp mhc_post_stats_plan)."""
    g = mesh.compute_with_storage_grid_size()
    cores = g.x * g.y
    ct, rt = cols // 32, rows // 32
    top = min(8, ct)
    best, bcb = None, 0
    for cb in range(1, top + 1):
        nb = -(-ct // cb)
        if nb > 64 and cb < top:
            continue
        cost = -(-(nb * rt) // cores) * (cb + 4)
        if best is None or cost <= best:
            best, bcb = cost, cb
    return [1, 1, rt * 32, -(-ct // bcb) * 32]


def _inputs(mesh, T, C, seed, dt):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    f = _dev(torch.randn(nd, 1, T, C, generator=g), mesh, dt)
    x = _dev(torch.randn(nd, 1, T, N * C, generator=g) * 3, mesh, dt)
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    w = torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5
    wd = ttnn.from_torch(
        w,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return f, x, _dev(mix, mesh), wd


CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)


@pytest.mark.parametrize(
    "T, C, dt", [(640, 1280, ttnn.bfloat16), (64, 256, ttnn.float32)], ids=["ds41_8x4_bf16", "small_fp32"]
)
def test_stats_buffers(mesh, expect_error, T, C, dt):
    nd = mesh.get_num_devices()
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    xo = _dev(torch.full((nd, 1, T, N * C), 1.0e4), mesh, dt)
    ro = _dev(torch.full((nd, 1, T, 32), 1.0e4), mesh)
    so = ttnn.allocate_tensor_on_device(
        ttnn.Shape(scratch_shape(mesh, T, C)), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    for seed in (1, 2):  # the second call reuses every buffer (a cache hit)
        f, x, md, wd = _inputs(mesh, T, C, seed + T, dt)
        kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N)
        x0, s0 = ttnn.bringup.mhc_post(f, x, md, md, **kw)
        x1, s1 = ttnn.bringup.mhc_post(f, x, md, md, output_tensor=xo, stats_output=ro, stats_scratch=so, **kw)
        assert x1.buffer_address() == xo.buffer_address() and s1.buffer_address() == ro.buffer_address()
        _same(xo, x0)
        _same(ro, s0)
    for sub in (dict(stats_output=ro), dict(stats_scratch=so), dict(stats_output=ro, stats_scratch=so)):
        x2, s2 = ttnn.bringup.mhc_post(f, x, md, md, **kw, **sub)
        _same(x2, x0)
        _same(s2, s0)
    with expect_error(RuntimeError, "stats_scratch spec"):
        ttnn.bringup.mhc_post(f, x, md, md, **kw, stats_scratch=ro)
    with expect_error(RuntimeError, "must not be an input"):
        ttnn.bringup.mhc_post(f, x, md, md, **kw, output_tensor=x)
    with expect_error(Exception, "need stats_weight"):
        ttnn.bringup.mhc_post(f, x, md, md, post_col=N, comb_col=2 * N, stats_output=ro)


def test_plain_output(mesh):
    nd, T, C = mesh.get_num_devices(), 640, 1280
    f, x, md, _ = _inputs(mesh, T, C, 5, ttnn.bfloat16)
    want = ttnn.bringup.mhc_post(f, x, md, md, post_col=N, comb_col=2 * N)
    out = _dev(torch.full((nd, 1, T, N * C), 1.0e4), mesh, ttnn.bfloat16)
    got = ttnn.bringup.mhc_post(f, x, md, md, post_col=N, comb_col=2 * N, output_tensor=out)
    assert got.buffer_address() == out.buffer_address()
    _same(out, want)

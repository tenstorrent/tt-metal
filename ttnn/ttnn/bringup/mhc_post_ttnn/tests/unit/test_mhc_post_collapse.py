# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., stats_weight=W, collapse_pre=P, hc_scale=, hc_base=, ...) (deepseek_v41_flash_d_p P.34,
setting POST_COLLAPSE): the stats pass also writes the next mHC step's inputs, y = sum_i P[:, i] X'_i and that step's
hc (DeepSeek's coefficients + Sinkhorn of the stats row), i.e. what ttnn.bringup.mhc_pre_xing(stats, X',
deepseek=True, collapse_pre=P, ...) computes from the op's own outputs.

On the whole box (FABRIC_2D), random per-chip inputs, DeepSeek-V4.1's call (bf16 F and streams, the packed mixes with
post_col / comb_col, the model's grid cap when the box is a Galaxy):
  * X' and stats bit-identical to the call without collapse_pre; hc and y bit-identical to mhc_pre_xing on them;
  * the pre given as the packed mixes themselves (the model's call: post = comb = collapse_pre) and as its own tensor;
  * a second call with other scalars / inputs into given hc / y buffers (a program-cache hit: the scalars and
    addresses are runtime args) matches its own reference; the buffers' marker contents are overwritten;
  * negative control: another pre gives another y;
  * refusals: fp32 streams, collapse_pre without stats_weight, hc_base of the wrong length."""

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


def _grid(mesh):
    """The model's cap on a Galaxy (agent rule 18: 11 x 10 = 110 cores); elsewhere the whole grid."""
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), min(g.y, 10)) if g.x * g.y > 110 else None


def _dev(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _bits(t):
    out = []
    for c in map(ttnn.to_torch, ttnn.get_device_tensors(t)):
        out.append(c.view(torch.int16) if c.dtype == torch.bfloat16 else c.view(torch.int32))
    return out


def _same(a, b):
    for x, y in zip(_bits(a), _bits(b)):
        assert torch.equal(x, y)


def _differ(a, b):
    return any(not torch.equal(x, y) for x, y in zip(_bits(a), _bits(b)))


def _inputs(mesh, T, C, seed):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    f = _dev(torch.randn(nd, 1, T, C, generator=g), mesh, ttnn.bfloat16)
    x = _dev(torch.randn(nd, 1, T, N * C, generator=g) * 3, mesh, ttnn.bfloat16)
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    pre = torch.rand(nd, 1, T, NG, generator=g) + 0.05
    w = torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5
    hc = dict(
        hc_scale=[float(v) for v in (torch.rand(3, generator=g) * 0.5 + 0.1)],
        hc_base=[float(v) for v in torch.randn(NG, generator=g) * 0.5],
        hc_norm_width=float(N * C),
        hc_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    return f, x, _dev(mix, mesh), _dev(pre, mesh), _dev(w, mesh, shard=False), hc


def _reference(f, x, md, pre, wd, hc, cfg, grid):
    """The path POST_COLLAPSE=off: mhc_post(stats_weight=) then mhc_pre_xing(stats, X', collapse_pre=pre)."""
    x1, s1 = ttnn.bringup.mhc_post(
        f, x, md, md, stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=grid
    )
    h1, y1 = ttnn.bringup.mhc_pre_xing(
        s1,
        x1,
        scale=hc["hc_scale"],
        base=hc["hc_base"],
        norm_width=hc["hc_norm_width"],
        n=N,
        norm_eps=hc["hc_norm_eps"],
        hc_eps=hc["hc_eps"],
        sinkhorn_iters=hc["hc_sinkhorn_iters"],
        deepseek=True,
        collapse_pre=pre,
        max_grid=grid,
    )
    return x1, s1, h1, y1


CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)


@pytest.mark.parametrize(
    "T, C", [(160, 5120), (640, 1280), (64, 640)], ids=["ds41_8x4_own_rows", "ds41_2x2", "small_ragged"]
)
@pytest.mark.parametrize("own_pre", [False, True], ids=["pre_is_mixes", "pre_tensor"])
def test_collapse_bit_identical(mesh, T, C, own_pre):
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    grid = _grid(mesh)
    f, x, md, pd, wd, hc = _inputs(mesh, T, C, T + C + int(own_pre))
    pre = pd if own_pre else md
    want = _reference(f, x, md, pre, wd, hc, cfg, grid)
    got = ttnn.bringup.mhc_post(
        f,
        x,
        md,
        md,
        stats_weight=wd,
        compute_kernel_config=cfg,
        post_col=N,
        comb_col=2 * N,
        max_grid=grid,
        collapse_pre=pre,
        **hc,
    )
    assert len(got) == 4
    assert list(got[2].shape) == list(want[2].shape) and got[2].dtype == ttnn.float32
    assert list(got[3].shape) == list(want[3].shape) and got[3].dtype == ttnn.bfloat16
    for g, w in zip(got, want):
        _same(g, w)


def test_collapse_buffers_and_cache_hit(mesh):
    nd, T, C = mesh.get_num_devices(), 160, 5120
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    grid = _grid(mesh)
    ho = _dev(torch.full((nd, 1, T, NG), 1.0e4), mesh)
    yo = _dev(torch.full((nd, 1, T, C), 1.0e4), mesh, ttnn.bfloat16)
    ys = []
    for seed in (11, 12):  # the second call: other scalars and inputs, the same program
        f, x, md, pd, wd, hc = _inputs(mesh, T, C, seed)
        want = _reference(f, x, md, md, wd, hc, cfg, grid)
        kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=grid)
        got = ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, hc_output=ho, y_output=yo, **kw, **hc)
        assert got[2].buffer_address() == ho.buffer_address() and got[3].buffer_address() == yo.buffer_address()
        _same(ho, want[2])
        _same(yo, want[3])
        _same(got[0], want[0])
        _same(got[1], want[1])
        # negative control: another pre collapses to another y
        other = ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=pd, **kw, **hc)
        assert _differ(other[3], want[3])
        _same(other[2], want[2])  # (the hc does not depend on the pre)
        ys.append(want[3])
    assert _differ(ys[0], ys[1])


def test_collapse_refusals(mesh, expect_error):
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    f, x, md, pd, wd, hc = _inputs(mesh, 64, 640, 3)
    kw = dict(compute_kernel_config=cfg, post_col=N, comb_col=2 * N)
    with expect_error(Exception, "collapse_pre needs stats_weight, hc_scale and hc_base"):
        ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, **kw, **hc)
    bad = dict(hc, hc_base=hc["hc_base"][:-1])
    with expect_error(Exception, "hc_base has 23 values, want 24"):
        ttnn.bringup.mhc_post(f, x, md, md, stats_weight=wd, collapse_pre=md, **kw, **bad)
    nd = mesh.get_num_devices()
    f32 = _dev(torch.randn(nd, 1, 64, 640), mesh)
    x32 = _dev(torch.randn(nd, 1, 64, N * 640), mesh)
    with expect_error(Exception, "collapse_pre needs a bfloat16 residual and a bfloat16 input_tensor"):
        ttnn.bringup.mhc_post(f32, x32, md, md, stats_weight=wd, collapse_pre=md, **kw, **hc)
    with expect_error(Exception, "need collapse_pre"):
        ttnn.bringup.mhc_post(f, x, md, md, stats_weight=wd, hc_scale=hc["hc_scale"], **kw)

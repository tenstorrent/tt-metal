# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., collapse_pre=, fast=True) (deepseek_v41_flash_d_p P.83, setting MHC_POST_FAST): the
pipelined program (columns streamed in a transaction id each, X' out a column at a time, a block's W read once and
shared through L1 with one unit a core, the coefficient / pre-block tiles built on the SFPU, the sum-of-squares tile
folded before its transpose, the feed's port reads pipelined) against the call without it (fast=False):
  * X', y (normalized or not) and feed_output bit-identical;
  * stats bit-identical except column n (n + 2) (sum x'^2, summed in another fp32 order: rel <= 1e-5) and hc within
    1e-5 (the rsqrt of that sum);
  * shapes: a ragged [64, 640] first (the smallest), the 8x4 own rows [160, 5120] (one unit a core: W shared), the 2x2
    own rows [1280, 5120] (several units a core: W per core), [640, 1280];
  * with / without norm, with / without the feed (links 2), the packed mixes (post_col / comb_col) and separate post /
    comb / pre tensors (comb transposed and not).
Runs on the 8x4 Galaxy's mesh (FABRIC_2D_TORUS_XY, the model's 11 x 10 cap); `check_fast` is shared with
models/demos/deepseek_v41_flash_d_p/tests/test_mhc_post_fast.py (the same cases on the model's resident mesh)."""

import pytest
import torch

import ttnn

N = 4
NG = N * (N + 2)
CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)

CASES = [
    # (T, C, norm, feed, packed, comb_transposed)
    (64, 640, True, False, True, True),
    (64, 640, True, True, True, True),
    (160, 5120, True, False, True, True),
    (160, 5120, True, True, True, True),
    (160, 5120, False, False, False, False),
    (640, 1280, True, True, True, True),
    (1280, 5120, True, False, True, True),
]
IDS = ["small_ragged", "small_ragged_feed", "ds41_8x4", "ds41_8x4_feed", "ds41_8x4_plain_sep", "640x1280_feed", "2x2"]

_SEMS = {}


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), min(g.y, 10)) if g.x * g.y > 110 else None


def _sems(mesh):
    if id(mesh) not in _SEMS:
        g = _grid(mesh) or mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        handles = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        ttnn.synchronize_device(mesh)
        _SEMS[id(mesh)] = (handles, [int(ttnn.get_global_semaphore_address(h)) for h in handles])
    return _SEMS[id(mesh)][1]


def _dev(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _chips(t):
    return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]


def _bits(c):
    return c.view(torch.int16) if c.dtype == torch.bfloat16 else c.view(torch.int32)


def _same(a, b, what):
    for i, (x, y) in enumerate(zip(_chips(a), _chips(b))):
        diff = int((_bits(x) != _bits(y)).sum())
        assert diff == 0, f"{what}: chip {i}: {diff} elements differ"


def check_fast(mesh, T, C, norm, feed, packed, comb_t, seed=0):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed + T + C)
    f = _dev(torch.randn(nd, 1, T, C, generator=g), mesh, ttnn.bfloat16)
    x = _dev(torch.randn(nd, 1, T, N * C, generator=g) * 3, mesh, ttnn.bfloat16)
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    w = _dev(torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5, mesh, shard=False)
    hc = dict(
        hc_scale=[float(v) for v in (torch.rand(3, generator=g) * 0.5 + 0.1)],
        hc_base=[float(v) for v in torch.randn(NG, generator=g) * 0.5],
        hc_norm_width=float(N * C),
        hc_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    kw = dict(stats_weight=w, compute_kernel_config=ttnn.ComputeConfigDescriptor(**CFG), max_grid=_grid(mesh))
    if packed:
        md = _dev(mix, mesh)
        post, comb, pre = md, md, md
        kw.update(post_col=N, comb_col=2 * N)
    else:
        post = _dev(mix[..., N : 2 * N], mesh)
        comb = _dev(mix[..., 2 * N :], mesh)
        pre = _dev(mix[..., :N], mesh)
        kw.update(comb_transposed=comb_t)
    if norm:
        kw.update(norm_weight=_dev(torch.randn(1, 1, 1, C, generator=g) * 0.2 + 1.0, mesh, shard=False), norm_eps=1e-6)
    outs = {}
    for fast in (False, True):
        fo = None
        k2 = dict(kw)
        if feed:
            G = tuple(mesh.shape)[1]
            fo = _dev(torch.full((nd, 1, G * T, C // G), 7.0e3 if fast else -7.0e3), mesh, ttnn.bfloat16)
            k2.update(feed_output=fo, feed_semaphores=_sems(mesh), feed_num_links=2, feed_cluster_axis=1)
        got = ttnn.bringup.mhc_post(f, x, post, comb, collapse_pre=pre, fast=fast, **k2, **hc)
        outs[fast] = (got, fo)
    (a, fa), (b, fb) = outs[False], outs[True]
    _same(a[0], b[0], "X'")
    if not feed:  # with the feed, y holds only the staged (other chips') blocks
        _same(a[3], b[3], "y")
    else:
        _same(fa, fb, "feed_output")
    ss = N * (N + 2)
    for i, (p, q) in enumerate(zip(_chips(a[1]), _chips(b[1]))):
        p, q = p.reshape(-1, 32), q.reshape(-1, 32)
        keep = [c for c in range(32) if c != ss]
        assert torch.equal(_bits(p[:, keep]), _bits(q[:, keep])), f"stats chip {i}: columns other than {ss} differ"
        rel = float(((p[:, ss] - q[:, ss]).abs() / p[:, ss].abs().clamp_min(1e-30)).max())
        assert rel <= 1e-5, f"stats chip {i}: sum x'^2 rel {rel}"
    for i, (p, q) in enumerate(zip(_chips(a[2]), _chips(b[2]))):
        d = float((p - q).abs().max())
        assert d <= 1e-5, f"hc chip {i}: max abs {d}"


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    shape = ttnn.MeshShape(8, 4) if ttnn.get_num_devices() == 32 else None
    m = ttnn.open_mesh_device(mesh_shape=shape, l1_small_size=24576) if shape else ttnn.open_mesh_device()
    yield m
    _SEMS.pop(id(m), None)
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("T, C, norm, feed, packed, comb_t", CASES, ids=IDS)
def test_fast(mesh, T, C, norm, feed, packed, comb_t):
    check_fast(mesh, T, C, norm, feed, packed, comb_t)


def test_fast_refusal(mesh, expect_error):
    nd = mesh.get_num_devices()
    f = _dev(torch.randn(nd, 1, 64, 640), mesh, ttnn.bfloat16)
    x = _dev(torch.randn(nd, 1, 64, 4 * 640), mesh, ttnn.bfloat16)
    md = _dev(torch.rand(nd, 1, 64, NG), mesh)
    with expect_error(Exception, "fast need collapse_pre"):
        ttnn.bringup.mhc_post(f, x, md, md, post_col=N, comb_col=2 * N, fast=True)

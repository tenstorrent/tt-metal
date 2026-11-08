# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., norm_weight=w, feed_output=X, feed_semaphores=[ready, arrival]) (deepseek_v41_flash_d_p
P.36, setting FFN_FEED): the normalized y also goes where an all-to-all over the TP axis would put it (the MoE head's
input, the column split): chip p of a group holds column block p of every group chip's y, chip q's rows at
[q T, (q + 1) T). The own block is written locally, the others travel over the fabric from the op's port cores.

On the 8x4 Galaxy (FABRIC_2D and the model's FABRIC_2D_TORUS_XY, the model's 11 x 10 grid cap, mesh opened 8 x 4),
random per-chip inputs, DeepSeek-V4.1's call (bf16 F and streams, the packed mixes):
  * feed_output bit-identical to the column blocks of the op's own normalized y (the call without feed_output) on every
    chip; X', hc and stats bit-identical to that call;
  * shapes: the 8x4 own rows [160, 5120] (one unit a core), the replay128 tail [32, 5120], the 2x2 own rows
    [1280, 5120] (several units a core: the units split over the cores before the ports), [640, 1280] (column blocks
    straddle the chips' blocks), a ragged [64, 640];
  * links 1 and 2 per direction;
  * repeated calls into the same buffer with the same semaphores (the ready fence and the re-armed counts), with other
    inputs each time, and a second feed buffer (a program-cache hit with another address);
  * the same result as the stock all_to_all_async_generic of the y (the path FFN_FEED=off runs);
  * refusals: feed_output without norm_weight, without semaphores, of the wrong shape."""

import pytest
import torch

import ttnn

N = 4
NG = N * (N + 2)
AXIS = 1
CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)


@pytest.fixture(
    scope="module",
    params=[ttnn.FabricConfig.FABRIC_2D, ttnn.FabricConfig.FABRIC_2D_TORUS_XY],
    ids=["fabric_2d", "fabric_2d_torus_xy"],
)
def mesh(request):
    """FABRIC_2D (a line along each axis) and DeepSeek-V4.1's FABRIC_2D_TORUS_XY (rings: the far chip of a 4-chip
    group is reached round the wrap, through the other side's port)."""
    ttnn.set_fabric_config(request.param)
    shape = ttnn.MeshShape(8, 4) if ttnn.get_num_devices() == 32 else None
    m = ttnn.open_mesh_device(mesh_shape=shape, l1_small_size=24576) if shape else ttnn.open_mesh_device()
    yield m
    _SEMS.pop(id(m), None)
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _grid(mesh):
    """The model's cap on a Galaxy (agent rule 18: 11 x 10 = 110 cores); elsewhere the whole grid."""
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), min(g.y, 10)) if g.x * g.y > 110 else None


_SEMS = {}


def _sems(mesh):
    """[ready, arrival]: two global semaphores over the (capped) grid, at 0 on every chip before any call."""
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


def _same(a, b):
    for x, y in zip(_chips(a), _chips(b)):
        assert torch.equal(_bits(x), _bits(y))


def _inputs(mesh, T, C, seed):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    f = _dev(torch.randn(nd, 1, T, C, generator=g), mesh, ttnn.bfloat16)
    x = _dev(torch.randn(nd, 1, T, N * C, generator=g) * 3, mesh, ttnn.bfloat16)
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    w = torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5
    hc = dict(
        hc_scale=[float(v) for v in (torch.rand(3, generator=g) * 0.5 + 0.1)],
        hc_base=[float(v) for v in torch.randn(NG, generator=g) * 0.5],
        hc_norm_width=float(N * C),
        hc_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    gamma = _dev(torch.randn(1, 1, 1, C, generator=g) * 0.2 + 1.0, mesh, shard=False)
    return f, x, _dev(mix, mesh), _dev(w, mesh, shard=False), hc, gamma


def _feed_buffer(mesh, T, C, fill):
    rows, cols = tuple(mesh.shape)
    G = (rows, cols)[AXIS]
    nd = mesh.get_num_devices()
    return _dev(torch.full((nd, 1, G * T, C // G), fill), mesh, ttnn.bfloat16)


def _expected(mesh, y):
    """Per chip (row-major mesh order): the all-to-all over AXIS of the chips' y [1, 1, T, C] (in_dim 2, out_dim 3)."""
    rows, cols = tuple(mesh.shape)
    ys = [c.reshape(c.shape[-2], c.shape[-1]) for c in _chips(y)]
    T, C = ys[0].shape
    G = (rows, cols)[AXIS]
    cb = C // G
    out = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        p = (r, c)[AXIS]
        group = [r * cols + q for q in range(cols)] if AXIS == 1 else [q * cols + c for q in range(rows)]
        out.append(torch.cat([ys[j][:, p * cb : (p + 1) * cb] for j in group], dim=0))
    return out


def _check_feed(mesh, fo, y):
    for got, want in zip(_chips(fo), _expected(mesh, y)):
        got = got.reshape(want.shape)
        assert torch.equal(_bits(got), _bits(want)), int((_bits(got) != _bits(want)).sum())


def _call(mesh, f, x, md, wd, hc, gamma, *, feed=None, links=1, y_output=None):
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=_grid(mesh))
    if feed is not None:
        kw.update(feed_output=feed, feed_semaphores=_sems(mesh), feed_num_links=links, feed_cluster_axis=AXIS)
    return ttnn.bringup.mhc_post(
        f, x, md, md, collapse_pre=md, norm_weight=gamma, norm_eps=1e-6, y_output=y_output, **kw, **hc
    )


@pytest.mark.parametrize("links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize(
    "T, C",
    [(160, 5120), (32, 5120), (1280, 5120), (640, 1280), (64, 640)],
    ids=["ds41_8x4_own_rows", "ds41_8x4_tail1024", "ds41_2x2_own_rows", "straddling_blocks", "small_ragged"],
)
def test_feed(mesh, T, C, links):
    f, x, md, wd, hc, gamma = _inputs(mesh, T, C, T + C + links)
    want = _call(mesh, f, x, md, wd, hc, gamma)
    fo = _feed_buffer(mesh, T, C, 3.0e4)
    got = _call(mesh, f, x, md, wd, hc, gamma, feed=fo, links=links)
    assert len(got) == 4
    _same(got[0], want[0])  # X'
    _same(got[1], want[1])  # stats
    _same(got[2], want[2])  # hc
    _check_feed(mesh, fo, want[3])


def test_feed_repeated_and_cache_hit(mesh):
    """Five calls, other inputs each, two feed buffers alternating (same program, another address), the same
    semaphores: each feed matches its own call's y; and the stock all_to_all of y gives the same bytes."""
    T, C = 160, 5120
    bufs = [_feed_buffer(mesh, T, C, 1.0e4), _feed_buffer(mesh, T, C, -1.0e4)]
    for k in range(5):
        f, x, md, wd, hc, gamma = _inputs(mesh, T, C, 100 + k)
        want = _call(mesh, f, x, md, wd, hc, gamma)
        fo = bufs[k % 2]
        got = _call(mesh, f, x, md, wd, hc, gamma, feed=fo, links=2)
        _same(got[2], want[2])
        _check_feed(mesh, fo, want[3])
        if k == 0:
            a2a = ttnn.experimental.all_to_all_async_generic(
                want[3], in_dim=2, out_dim=3, num_links=2, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=AXIS
            )
            _same(a2a, fo)


def test_feed_refusals(mesh, expect_error):
    T, C = 64, 640
    f, x, md, wd, hc, gamma = _inputs(mesh, T, C, 7)
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=_grid(mesh))
    fo = _feed_buffer(mesh, T, C, 0.0)
    with expect_error(Exception, "mhc_post: feed_output needs norm_weight"):
        ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, feed_output=fo, feed_semaphores=_sems(mesh), **kw, **hc)
    with expect_error(Exception, "mhc_post: feed_output needs feed_semaphores"):
        ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, norm_weight=gamma, feed_output=fo, **kw, **hc)
    bad = _feed_buffer(mesh, T, 2 * C, 0.0)
    with expect_error(Exception, "mhc_post: feed_output must be"):
        ttnn.bringup.mhc_post(
            f, x, md, md, collapse_pre=md, norm_weight=gamma, feed_output=bad, feed_semaphores=_sems(mesh), **kw, **hc
        )

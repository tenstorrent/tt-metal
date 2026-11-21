# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., collapse_pre=, fast=True, v2=True) (deepseek_v41_flash_d_p PB.8, setting MHC_POST_V2)
against fast=True alone (the P.83 program):
  * every output bit-identical: X', stats (all columns), hc, y (without the feed) and feed_output (with it): v2 moves
    the same values (the feed's tiles land in the ports' L1 instead of DRAM, a W share goes out by multicast, the hc
    row's transposes run on the SFPU), every sum keeps its order;
  * repeat determinism: three v2 calls on the same inputs give the same bytes (and the feed's semaphores re-arm);
  * shapes, smallest first: ragged [64, 640] (segments of 2), the 8x4 own rows [160, 5120] (W segments + multicast,
    landing), [256, 5120] (the packed 8192 path: two units a core, no segments; landing), [640, 1280] feed, the 2x2's
    [1280, 5120] (several units a core; the landing would not fit: the staging path).
`check_v2` is shared with models/demos/deepseek_v41_flash_d_p/tests/test_mhc_post_v2.py (the model's resident mesh)."""

import pytest
import torch

import ttnn

from .test_mhc_post_fast import CFG, N, NG, _bits, _chips, _dev, _grid, _sems

CASES = [
    # (T, C, norm, feed, packed, comb_transposed)
    (64, 640, True, False, True, True),
    (64, 640, True, True, True, True),
    (160, 5120, True, False, True, True),
    (160, 5120, True, True, True, True),
    (256, 5120, True, True, True, True),
    (256, 5120, True, False, True, True),
    (640, 1280, True, True, True, True),
    (1280, 5120, True, False, True, True),
]
IDS = [
    "small_ragged",
    "small_ragged_feed",
    "ds41_8x4",
    "ds41_8x4_feed",
    "packed8k_feed",
    "packed8k",
    "640x1280_feed",
    "2x2",
]


def _same(a, b, what):
    for i, (x, y) in enumerate(zip(_chips(a), _chips(b))):
        diff = int((_bits(x) != _bits(y)).sum())
        assert diff == 0, f"{what}: chip {i}: {diff} elements differ"


def check_v2(mesh, T, C, norm, feed, packed, comb_t, seed=0, repeats=3):
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

    def call(v2, fill):
        fo = None
        k2 = dict(kw)
        if feed:
            G = tuple(mesh.shape)[1]
            fo = _dev(torch.full((nd, 1, G * T, C // G), fill), mesh, ttnn.bfloat16)
            k2.update(feed_output=fo, feed_semaphores=_sems(mesh), feed_num_links=2, feed_cluster_axis=1)
        got = ttnn.bringup.mhc_post(f, x, post, comb, collapse_pre=pre, fast=True, v2=v2, **k2, **hc)
        return [got[0], got[1], got[2]] + ([fo] if feed else [got[3]])

    names = ["X'", "stats", "hc", "feed_output" if feed else "y"]
    ref = call(False, -7.0e3)
    outs = [call(True, 7.0e3 + i) for i in range(repeats)]
    for i, got in enumerate(outs):
        for name, a, b in zip(names, ref, got):
            _same(a, b, f"{name} (v2 call {i} vs fast)")


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    shape = ttnn.MeshShape(8, 4) if ttnn.get_num_devices() == 32 else None
    m = ttnn.open_mesh_device(mesh_shape=shape, l1_small_size=24576) if shape else ttnn.open_mesh_device()
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("T, C, norm, feed, packed, comb_t", CASES, ids=IDS)
def test_v2(mesh, T, C, norm, feed, packed, comb_t):
    check_v2(mesh, T, C, norm, feed, packed, comb_t)


def test_v2_refusal(mesh, expect_error):
    nd = mesh.get_num_devices()
    f = _dev(torch.randn(nd, 1, 64, 640), mesh, ttnn.bfloat16)
    x = _dev(torch.randn(nd, 1, 64, 4 * 640), mesh, ttnn.bfloat16)
    md = _dev(torch.rand(nd, 1, 64, NG), mesh)
    with expect_error(Exception, "fast / v2 need collapse_pre"):
        ttnn.bringup.mhc_post(f, x, md, md, post_col=N, comb_col=2 * N, v2=True)

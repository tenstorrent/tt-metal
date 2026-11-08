# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post(..., collapse_pre=P, norm_weight=w, norm_eps=eps) (deepseek_v41_flash_d_p P.35, setting
NORM_IN_MHC): the collapse's y comes out RMS-normalized, y_n = bf16((y * rsqrt(sum_c y^2 / C + eps)) * w) with y the
bf16 collapse the op writes without norm_weight.

On the whole box (FABRIC_2D), random per-chip inputs, DeepSeek-V4.1's call (bf16 F and streams, the packed mixes with
post_col / comb_col, the model's grid cap when the box is a Galaxy):
  * X' and hc bit-identical to the call without norm_weight; stats too except column n (n + 2) + 1, which holds the
    row's sum of y^2 (fp32, vs float64 of the bf16 y: within fp32 summation error);
  * y_n against float64 of the same formula on the op's own bf16 y (from the call without norm_weight): at most one
    bf16 step anywhere, and on almost every element none (the statistic is an exact fp32 sum, the rsqrt fp32-grade);
  * shapes: the 8x4 own rows [160, 5120] (one unit a core), the 2x2 own rows [1280, 5120] (several units a core,
    several rooted rows), [640, 1280], a ragged last block [64, 640];
  * a second call (program-cache hit) with another weight / eps into a given y buffer matches its own reference;
  * negative control: another weight gives another y_n;
  * refusals: norm_weight without collapse_pre, a weight of the wrong width or dtype."""

import pytest
import torch

import ttnn

N = 4
NG = N * (N + 2)
SY = NG + 1  # stats column of sum y^2


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
    gamma = torch.randn(1, 1, 1, C, generator=g) * 0.2 + 1.0
    return f, x, _dev(mix, mesh), _dev(w, mesh, shard=False), hc, gamma


def _gamma(gamma, mesh):
    return _dev(gamma, mesh, shard=False)


def _norm_ref(y_bf16: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """float64 of the op's formula on its own bf16 y (the statistic summed exactly)."""
    y = y_bf16.double()
    rs = torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y * rs) * gamma.double().reshape(1, 1, -1)


def _check_norm(got, y_plain, gamma, eps, T):
    """At most one bf16 step anywhere; < 0.1 % of the elements one step off the float64 value's rounding."""
    off = total = 0
    for yn, y in zip(_chips(got), _chips(y_plain)):
        y = y.reshape(1, T, -1)
        want = _norm_ref(y, gamma, eps)
        g64 = yn.reshape(want.shape).double()
        step = want.to(torch.bfloat16).double()
        _, e = torch.frexp(step)
        ulp = torch.ldexp(torch.ones_like(step), e - 8).clamp_min(2.0**-133)  # one bf16 step at the value
        err = (g64 - want).abs() / ulp
        assert float(err.max()) <= 1.0, float(err.max())
        off += int((g64 != step).sum())
        total += step.numel()
    assert off <= 1e-3 * total, (off, total)
    return off / total


CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)


@pytest.mark.parametrize(
    "T, C",
    [(160, 5120), (1280, 5120), (640, 1280), (64, 640)],
    ids=["ds41_8x4_own_rows", "ds41_2x2_own_rows", "ds41_2x2_split", "small_ragged"],
)
def test_norm(mesh, T, C):
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    grid = _grid(mesh)
    f, x, md, wd, hc, gamma = _inputs(mesh, T, C, T + C)
    kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=grid)
    want = ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, **kw, **hc)
    eps = 1e-6
    got = ttnn.bringup.mhc_post(
        f, x, md, md, collapse_pre=md, norm_weight=_gamma(gamma, mesh), norm_eps=eps, **kw, **hc
    )
    assert len(got) == 4
    assert list(got[3].shape) == list(want[3].shape) and got[3].dtype == ttnn.bfloat16
    _same(got[0], want[0])  # X'
    _same(got[2], want[2])  # hc
    for gs, ws, y in zip(_chips(got[1]), _chips(want[1]), _chips(want[3])):
        keep = torch.ones(32, dtype=torch.bool)
        keep[SY] = False
        assert torch.equal(_bits(gs[..., keep]), _bits(ws[..., keep]))
        ss = y.double().square().sum(-1)
        rel = ((gs[..., SY].double() - ss).abs() / ss.clamp_min(1e-30)).max()
        assert float(rel) < 1e-5, float(rel)
    frac = _check_norm(got[3], want[3], gamma, eps, T)
    print(f"T={T} C={C}: {frac:.2e} of y_n one bf16 step off the float64 rounding")


def test_norm_buffers_and_cache_hit(mesh):
    nd, T, C = mesh.get_num_devices(), 160, 5120
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    grid = _grid(mesh)
    yo = _dev(torch.full((nd, 1, T, C), 1.0e4), mesh, ttnn.bfloat16)
    outs = []
    for seed, eps in ((21, 1e-6), (22, 1e-2)):  # the second call: other inputs, weight, eps; the same program
        f, x, md, wd, hc, gamma = _inputs(mesh, T, C, seed)
        kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=grid)
        plain = ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, **kw, **hc)
        got = ttnn.bringup.mhc_post(
            f, x, md, md, collapse_pre=md, y_output=yo, norm_weight=_gamma(gamma, mesh), norm_eps=eps, **kw, **hc
        )
        assert got[3].buffer_address() == yo.buffer_address()
        _check_norm(yo, plain[3], gamma, eps, T)
        # negative control: another weight gives another y_n
        other = ttnn.bringup.mhc_post(
            f, x, md, md, collapse_pre=md, norm_weight=_gamma(gamma * 1.5, mesh), norm_eps=eps, **kw, **hc
        )
        assert any(not torch.equal(_bits(a), _bits(b)) for a, b in zip(_chips(other[3]), _chips(yo)))
        outs.append(_chips(yo)[0].clone())
    assert not torch.equal(_bits(outs[0]), _bits(outs[1]))


def test_norm_refusals(mesh, expect_error):
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    f, x, md, wd, hc, gamma = _inputs(mesh, 64, 640, 3)
    kw = dict(compute_kernel_config=cfg, post_col=N, comb_col=2 * N)
    gd = _gamma(gamma, mesh)
    with expect_error(Exception, "hc_scale / hc_base / hc_output / y_output / norm_weight need collapse_pre"):
        ttnn.bringup.mhc_post(f, x, md, md, stats_weight=wd, norm_weight=gd, **kw)
    with expect_error(Exception, "float32 TILE, DRAM interleaved, got"):
        ttnn.bringup.mhc_post(
            f, x, md, md, stats_weight=wd, collapse_pre=md, norm_weight=_gamma(gamma[..., :320], mesh), **kw, **hc
        )
    bf = ttnn.typecast(gd, ttnn.bfloat16)
    with expect_error(Exception, "float32 TILE, DRAM interleaved, got"):
        ttnn.bringup.mhc_post(f, x, md, md, stats_weight=wd, collapse_pre=md, norm_weight=bf, **kw, **hc)

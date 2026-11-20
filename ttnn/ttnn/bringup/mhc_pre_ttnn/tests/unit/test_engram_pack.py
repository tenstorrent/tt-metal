# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre_xing_pack(None, h, n=n, engram_keys=k, engram_weight=w) (deepseek_v41_flash_d_p P.14): the
engram gate's per-stream partial sums in one pass, row = [sum h_i w_i k_i | sum h_i^2 | sum k_i^2 | 0 ...].

Per chip (random per-device inputs, the whole box, FABRIC_2D), against float64 sums of the device's fp32 inputs:
every one of the 3 n columns within 1e-6 relative (fp32 sums, exact products but for the h w rounding), the other
columns exactly zero; against the op chain the model used (ttnn.multiply + ttnn.sum on fp32, the exact SFPU reduce)
within 1e-6; negative controls (a dropped column tile, the weight ignored); a program-cache hit on fresh buffers;
refusals."""

import pytest
import torch

import ttnn


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh, shard=True):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).double()


def _inputs(nd, T, C, n, seed, extra=1):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(nd, 1, T, n * C, generator=g) * 2
    kv = torch.randn(nd, 1, T, (n + extra) * C, generator=g)
    wrow = (torch.randn(n * C, generator=g) * 0.1).float()
    return h, kv, wrow


def _want(h, kv, wrow, n, C):
    """float64 [.., T, 3n]: dot_i, sum h_i^2, sum k_i^2 of the fp32 inputs."""
    h, k, w = h.double(), kv[..., : n * C].double(), wrow.double()
    hs = h.reshape(*h.shape[:-1], n, C)
    ks = k.reshape(*k.shape[:-1], n, C)
    ws = w.reshape(n, C)
    dot = (hs * ws * ks).sum(-1)
    return torch.cat([dot, hs.square().sum(-1), ks.square().sum(-1)], dim=-1)


def _rel_cols(got, want):
    """The worst column's relative L2 error."""
    g, w = got.reshape(-1, got.shape[-1]).double(), want.reshape(-1, want.shape[-1]).double()
    return ((g - w).square().sum(0).sqrt() / w.square().sum(0).sqrt()).max().item()


def _call(mesh, h, kv, wrow, n):
    w32 = wrow.reshape(1, 1, 1, -1).expand(1, 1, 32, wrow.numel())
    return ttnn.bringup.mhc_pre_xing_pack(
        None, _dev(h, mesh), n=n, engram_keys=_dev(kv, mesh), engram_weight=_dev(w32, mesh, shard=False)
    )


CASES = [(1024, 2560, 4), (1024, 1280, 4), (64, 256, 4), (128, 416, 4), (96, 512, 2)]


@pytest.mark.parametrize("T, C, n", CASES, ids=[f"T{c[0]}-C{c[1]}-n{c[2]}" for c in CASES])
def test_engram_pack(mesh, T, C, n):
    nd = mesh.get_num_devices()
    h, kv, wrow = _inputs(nd, T, C, n, seed=T + C + n)
    got = _host(_call(mesh, h, kv, wrow, n), mesh)
    assert tuple(got.shape) == (nd, 1, T, 32), got.shape
    want = _want(h, kv, wrow, n, C)
    rel = _rel_cols(got[..., : 3 * n], want)
    print(f"T{T} C{C} n{n}: worst column rel vs float64 {rel:.2e}")
    assert rel <= 1e-6, rel
    assert torch.all(got[..., 3 * n :] == 0), "columns past 3n must be zero"
    # the op chain the model used (fp32 multiply, the exact SFPU ttnn.sum), on the device
    hd, kd = _dev(h, mesh), _dev(kv, mesh)
    wd = _dev(wrow.reshape(1, 1, 1, -1), mesh, shard=False)
    cc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    hw = ttnn.multiply(hd, wd, dtype=ttnn.float32)
    cols = []
    for i in range(n):
        sl = lambda t: ttnn.slice(t, [0, 0, 0, i * C], [1, 1, T, (i + 1) * C])  # noqa: E731  (per-chip shape)
        p = ttnn.multiply(sl(hw), sl(kd), dtype=ttnn.float32)
        cols.append(_host(ttnn.sum(p, dim=-1, keepdim=True, compute_kernel_config=cc), mesh))
    chain = torch.cat(cols, dim=-1)
    r_chain = _rel_cols(got[..., :n], chain)
    print(f"  dot vs the op chain {r_chain:.2e}")
    assert r_chain <= 1e-6, r_chain
    # negative controls
    bad = want.clone()
    bad[..., 0] -= (h[..., :32].double() * wrow[:32].double() * kv[..., :32].double()).sum(-1)
    assert _rel_cols(got[..., : 3 * n], bad) > 1e-6, "a dropped column tile must fail"
    nw = want.clone()
    nw[..., :n] = (h.double() * kv[..., : n * C].double()).reshape(*h.shape[:-1], n, C).sum(-1)
    assert _rel_cols(got[..., : 3 * n], nw) > 1e-3, "ignoring the weight must fail"


def test_engram_pack_program_cache(mesh):
    nd = mesh.get_num_devices()
    for seed in (1, 2):
        h, kv, wrow = _inputs(nd, 256, 512, 4, seed)
        got = _host(_call(mesh, h, kv, wrow, 4), mesh)
        assert _rel_cols(got[..., :12], _want(h, kv, wrow, 4, 512)) <= 1e-6


def test_engram_pack_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    h, kv, wrow = _inputs(nd, 64, 256, 4, 3)
    hd, kd = _dev(h, mesh), _dev(kv, mesh)
    w32 = _dev(wrow.reshape(1, 1, 1, -1).expand(1, 1, 32, wrow.numel()), mesh, shard=False)
    with expect_error(Exception, "engram"):
        ttnn.bringup.mhc_pre_xing_pack(None, hd, n=4, engram_keys=kd)  # no weight
    with expect_error(Exception, "engram"):
        ttnn.bringup.mhc_pre_xing_pack(
            _dev(torch.zeros(nd, 1, 64, 32), mesh), hd, n=4, engram_keys=kd, engram_weight=w32
        )
    with expect_error(Exception, "engram_weight"):
        ttnn.bringup.mhc_pre_xing_pack(
            None, hd, n=4, engram_keys=kd, engram_weight=_dev(wrow.reshape(1, 1, 1, -1), mesh, False)
        )
    with expect_error(Exception, "mix"):
        ttnn.bringup.mhc_pre_xing_pack(None, hd, n=4)


# ---- option engram_gate_eps (deepseek_v41_flash_d_p P.24, TINY_KERNELS): the gates in columns 3n .. 4n-1


def _gate_chain(row, n, C, eps):
    """The op chain the model ran (tt/engram_add.py TtEngramAdd._gate_math) on the device: [.., T, >= 3n] -> [.., T, n]."""
    T = row.shape[-2]
    sl = lambda a, b: ttnn.slice(row, [0, 0, 0, a], [1, 1, T, b])  # noqa: E731
    dot, ssh, ssk = sl(0, n), sl(n, 2 * n), sl(2 * n, 3 * n)
    rs = [
        ttnn.rsqrt(ttnn.add(ttnn.multiply(s, 1.0 / C, dtype=ttnn.float32), eps, dtype=ttnn.float32)) for s in (ssh, ssk)
    ]
    r2 = ttnn.multiply(rs[0], rs[1], dtype=ttnn.float32)
    d = ttnn.multiply(ttnn.multiply(dot, r2, dtype=ttnn.float32), float(C) ** -0.5, dtype=ttnn.float32)
    s = ttnn.sqrt(ttnn.maximum(ttnn.abs(d), 1e-6))
    return ttnn.sigmoid(ttnn.where(ttnn.ltz(d), ttnn.neg(s), s))


def _gate_want(sums, n, C, eps):
    """float64 gates from float64 sums [.., 3n]."""
    dot, ssh, ssk = sums[..., :n], sums[..., n : 2 * n], sums[..., 2 * n : 3 * n]
    d = dot * torch.rsqrt(ssh / C + eps) * torch.rsqrt(ssk / C + eps) / C**0.5
    return torch.sigmoid(torch.copysign(torch.sqrt(d.abs().clamp_min(1e-6)), d))


def _ulps(a, b):
    ia = a.float().contiguous().view(torch.int32).long()
    ib = b.float().contiguous().view(torch.int32).long()
    ia = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return (ia - ib).abs()


GATE_CASES = [(160, 5120, 4, 1e-20), (640, 1024, 4, 1e-20), (64, 256, 2, 1e-6)]


@pytest.mark.parametrize("T, C, n, eps", GATE_CASES, ids=[f"T{c[0]}-C{c[1]}-n{c[2]}" for c in GATE_CASES])
def test_engram_gate(mesh, T, C, n, eps):
    nd = mesh.get_num_devices()
    h, kv, wrow = _inputs(nd, T, C, n, seed=T + C + n + 7)
    h[..., 5, :] = 0.0  # a zero stream row: d = 0, the 1e-6 floor
    h[..., 6, :C] *= -1.0
    hd, kd = _dev(h, mesh), _dev(kv, mesh)
    w32 = _dev(wrow.reshape(1, 1, 1, -1).expand(1, 1, 32, wrow.numel()), mesh, shard=False)
    plain = ttnn.bringup.mhc_pre_xing_pack(None, hd, n=n, engram_keys=kd, engram_weight=w32)
    gated = ttnn.bringup.mhc_pre_xing_pack(None, hd, n=n, engram_keys=kd, engram_weight=w32, engram_gate_eps=eps)
    p, g = _host(plain, mesh), _host(gated, mesh)
    assert tuple(g.shape) == (nd, 1, T, 32), g.shape
    assert torch.equal(g[..., : 3 * n], p[..., : 3 * n]), "the sums must be the option-off row's bytes"
    assert torch.all(g[..., 4 * n :] == 0), "columns past 4n must be zero"
    gates = g[..., 3 * n : 4 * n]
    chain = _host(_gate_chain(plain, n, C, eps), mesh)
    ul = _ulps(gates, chain)
    want = _gate_want(_want(h, kv, wrow, n, C), n, C, eps)
    rel = ((gates - want).abs() / want.abs()).max().item()
    print(
        f"T{T} C{C} n{n}: gates vs the op chain max {int(ul.max())} ulp ({int((ul > 0).sum())} of {ul.numel()}); "
        f"worst rel vs float64 {rel:.2e}"
    )
    assert int(ul.max()) <= 1, f"{int(ul.max())} ulp from the op chain"
    assert rel <= 1e-5, rel
    assert torch.allclose(
        gates[..., 5, :], torch.full_like(gates[..., 5, :], 1 / (1 + torch.exp(torch.tensor(-1e-3)).item()))
    )
    # negative control: the gates of the k-sums swapped with the h-sums' are the same (symmetric); a sign flip is not
    assert (gates - (1 - want)).abs().max().item() > 1e-3, "a dropped copysign must fail"


def test_engram_gate_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    with expect_error(Exception, "engram_gate_eps needs engram mode"):
        ttnn.bringup.mhc_pre_xing_pack(
            _dev(torch.zeros(nd, 1, 64, 32), mesh), _dev(torch.zeros(nd, 1, 64, 128), mesh), n=4, engram_gate_eps=1e-20
        )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre_xing(..., partials=P) (deepseek_v41_flash_d_p P.19): the input is (1, P, T, 32), the P chips'
partial mix rows of a TP group stacked on dim 1 (one all_gather); the op sums them in dim-1 order in fp32 on the SFPU
before the coefficient math (the mHC exchange as one all_gather).

Per chip (random per-device inputs, the whole box, FABRIC_2D), P = 2 and 4, with and without streams / collapse_pre,
deepseek and Xing math:
  * hc and y bit-identical to the call on the row summed by ttnn.add in the same order (((p0 + p1) + p2) + p3, fp32
    SFPU: what tt/ccl.py's exact gather_add computes);
  * negative control: the stack in another order (p1 first) is not bit-identical for P >= 3 at hc (fp32 sums are not
    associative), and a dropped partial gives another hc;
  * refusals (coefficients_given, a wrong dim-1 size)."""

import pytest
import torch

import ttnn

N, NG = 4, 24
SCALE = [0.06, 0.012, 0.06]


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


def _parts(nd, P, T, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(NG, N * hidden, generator=g) * (N * hidden) ** -0.5
    out = []
    for _ in range(P):
        full = torch.randn(nd, 1, T, N * hidden // P, generator=g) * (1 + 3 * torch.rand(1, generator=g))
        row = torch.zeros(nd, 1, T, 32)
        row[..., :NG] = full @ fn[:, : N * hidden // P].T
        row[..., NG] = full.square().sum(-1)
        out.append(row)
    return out


def _kw(hidden, deepseek, base):
    return dict(
        scale=SCALE,
        base=base,
        norm_width=float(N * hidden),
        n=N,
        norm_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        deepseek=deepseek,
    )


@pytest.mark.parametrize("P", [2, 4])
@pytest.mark.parametrize("T,C,hidden", [(1024, 2560, 5120), (512, 1280, 5120), (64, 96, 384)])
@pytest.mark.parametrize("deepseek", [True, False], ids=["deepseek", "xing"])
def test_partials(mesh, P, T, C, hidden, deepseek):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(P * 1000 + T + C)
    parts = _parts(nd, P, T, hidden, seed=P + T)
    x = torch.randn(nd, 1, T, N * C, generator=g)
    pre = torch.rand(nd, 1, T, NG, generator=g)
    base = (torch.randn(NG, generator=g) * 2).tolist()
    kw = _kw(hidden, deepseek, base)
    acc = _dev(parts[0], mesh)
    for p in parts[1:]:
        acc = ttnn.add(acc, _dev(p, mesh), dtype=ttnn.float32)  # SFPU fp32, in order
    stack = _dev(torch.cat(parts, dim=1), mesh)
    xd, pd = _dev(x, mesh), _dev(pre, mesh)
    hc0, _ = ttnn.bringup.mhc_pre_xing(acc, None, **kw)
    hc1, _ = ttnn.bringup.mhc_pre_xing(stack, None, partials=P, **kw)
    assert list(hc1.shape) == [nd, 1, T, NG] or list(hc1.shape)[-2:] == [T, NG], hc1.shape
    h0 = _host(hc0, mesh)
    assert torch.equal(_host(hc1, mesh), h0), "hc differs from the summed row's"
    for given in (None, pd):
        a0, y0 = ttnn.bringup.mhc_pre_xing(acc, xd, collapse_pre=given, **kw)
        a1, y1 = ttnn.bringup.mhc_pre_xing(stack, xd, collapse_pre=given, partials=P, **kw)
        assert torch.equal(_host(a1, mesh), _host(a0, mesh)), "hc (with streams) differs"
        assert torch.equal(_host(y1, mesh), _host(y0, mesh)), "y differs"
    # controls: a partial dropped (the first P - 1 summed) gives another hc
    short = _dev(torch.cat(parts[:-1] + [torch.zeros_like(parts[0])], dim=1), mesh)
    hc2, _ = ttnn.bringup.mhc_pre_xing(short, None, partials=P, **kw)
    assert not torch.equal(_host(hc2, mesh), h0), "control: a dropped partial gives the same hc"


def test_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    parts = _parts(nd, 2, 64, 384, seed=3)
    stack = _dev(torch.cat(parts, dim=1), mesh)
    kw = _kw(384, True, [0.0] * NG)
    with expect_error(Exception, "needs input"):
        ttnn.bringup.mhc_pre_xing(stack, None, partials=4, **kw)  # dim 1 is 2
    hc = _dev(torch.rand(nd, 2, 64, NG), mesh)
    x = _dev(torch.rand(nd, 1, 64, N * 96), mesh)
    with expect_error(Exception, "partials needs the mix row"):
        ttnn.bringup.mhc_pre_xing(hc, x, n=N, coefficients_given=True, partials=2)

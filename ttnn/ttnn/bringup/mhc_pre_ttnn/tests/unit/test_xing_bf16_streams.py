# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""bf16 streams (deepseek_v41_flash_d_p P.20, the GPU code's bf16 residual streams) through ttnn.bringup.mhc_pre_xing
(collapse with coefficients / collapse_pre / coefficients_given) and mhc_pre_xing_pack (sum of squares; engram mode
with bf16 keys).

The bf16 tiles are unpacked to the fp32 DEST exactly, so every sum is the fp32 program's on the same values. Per chip
(random per-device inputs on the whole box, FABRIC_2D):
  * pack: the row bit-identical to the fp32 call on the streams upcast to fp32;
  * collapse: hc bit-identical to the fp32 call's, y bf16 = the fp32 call's y rounded once by the packer (to nearest,
    ties away from zero) bit for bit, for collapse_pre and coefficients_given;
  * engram pack (bf16 h and bf16 keys): bit-identical to the fp32 call on the upcast values;
  * negative control: y truncated instead of rounded to nearest differs;
  * refusals (bf16 keys with fp32 streams, a bf16 mix row)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_TESTS = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_pre_ttnn_unit_bf16_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
N, NG = 4, 24
SCALE = [0.06, 0.012, 0.06]


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _streams(nd, T, C, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(nd, 1, T, N * C, generator=g) * 3).bfloat16().float()  # bf16-valued


def _rows(nd, T, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(NG, N * hidden, generator=g) * (N * hidden) ** -0.5
    row = torch.zeros(nd, 1, T, 32)
    row[..., :NG] = torch.randn(nd, 1, T, N * hidden, generator=g) @ fn.T
    row[..., NG] = (N * hidden) * (1 + torch.rand(nd, 1, T, generator=g))
    base = (torch.randn(NG, generator=g) * 2).tolist()
    row2 = torch.zeros(nd, 1, T, 32)
    row2[..., :NG] = torch.randn(nd, 1, T, N * hidden, generator=g) @ fn.T
    row2[..., NG] = (N * hidden) * (1 + torch.rand(nd, 1, T, generator=g))
    prev = ref.mhc_pre_xing_coefficients(row2, SCALE, base, float(N * hidden), N, 1e-6, 1e-6, 20, (-30, 30), True)
    return row, base, prev.float()


def _coef(row_d, x_d, base, hidden, **kw):
    return ttnn.bringup.mhc_pre_xing(
        row_d,
        x_d,
        scale=SCALE,
        base=base,
        norm_width=float(N * hidden),
        n=N,
        norm_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        deepseek=True,
        **kw,
    )


def _rna_bf16(t):
    """fp32 -> bf16 rounded to nearest, ties away from zero: the packer's fp32 -> bf16 rounding (ties are ~1e-5 of
    random values; round-to-nearest-even differs only there, by one ulp)."""
    b = t.float().contiguous().view(torch.int32)
    sign = b & torch.tensor(-0x80000000, dtype=torch.int32)
    mag = ((b & 0x7FFFFFFF) + 0x8000) & ~0xFFFF
    return (sign | mag).view(torch.float32)


def _rtz_bf16(t):
    """fp32 -> bf16 by truncation (the negative control for the packer's rounding)."""
    return (t.float().contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32)


SHAPES = [(1024, 2560, 5120), (512, 1280, 5120), (64, 256, 512)]  # DS41 2x2 / 4x4 per chip, small


@pytest.mark.parametrize("T, C, hidden", SHAPES)
def test_pack_bf16(mesh, T, C, hidden):
    nd = mesh.get_num_devices()
    x = _streams(nd, T, C, seed=T + C)
    mix = torch.randn(nd, 1, T, 32, generator=torch.Generator().manual_seed(C))
    mix_d = _dev(mix, mesh)
    got = _host(ttnn.bringup.mhc_pre_xing_pack(mix_d, _dev(x, mesh, ttnn.bfloat16), n=N), mesh)
    want = _host(ttnn.bringup.mhc_pre_xing_pack(mix_d, _dev(x, mesh), n=N), mesh)
    assert got.dtype == torch.float32
    assert torch.equal(got, want), "bf16 streams: the packed row differs from the fp32 call on the same values"
    ss = x.double().square().sum(-1)
    r = ((got[..., NG].double() - ss).norm() / ss.norm()).item()
    print(f"T {T} C {C}: sum x^2 vs float64 rel {r:.2e}")
    assert r <= 1e-6, r


@pytest.mark.parametrize("T, C, hidden", SHAPES)
def test_collapse_bf16(mesh, T, C, hidden):
    nd = mesh.get_num_devices()
    x = _streams(nd, T, C, seed=2 * T + C)
    row, base, prev = _rows(nd, T, hidden, seed=T + 3 * C)
    row_d, prev_d = _dev(row, mesh), _dev(prev, mesh)
    xb, xf = _dev(x, mesh, ttnn.bfloat16), _dev(x, mesh)
    for name, kw in (("coefficients", {}), ("collapse_pre", {"collapse_pre": prev_d})):
        hc_b, y_b = _coef(row_d, xb, base, hidden, **kw)
        hc_f, y_f = _coef(row_d, xf, base, hidden, **kw)
        assert y_b.dtype == ttnn.bfloat16, y_b.dtype
        assert torch.equal(_host(hc_b, mesh), _host(hc_f, mesh)), f"{name}: hc differs"
        yb, yf = _host(y_b, mesh).float(), _host(y_f, mesh)
        assert torch.equal(yb, _rna_bf16(yf)), f"{name}: y is not the fp32 y rounded to nearest (ties away)"
        assert not torch.equal(yb, _rtz_bf16(yf)), f"control ({name}): truncation gives the same y"
    none, y_b = ttnn.bringup.mhc_pre_xing(prev_d, xb, n=N, coefficients_given=True)
    none2, y_f = ttnn.bringup.mhc_pre_xing(prev_d, xf, n=N, coefficients_given=True)
    assert none is None and none2 is None and y_b.dtype == ttnn.bfloat16
    assert torch.equal(_host(y_b, mesh).float(), _rna_bf16(_host(y_f, mesh))), "coefficients_given: y"


ENGRAM = [(1024, 2560, 4), (512, 1280, 4), (64, 256, 4)]


@pytest.mark.parametrize("T, C, n", ENGRAM)
def test_engram_pack_bf16(mesh, T, C, n):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(T + C + 7)
    h = (torch.randn(nd, 1, T, n * C, generator=g) * 2).bfloat16().float()
    kv = torch.randn(nd, 1, T, (n + 1) * C, generator=g).bfloat16().float()
    wrow = (torch.randn(n * C, generator=g) * 0.1).float()
    w32 = _dev(wrow.reshape(1, 1, 1, -1).expand(1, 1, 32, n * C), mesh, shard=False)

    def call(dt):
        return _host(
            ttnn.bringup.mhc_pre_xing_pack(
                None, _dev(h, mesh, dt), n=n, engram_keys=_dev(kv, mesh, dt), engram_weight=w32
            ),
            mesh,
        )

    got, want = call(ttnn.bfloat16), call(ttnn.float32)
    assert torch.equal(got, want), "bf16 h / keys: the engram row differs from the fp32 call on the same values"
    assert torch.all(got[..., 3 * n :] == 0)


def test_bf16_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    h = _streams(nd, 64, 256, seed=1)
    kv = torch.randn(nd, 1, 64, 5 * 256)
    w32 = _dev(torch.zeros(1, 1, 32, 4 * 256), mesh, shard=False)
    with expect_error(Exception, "engram_keys and streams must share a dtype"):
        ttnn.bringup.mhc_pre_xing_pack(
            None, _dev(h, mesh, ttnn.bfloat16), n=N, engram_keys=_dev(kv, mesh), engram_weight=w32
        )
    with expect_error(Exception, "mix must be float32 TILE"):
        ttnn.bringup.mhc_pre_xing_pack(
            _dev(torch.zeros(nd, 1, 64, 32), mesh, ttnn.bfloat16), _dev(h, mesh, ttnn.bfloat16), n=N
        )

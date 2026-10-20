# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre_xing(..., collapse_pre=) (deepseek_v41_flash_d_p P.14): the coefficients from the reduced row
AND the collapse y = sum_i collapse_pre_i x_i (a given pre, the previous sublayer's: single-pass mHC) in one program.

Checks per chip (random per-device inputs on the whole box, FABRIC_2D):
  * hc is bit-identical to the plain coefficient call (same row, no streams);
  * y is bit-identical to mhc_pre_xing(collapse_pre, streams, coefficients_given=True) (the same SFPU collapse) and
    within 1e-6 of the float64 reference sum_i pre_i x_i;
  * negative control: y is far from the collapse with the pre just computed (the option is not ignored);
  * a collapse_pre of 32 columns (only 0..n-1 read) gives the same y;
  * refusals (coefficients_given, no streams, a too-wide pre)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_TESTS = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_pre_ttnn_unit_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
N, NG = 4, 24
SCALE = [0.06, 0.012, 0.06]  # DeepSeek-V4.1-like hc scales


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


def _inputs(nd, T, C, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(nd, 1, T, N * C, generator=g)
    full = torch.randn(nd, 1, T, N * hidden, generator=g)
    fn = torch.randn(NG, N * hidden, generator=g) * (N * hidden) ** -0.5
    row = torch.zeros(nd, 1, T, 32)
    row[..., :NG] = full @ fn.T
    row[..., NG] = full.square().sum(-1)
    base = (torch.randn(NG, generator=g) * 2).tolist()
    # the previous sublayer's coefficients (another row): the given pre
    row2 = torch.zeros(nd, 1, T, 32)
    row2[..., :NG] = torch.randn(nd, 1, T, N * hidden, generator=g) @ fn.T
    row2[..., NG] = (N * hidden) * (1 + torch.rand(nd, 1, T, generator=g))
    prev = ref.mhc_pre_xing_coefficients(row2, SCALE, base, float(N * hidden), N, 1e-6, 1e-6, 20, (-30, 30), True)
    return x, row, base, prev.float()


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


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("T, C, hidden", [(1024, 2560, 5120), (64, 256, 512), (1024, 1280, 5120)])
def test_collapse_pre(mesh, T, C, hidden):
    nd = mesh.get_num_devices()
    x, row, base, prev = _inputs(nd, T, C, hidden, seed=T + C)
    row_d, x_d, prev_d = _dev(row, mesh), _dev(x, mesh), _dev(prev, mesh)
    hc, y = _coef(row_d, x_d, base, hidden, collapse_pre=prev_d)
    hc0, none = _coef(row_d, None, base, hidden)
    assert none is None
    hc_h, hc0_h = _host(hc, mesh), _host(hc0, mesh)
    assert torch.equal(hc_h, hc0_h), "hc differs from the plain coefficient call"
    y_h = _host(y, mesh)
    none, y_given = ttnn.bringup.mhc_pre_xing(prev_d, x_d, n=N, coefficients_given=True)
    assert none is None
    assert torch.equal(y_h, _host(y_given, mesh)), "y differs from the coefficients_given collapse"
    want = ref.mhc_pre_xing_collapse(prev, x, N)
    r = _rel(y_h, want)
    print(f"T {T} C {C}: y vs float64 rel {r:.3e}")
    assert r <= 1e-6, r
    wrong = ref.mhc_pre_xing_collapse(hc_h, x, N)  # negative control: the pre just computed
    assert _rel(y_h, wrong) > 1e-3, "y matches the computed pre: collapse_pre ignored"
    # a 32-column collapse_pre (only 0..n-1 are read)
    wide = torch.cat([prev, torch.randn(*prev.shape[:-1], 32 - NG)], dim=-1)
    _, y32 = _coef(row_d, x_d, base, hidden, collapse_pre=_dev(wide, mesh))
    assert torch.equal(_host(y32, mesh), y_h), "a 32-column collapse_pre changed y"


def test_collapse_pre_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    x, row, base, prev = _inputs(nd, 64, 256, 512, seed=3)
    row_d, x_d, prev_d = _dev(row, mesh), _dev(x, mesh), _dev(prev, mesh)
    with expect_error(Exception, "collapse_pre"):
        _coef(row_d, None, base, 512, collapse_pre=prev_d)  # no streams
    with expect_error(Exception, "collapse_pre"):
        ttnn.bringup.mhc_pre_xing(prev_d, x_d, n=N, coefficients_given=True, collapse_pre=prev_d)
    too_wide = _dev(torch.zeros(nd, 1, 64, 40), mesh)
    with expect_error(Exception, "collapse_pre"):
        _coef(row_d, x_d, base, 512, collapse_pre=too_wide)

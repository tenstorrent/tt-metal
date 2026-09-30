# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre_xing / mhc_pre_xing_pack (Xing4.0 mode of the fork) against the float64 torch reference
(reference.py: mhc_pre_xing_coefficients = xing40_a4b_d_p reference/xing_ref.py:hc_weights after its projection,
mhc_pre_xing_collapse), one case per model call (xing_cases.py). Inputs differ per device (sharded over the mesh on
dim 0); each chip's output is checked on its own data. test_reference_matches_xing_ref checks the reference itself
against xing_ref.hc_weights on the CPU."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_pre_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("xing_cases").CASES
NG = 24


@pytest.fixture(scope="module")
def mesh():
    c = CASES[0]
    p = c["device_params"]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, p["fabric_config"]))
    m = ttnn.open_mesh_device(ttnn.MeshShape(*c["mesh"]), l1_small_size=p["l1_small_size"])
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh):
    return ttnn.from_torch(
        t.float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _host(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).double()


def _check(name, got, want, pcc, max_rel):
    got, want = got.double().reshape(want.shape), want.double()
    p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
    rel = ((got - want).norm() / want.norm()).item()
    print(f"{name}: pcc {p:.8f} rel {rel:.3e}")
    assert p >= pcc and rel <= max_rel, f"{name}: pcc {p:.8f} (>= {pcc}), rel L2 {rel:.3e} (<= {max_rel})"


def _inputs(case, n_dev):
    """Per-device random streams [n_dev, 1, T, n*C], a full-width projection -> reduced row [n_dev, 1, T, 32]
    (mixes of the full 4 x hidden row, sum x^2), base, and a valid hc for the collapse."""
    g = torch.Generator().manual_seed(case["seed"])
    n, T, C, H = case["n"], case["T"], case["C"], case["hidden"]
    x = torch.randn(n_dev, 1, T, n * C, generator=g)
    full = torch.randn(n_dev, 1, T, n * H, generator=g)  # the whole row the all_reduce sums over
    fn = torch.randn(NG, n * H, generator=g) * (n * H) ** -0.5
    row = torch.zeros(n_dev, 1, T, 32)
    row[..., :NG] = full @ fn.T
    row[..., NG] = full.square().sum(-1)
    base = (torch.randn(NG, generator=g) * 0.5).tolist()
    return x, row, base


def _coef_ref(case, row, base):
    return ref.mhc_pre_xing_coefficients(
        row,
        case["scale"],
        base,
        float(case["n"] * case["hidden"]),
        case["n"],
        case["norm_eps"],
        case["hc_eps"],
        case["sinkhorn_iters"],
        case["clamp"],
    )


def _call_coef(case, row_d, streams_d, base):
    return ttnn.bringup.mhc_pre_xing(
        row_d,
        streams_d,
        scale=case["scale"],
        base=base,
        norm_width=float(case["n"] * case["hidden"]),
        n=case["n"],
        norm_eps=case["norm_eps"],
        hc_eps=case["hc_eps"],
        sinkhorn_iters=case["sinkhorn_iters"],
        clamp_min=case["clamp"][0],
        clamp_max=case["clamp"][1],
    )


def _check_hc(case, got, want):
    n = case["n"]
    for name, sl in (("pre", slice(0, n)), ("post", slice(n, 2 * n)), ("comb", slice(2 * n, NG))):
        _check(f"{case['id']} {name}", got[..., sl], want[..., sl], case["pcc"], case["max_rel"][name])
    col = (got[..., 2 * n :].reshape(*got.shape[:-1], n, n).sum(-2) - 1).abs().max().item()
    assert col <= 1e-5, f"comb column sums off 1 by {col}"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_mhc_pre_xing(mesh, case):
    n_dev = mesh.get_num_devices()
    n = case["n"]
    x, row, base = _inputs(case, n_dev)
    if case["mode"] == "pack":
        mix = row.clone()
        mix[..., NG] = 0.0
        out = ttnn.bringup.mhc_pre_xing_pack(_dev(mix, mesh), _dev(x, mesh), n=n)
        got = _host(out, mesh)
        _check(f"{case['id']} ss", got[..., NG], x.double().square().sum(-1), case["pcc"], case["max_rel"]["ss"])
        other = torch.cat([got[..., :NG], got[..., NG + 1 :]], -1)
        assert torch.equal(other, torch.cat([mix[..., :NG], mix[..., NG + 1 :]], -1).double()), "mix columns changed"
        return
    want = _coef_ref(case, row, base)
    if case["mode"] == "coef":
        hc, y = _call_coef(case, _dev(row, mesh), None, base)
        assert y is None and tuple(hc.shape)[-1] == NG
        _check_hc(case, _host(hc, mesh), want)
    elif case["mode"] == "collapse":
        hc_in = want.float()
        hc, y = ttnn.bringup.mhc_pre_xing(_dev(hc_in, mesh), _dev(x, mesh), n=n, coefficients_given=True)
        assert hc is None
        _check(
            f"{case['id']} y", _host(y, mesh), ref.mhc_pre_xing_collapse(hc_in, x, n), case["pcc"], case["max_rel"]["y"]
        )
    else:  # both
        hc, y = _call_coef(case, _dev(row, mesh), _dev(x, mesh), base)
        _check_hc(case, _host(hc, mesh), want)
        _check(
            f"{case['id']} y", _host(y, mesh), ref.mhc_pre_xing_collapse(want, x, n), case["pcc"], case["max_rel"]["y"]
        )


def test_reference_matches_xing_ref():
    """CPU only: the reduced-row reference == xing_ref.hc_weights on the same streams (a wrong gate, eps, clamp or
    Sinkhorn order differs by > 1e-3)."""
    try:
        from models.demos.xing40_a4b_d_p.reference.xing_ref import hc_weights
    except ImportError:
        pytest.skip("xing40_a4b_d_p reference not importable")
    from types import SimpleNamespace

    g = torch.Generator().manual_seed(7)
    n, H, S = 4, 256, 64
    cfg = SimpleNamespace(
        hc_mult=n,
        rms_norm_eps=1e-6,
        mhc_h_res_clamp_min=-30.0,
        mhc_h_res_clamp_max=30.0,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )
    x = torch.randn(S * n, H, generator=g, dtype=torch.float64)
    fn = torch.randn(NG, n * H, generator=g, dtype=torch.float64) * (n * H) ** -0.5
    base = torch.randn(NG, generator=g, dtype=torch.float64)
    for scale in ([0.8, 0.5, 3.0], [0.8, 0.5, 40.0]):  # the second saturates the clamp
        want = hc_weights(x, fn, base, torch.tensor(scale, dtype=torch.float64), cfg)
        flat = x.reshape(S, -1)
        row = torch.zeros(S, 32, dtype=torch.float64)
        row[:, :NG] = flat @ fn.T
        row[:, NG] = flat.square().sum(-1)
        got = ref.mhc_pre_xing_coefficients(row, scale, base.tolist(), float(n * H), n, 1e-6, 1e-6, 20, (-30.0, 30.0))
        # hc_weights rounds its mixes to float32 (mix = F.linear(...).float()): agreement to float32 level
        assert torch.allclose(got, want.double(), rtol=1e-5, atol=1e-6), (got - want).abs().max()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre_xing / mhc_pre_xing_pack (Xing4.0 mode of the fork) against the float64 torch reference
(reference.py: mhc_pre_xing_coefficients = xing40_a4b_d_p reference/xing_ref.py:hc_weights after its projection,
mhc_pre_xing_collapse), one case per model call (xing_cases.py). Inputs differ per device (sharded over the mesh on
dim 0); each chip's output is checked on its own data. test_reference_matches_xing_ref checks the reference itself
against xing_ref.hc_weights on the CPU.

Cases with `noise` (deepseek_v41_flash_d_p O.1: partials + collapse_pre + bf16 streams, the bf16 pack, the engram pack)
run in test_mhc_pre_xing_ds41, every output within fork_cases.noise_limit at its declared formats (F139)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit

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
    # The captured case mesh when the box has that many chips; otherwise the whole box (the inputs are per device and
    # the per-chip math does not depend on the mesh shape), as in test_mhc_pre_ttnn.py.
    c = CASES[0]
    p = c["device_params"]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, p["fabric_config"]))
    if c["mesh"][0] * c["mesh"][1] == ttnn.get_num_devices():
        m = ttnn.open_mesh_device(ttnn.MeshShape(*c["mesh"]), l1_small_size=p["l1_small_size"])
    else:
        m = ttnn.open_mesh_device(l1_small_size=p["l1_small_size"])
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


def _inputs(case, n_dev, seed=None):
    """Per-device random streams [n_dev, 1, T, n*C], a full-width projection -> reduced row [n_dev, 1, T, 32]
    (mixes of the full 4 x hidden row, sum x^2), base, and a valid hc for the collapse."""
    g = torch.Generator().manual_seed(case["seed"] if seed is None else seed)
    n, T, C, H = case["n"], case["T"], case["C"], case["hidden"]
    x = torch.randn(n_dev, 1, T, n * C, generator=g)
    full = torch.randn(n_dev, 1, T, n * H, generator=g)  # the whole row the all_reduce sums over
    fn = torch.randn(NG, n * H, generator=g) * (n * H) ** -0.5
    row = torch.zeros(n_dev, 1, T, 32)
    row[..., :NG] = full @ fn.T
    row[..., NG] = full.square().sum(-1)
    base = (torch.randn(NG, generator=g) * 0.5).tolist()
    if "base" in case:  # O.1 cases: the captured call's base scalars
        base = list(case["base"])
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
        deepseek=case.get("deepseek", False),
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
        **({"deepseek": True} if case.get("deepseek") else {}),
    )


def _check_hc(case, got, want):
    n = case["n"]
    for name, sl in (("pre", slice(0, n)), ("post", slice(n, 2 * n)), ("comb", slice(2 * n, NG))):
        _check(f"{case['id']} {name}", got[..., sl], want[..., sl], case["pcc"], case["max_rel"][name])
    col = (got[..., 2 * n :].reshape(*got.shape[:-1], n, n).sum(-2) - 1).abs().max().item()
    assert col <= 1e-5, f"comb column sums off 1 by {col}"


def _device_call(case, mesh, seed):
    """Host (x, row, base, reference coefficients) for `seed` and a call of the case's mode on their device tensors
    (built once here; the call only runs the op)."""
    n = case["n"]
    x, row, base = _inputs(case, mesh.get_num_devices(), seed)
    want = None if case["mode"] == "pack" else _coef_ref(case, row, base)
    if case["mode"] == "pack":
        mix = row.clone()
        mix[..., NG] = 0.0
        row = mix
        args = (_dev(mix, mesh), _dev(x, mesh))
        call = lambda: ttnn.bringup.mhc_pre_xing_pack(*args, n=n)
    elif case["mode"] == "coef":
        args = (_dev(row, mesh),)
        call = lambda: _call_coef(case, args[0], None, base)
    elif case["mode"] == "collapse":
        args = (_dev(want.float(), mesh), _dev(x, mesh))
        call = lambda: ttnn.bringup.mhc_pre_xing(*args, n=n, coefficients_given=True)
    else:  # both
        args = (_dev(row, mesh), _dev(x, mesh))
        call = lambda: _call_coef(case, args[0], args[1], base)
    return (x, row, base, want), call


_LEGACY = [c for c in CASES if "noise" not in c]


@pytest.mark.parametrize("case", _LEGACY, ids=[c["id"] for c in _LEGACY])
def test_mhc_pre_xing(mesh, case):
    n = case["n"]
    (x, row, base, want), run_a = _device_call(case, mesh, case["seed"])
    out = run_a()
    if case["mode"] == "pack":
        mix = row
        got = _host(out, mesh)
        _check(f"{case['id']} ss", got[..., NG], x.double().square().sum(-1), case["pcc"], case["max_rel"]["ss"])
        other = torch.cat([got[..., :NG], got[..., NG + 1 :]], -1)
        assert torch.equal(other, torch.cat([mix[..., :NG], mix[..., NG + 1 :]], -1).double()), "mix columns changed"
    elif case["mode"] == "coef":
        hc, y = out
        assert y is None and tuple(hc.shape)[-1] == NG
        _check_hc(case, _host(hc, mesh), want)
    elif case["mode"] == "collapse":
        hc_in = want.float()
        hc, y = out
        assert hc is None
        _check(
            f"{case['id']} y", _host(y, mesh), ref.mhc_pre_xing_collapse(hc_in, x, n), case["pcc"], case["max_rel"]["y"]
        )
    else:  # both
        hc, y = out
        _check_hc(case, _host(hc, mesh), want)
        _check(
            f"{case['id']} y", _host(y, mesh), ref.mhc_pre_xing_collapse(want, x, n), case["pcc"], case["max_rel"]["y"]
        )
    run_b = _device_call(case, mesh, case["seed"] + 1)[1]
    determinism.assert_deterministic(run_a, run_b, first=out, label=case["id"])


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


def test_reference_matches_deepseek_kernel():
    """CPU only: the deepseek=True reference == deepseek_v41_flash_d_p reference/kernels.py:sinkhorn_split (the
    CPU port of DeepSeek-V4's hc_split_sinkhorn_kernel, which computes in fp32) on the same mixes, element-wise to fp32
    level (measured <= 2.5e-6 relative); the Xing math (clamp, no pre eps, iters x (row, column)) differs by ~100 %
    on the entries the clamp and the eps placement touch."""
    try:
        from models.demos.deepseek_v41_flash_d_p.reference.kernels import sinkhorn_split
    except ImportError:
        pytest.skip("deepseek_v41_flash_d_p reference not importable")
    g = torch.Generator().manual_seed(11)
    n, H, S = 4, 256, 64
    flat = torch.randn(S, n * H, generator=g, dtype=torch.float64)
    fn = torch.randn(NG, n * H, generator=g, dtype=torch.float64) * (n * H) ** -0.5
    base = torch.randn(NG, generator=g, dtype=torch.float64) * 3
    base[2 * n] = -40.0  # a logit far below the old [-30, 30] clamp
    for scale in ([0.8, 0.5, 3.0], [0.06, 0.012, 0.06]):
        mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        pre, post, comb = sinkhorn_split(mixes.double(), torch.tensor(scale).double(), base, n, 20, 1e-6)
        want = torch.cat([pre, post, comb.flatten(-2)], dim=-1).double()
        row = torch.zeros(S, 32, dtype=torch.float64)
        row[:, :NG] = flat @ fn.T
        row[:, NG] = flat.square().sum(-1)
        got = ref.mhc_pre_xing_coefficients(
            row, scale, base.tolist(), float(n * H), n, 1e-6, 1e-6, 20, (-30.0, 30.0), deepseek=True
        )
        rel = ((got - want).abs() / want.abs()).max().item()
        assert rel <= 2e-5, rel
        xing = ref.mhc_pre_xing_coefficients(row, scale, base.tolist(), float(n * H), n, 1e-6, 1e-6, 20, (-30.0, 30.0))
        assert ((xing - want).abs() / want.abs()).max() > 1e-2, "the Xing math must differ from DeepSeek's"


# ---------------------------------------------------------------------------------------------------------------- DS41
_DS41 = [c for c in CASES if "noise" in c]


def _put(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _ds41_case(case, mesh, seed):
    """Host tensors (dict) and the device call (built once; the call only runs the op) of a DS41 case."""
    g = torch.Generator().manual_seed(seed)
    nd = mesh.get_num_devices()
    n, H, C, T = case["n"], case["hidden"], case["C"], case["T"]
    host = {"x": torch.randn(nd, 1, T, n * C, generator=g).to(torch.bfloat16)}  # the chip's bf16 streams
    if case["mode"] == "coef":
        # the TP group's partial rows: the two column halves of a random full-width row's projection / sum x^2
        full = torch.randn(nd, 1, T, n * H, generator=g)
        fn = torch.randn(NG, n * H, generator=g) * (n * H) ** -0.5
        P = case["partials"]
        rows = torch.zeros(nd, P, T, 32)
        for p in range(P):
            cols = slice(p * n * H // P, (p + 1) * n * H // P)
            rows[:, p, :, :NG] = full[:, 0, :, cols] @ fn[:, cols].T
            rows[:, p, :, NG] = full[:, 0, :, cols].square().sum(-1)
        row = rows[:, :1]
        for p in range(1, P):
            row = row + rows[:, p : p + 1]  # the op's exact fp32 sum, in dim-1 order
        host.update(row=row, pre=torch.rand(nd, 1, T, NG, generator=g))
        args = (_put(rows, mesh), _put(host["x"], mesh, ttnn.bfloat16))
        pre_d = _put(host["pre"], mesh)
        call = lambda: ttnn.bringup.mhc_pre_xing(  # noqa: E731
            *args,
            scale=case["scale"],
            base=case["base"],
            norm_width=float(n * H),
            n=n,
            norm_eps=case["norm_eps"],
            hc_eps=case["hc_eps"],
            sinkhorn_iters=case["sinkhorn_iters"],
            deepseek=True,
            collapse_pre=pre_d,
            partials=P,
        )
    elif case["mode"] == "pack":
        mix = torch.zeros(nd, 1, T, 32)
        mix[..., :NG] = torch.randn(nd, 1, T, NG, generator=g)
        host["mix"] = mix
        args = (_put(mix, mesh), _put(host["x"], mesh, ttnn.bfloat16))
        call = lambda: ttnn.bringup.mhc_pre_xing_pack(*args, n=n)  # noqa: E731
    else:  # engram
        host["x"] = (host["x"].float() * 2).to(torch.bfloat16)
        host["k"] = torch.randn(nd, 1, T, case["keys_width"], generator=g).to(torch.bfloat16)
        host["w"] = torch.randn(n * C, generator=g) * 0.1
        w32 = host["w"].reshape(1, 1, 1, -1).expand(1, 1, 32, n * C)
        args = (
            _put(host["x"], mesh, ttnn.bfloat16),
            _put(host["k"], mesh, ttnn.bfloat16),
            _put(w32, mesh, shard=False),
        )
        call = lambda: ttnn.bringup.mhc_pre_xing_pack(  # noqa: E731
            None, args[0], n=n, engram_keys=args[1], engram_weight=args[2]
        )
    return host, call


def _ds41_check(name, got, want, lim):
    got, want = got.double().reshape(want.shape), want.double()
    a, b = (got - got.mean()).flatten(), (want - want.mean()).flatten()
    p = float(a @ b / (a.norm() * b.norm()))
    rel = float((got - want).norm() / want.norm())
    print(
        f"{name}: rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e}) pcc {p:.9f} (min {lim['pcc']:.9f})"
    )
    assert rel <= lim["rel"], f"{name}: rel L2 {rel:.3e} > limit {lim['rel']:.3e} (pcc {p:.9f}, a diagnostic)"


def _sumsq(t):
    return ref.running_sum(t * t) if t.dtype == torch.float32 else (t * t).sum(-1)


@pytest.mark.parametrize("case", _DS41, ids=[c["id"] for c in _DS41])
def test_mhc_pre_xing_ds41(mesh, case):
    n, nz = case["n"], case["noise"]
    host, run_a = _ds41_case(case, mesh, case["seed"])
    out = run_a()
    nd = mesh.get_num_devices()
    for d in range(nd):
        one = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[d]).double()  # noqa: E731
        x = host["x"][d : d + 1]
        if case["mode"] == "coef":
            hc, y = out
            row = host["row"][d : d + 1]
            args = (case["scale"], case["base"], float(n * case["hidden"]), n, case["norm_eps"], case["hc_eps"])
            coef = lambda r: ref.mhc_pre_xing_coefficients(  # noqa: E731
                r, *args, case["sinkhorn_iters"], case["clamp"], deepseek=True, dtype=r.dtype
            )
            want = coef(row.double())
            got = one(hc).reshape(want.shape)
            for name, sl in (("pre", slice(0, n)), ("post", slice(n, 2 * n)), ("comb", slice(2 * n, NG))):
                lim = noise_limit(lambda r: coef(r)[..., sl], [row], nz["hc"]["inputs"], nz["hc"]["out"])
                _ds41_check(f"{case['id']} dev {d} {name}", got[..., sl], want[..., sl], lim)
            pre = host["pre"][d : d + 1, ..., :n]
            want_y = ref.mhc_pre_xing_collapse(pre, x, n)
            fy = lambda p, xx: ref.mhc_pre_xing_collapse(p, xx, n).to(p.dtype)  # noqa: E731
            assert y.dtype == ttnn.bfloat16, y.dtype
            _ds41_check(f"{case['id']} dev {d} y", one(y), want_y, noise_limit(fy, [pre, x], *nz["y"].values()))
        elif case["mode"] == "pack":
            got = one(out).reshape(*x.shape[:-1], 32)
            mix = host["mix"][d : d + 1].double()
            assert torch.equal(got[..., :NG], mix[..., :NG]) and torch.equal(got[..., NG + 1 :], mix[..., NG + 1 :])
            lim = noise_limit(_sumsq, [x], nz["ss"]["inputs"], nz["ss"]["out"])
            _ds41_check(f"{case['id']} dev {d} ss", got[..., NG], x.double().square().sum(-1), lim)
        else:  # engram
            got = one(out).reshape(*x.shape[:-1], 32)
            k, w = host["k"][d : d + 1], host["w"]
            fe = lambda hh, kk, ww: ref.engram_pack(hh, kk, ww, n, running=hh.dtype == torch.float32)  # noqa: E731
            want = ref.engram_pack(x.double(), k.double(), w.double(), n)
            for name, sl in (("dot", slice(0, n)), ("hh", slice(n, 2 * n)), ("kk", slice(2 * n, 3 * n))):
                lim = noise_limit(lambda *t: fe(*t)[..., sl], [x, k, w], nz["row"]["inputs"], nz["row"]["out"])
                _ds41_check(f"{case['id']} dev {d} {name}", got[..., sl], want[..., sl], lim)
            assert bool((got[..., 3 * n :] == 0).all()), "engram row: columns past 3 n must be zero"
    run_b = _ds41_case(case, mesh, case["seed"] + 1)[1]
    determinism.assert_deterministic(run_a, run_b, first=out, label=case["id"])

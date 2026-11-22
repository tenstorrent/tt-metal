# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post against its torch semantics (reference.py), one random-input case per captured model call
(cases.py). Math: PCC plus a max relative error (rel L2) per output, per device. The inputs differ per device (sharded
over the mesh on dim 0), so each chip's output is checked on its own data; the weights are replicated.

Cases with `noise` (deepseek_v41_flash_d_p O.1: bf16 streams, packed mixes, stats_weight) run in test_mhc_post_ds41,
each output within fork_cases.noise_limit at its declared formats (F139)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_post_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES
_DTYPE = {"BFLOAT16": (ttnn.bfloat16, torch.bfloat16), "FLOAT32": (ttnn.float32, torch.float32)}


@pytest.fixture(scope="module")
def mesh():
    # Open the mesh of a case captured on a box of this size (a 2x2 submesh of a 4x2 box fails the 2D fabric
    # handshake); the per-chip math does not depend on the mesh shape. No case captured on a box of this size (e.g.
    # the 32-chip Galaxy): the whole box, as test_mhc_pre_xing.py (deepseek_v41_flash_d_p P.16).
    n_sys = ttnn.get_num_devices()
    c = next((c for c in CASES if c["mesh"][0] * c["mesh"][1] == n_sys), None)
    p = (c or CASES[0])["device_params"]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, p["fabric_config"]))
    if c is not None:
        m = ttnn.open_mesh_device(ttnn.MeshShape(*c["mesh"]), l1_small_size=p["l1_small_size"])
    else:
        m = ttnn.open_mesh_device(l1_small_size=p["l1_small_size"])
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _devices(mesh):
    return mesh.get_num_devices()


def _host(spec, g, per_device, n_dev, scale=1.0):
    """per_device: a different random tensor per chip, stacked on dim 0 (sharded); else one (replicated)."""
    shape = list(spec["shape"])
    if per_device:
        shape = [shape[0] * n_dev] + shape[1:]
    return (torch.randn(shape, generator=g) * scale).to(_DTYPE[spec["dtype"]][1])


def _to_device(t, spec, mesh, per_device):
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if per_device else ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(
        t,
        dtype=_DTYPE[spec["dtype"]][0],
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _from_device(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _check(name, got, want, pcc, max_rel):
    got, want = got.double().reshape(want.shape), want.double()
    p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
    rel = ((got - want).norm() / want.norm()).item()
    assert p >= pcc and rel <= max_rel, f"{name}: pcc {p:.7f} (>= {pcc}), rel L2 {rel:.6f} (<= {max_rel})"
    return p, rel


def _inputs(case, seed, mesh):
    """Host (f, x, post, comb) for `seed` and their device tensors."""
    g = torch.Generator().manual_seed(seed)
    n_dev = _devices(mesh)
    f = _host(case["input"], g, True, n_dev)
    x = _host(case["residual"], g, True, n_dev)
    post = (_host(case["post"], g, True, n_dev).float().sigmoid() * 2).to(torch.float32)  # post = 2 sigmoid(.)
    comb = _host(case["comb"], g, True, n_dev).float().abs().to(torch.float32) / 4  # a comb-like positive mix
    dev = tuple(
        _to_device(t, case[k], mesh, True) for t, k in ((f, "input"), (x, "residual"), (post, "post"), (comb, "comb"))
    )
    return (f, x, post, comb), dev


_LEGACY = [c for c in CASES if "noise" not in c]


@pytest.mark.parametrize("case", _LEGACY, ids=[c["id"] for c in _LEGACY])
def test_mhc_post(mesh, case):
    (f, x, post, comb), dev = _inputs(case, case["seed"], mesh)
    kw = {"comb_transposed": case["comb_transposed"]} if "comb_transposed" in case else {}
    out = ttnn.bringup.mhc_post(*dev, **kw)
    want = ref.mhc_post(f, x, post, comb, **kw)
    p, r = _check("out", _from_device(out, mesh), want, case["pcc"], case["max_rel"])
    if kw.get("comb_transposed") is False:
        # the option must change the math: the default (comb^T) result is far from the comb-as-stored one
        _, r_t = _check("out_t", want, ref.mhc_post(f, x, post, comb), -1.0, float("inf"))
        assert r_t > 50 * max(r, 1e-6), f"comb vs comb^T references too close (rel {r_t})"
    print(f"{case['id']}: pcc {p:.7f} rel {r:.6f}")
    dev_b = _inputs(case, case["seed"] + 1, mesh)[1]
    determinism.assert_deterministic(
        lambda: ttnn.bringup.mhc_post(*dev, **kw),
        lambda: ttnn.bringup.mhc_post(*dev_b, **kw),
        first=out,
        label=case["id"],
    )


# ---------------------------------------------------------------------------------------------------------------- DS41
_NOISE = [c for c in CASES if "noise" in c]


def _r9(w):
    """fp32 -> 9 explicit mantissa bits, nearest-even (what the FPU's srcA multiplies; the model's fpu_weight)."""
    i = w.float().contiguous().view(torch.int32)
    sh = 14
    lsb = (i >> sh) & 1
    return (((i + ((1 << (sh - 1)) - 1) + lsb) >> sh) << sh).view(torch.float32)


def _ds41_inputs(case, seed, mesh):
    """Host (f, x, post, comb, w) for `seed` (post / comb sliced out of the packed mixes when the case packs them) and
    the device call's (args, kwargs)."""
    g = torch.Generator().manual_seed(seed)
    n_dev = _devices(mesh)
    f = _host(case["input"], g, True, n_dev)
    x = (_host(case["residual"], g, True, n_dev).float() * 3).to(torch.bfloat16)
    n = 4
    if "mixes" in case:
        mixes = torch.rand([n_dev] + case["mixes"]["shape"][1:], generator=g)
        mixes[..., n : 2 * n] *= 2
        post, comb = mixes[..., case["post_col"] : case["post_col"] + n], mixes[..., case["comb_col"] :]
        md = _to_device(mixes, case["mixes"], mesh, True)
        args = [_to_device(f, case["input"], mesh, True), _to_device(x, case["residual"], mesh, True), md, md]
        kw = {"post_col": case["post_col"], "comb_col": case["comb_col"]}
    else:
        post = torch.rand([n_dev] + case["post"]["shape"][1:], generator=g) * 2
        comb = torch.rand([n_dev] + case["comb"]["shape"][1:], generator=g)
        args = [
            _to_device(t, case[k], mesh, True)
            for t, k in ((f, "input"), (x, "residual"), (post, "post"), (comb, "comb"))
        ]
        kw = {}
    w = None
    if "stats_weight" in case:
        w = torch.randn(case["stats_weight"]["shape"], generator=g) * case["stats_weight"]["shape"][-2] ** -0.5
        w[..., 6 * n :] = 0
        w = _r9(w)
        kw["stats_weight"] = _to_device(w, case["stats_weight"], mesh, False)
        ck = case["compute_kernel_config"]
        kw["compute_kernel_config"] = ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
            math_approx_mode=ck["math_approx_mode"],
            fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        )
    return (f, x, post.contiguous(), comb.contiguous(), w), args, kw


def _ds41_check(name, got, want, lim):
    got, want = got.double().reshape(want.shape), want.double()
    a, b = (got - got.mean()).flatten(), (want - want.mean()).flatten()
    p = float(a @ b / (a.norm() * b.norm()))
    rel = float((got - want).norm() / want.norm())
    print(
        f"{name}: rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e}) pcc {p:.8f} (min {lim['pcc']:.8f})"
    )
    assert rel <= lim["rel"], f"{name}: rel L2 {rel:.3e} > limit {lim['rel']:.3e} (pcc {p:.8f}, a diagnostic)"


@pytest.mark.parametrize("case", _NOISE, ids=[c["id"] for c in _NOISE])
def test_mhc_post_ds41(mesh, case):
    (f, x, post, comb, w), args, kw = _ds41_inputs(case, case["seed"], mesh)
    out = ttnn.bringup.mhc_post(*args, **kw)
    xo, st = out if "stats_weight" in kw else (out, None)
    assert xo.dtype == ttnn.bfloat16, xo.dtype
    got_x = _from_device(xo, mesh)
    nz = case["noise"]
    # (the reference per chip: the noise floor's rel is per-tensor, as the check's)
    for d in range(_devices(mesh)):
        ins = [f[d : d + 1], x[d : d + 1], post[d : d + 1], comb[d : d + 1]]
        fn = lambda ff, xx, pp, cc: ref.mhc_post(ff, xx, pp, cc).to(ff.dtype)  # noqa: E731
        want = ref.mhc_post(*ins)
        _ds41_check(
            f"{case['id']} dev {d} y", got_x[d : d + 1], want, noise_limit(fn, ins, nz["y"]["inputs"], nz["y"]["out"])
        )
        if st is None:
            continue
        s = ttnn.to_torch(ttnn.get_device_tensors(st)[d]).double().reshape(*want.shape[:-1], 32)
        k = 24  # n (n + 2) projection columns, then sum X'^2, then zeros
        # the projection of the device's own bf16 X' (K-40), at the call's fidelity (reference.fpu_operands)
        xp = got_x[d : d + 1].reshape(-1, want.shape[-1]).float()
        wk = w.reshape(-1, 32)[:, :k]
        fid = case["compute_kernel_config"]["math_fidelity"]

        def proj(a, b):
            if a.dtype == torch.float64:
                return a @ b
            a, b = ref.fpu_operands(a, b, fid)
            return a @ b

        want_p = xp.double() @ wk.double()
        lim_p = noise_limit(proj, [xp, wk], nz["proj"]["inputs"], nz["proj"]["out"])
        _ds41_check(f"{case['id']} dev {d} proj ({fid})", s.reshape(-1, 32)[:, :k], want_p, lim_p)
        want_ss = ref.mhc_post_sumsq(*[t.double() for t in ins])
        lim_ss = noise_limit(ref.mhc_post_sumsq, ins, nz["ss"]["inputs"], nz["ss"]["out"])
        _ds41_check(f"{case['id']} dev {d} ss", s[..., k], want_ss, lim_ss)
        assert bool((s[..., k + 1 :] == 0).all()), "stats columns past n (n + 2) must be zero"
    args_b, kw_b = _ds41_inputs(case, case["seed"] + 1, mesh)[1:]
    determinism.assert_deterministic(
        lambda: ttnn.bringup.mhc_post(*args, **kw),
        lambda: ttnn.bringup.mhc_post(*args_b, **kw_b),
        first=out,
        label=case["id"],
    )

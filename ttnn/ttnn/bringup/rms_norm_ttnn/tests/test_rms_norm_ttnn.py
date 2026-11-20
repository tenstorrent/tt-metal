# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rms_norm against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Math: PCC plus an elementwise atol/rtol bound, per device. The input differs per device (sharded over the
mesh columns) so each chip's output is checked on its own data; the weight is replicated.

Cases with `noise` (deepseek_v41_flash_d_p O.1) run in test_rms_norm_ttnn_ds41: the split program's options (stats_out,
partial_stats / stats_width, input_window, dtype); rel L2 (PCC printed, a diagnostic) vs the float64 reference within
fork_cases.noise_limit at the case's declared formats (F139)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_rms_norm_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    # A case runs on a box of its own mesh size only (conftest skips it elsewhere): a smaller mesh opened on a bigger
    # box fails the FABRIC_2D router handshake (e.g. a 2x2 case on a 4x2 box), and the case's math depends on its mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def _host_dtype(spec):
    return torch.float32 if spec["dtype"] == "FLOAT32" else torch.bfloat16


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES if "noise" not in c],
    ids=[c["id"] for c in CASES if "noise" not in c],
    indirect=["mesh_device", "device_params"],
)
def test_rms_norm_ttnn(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    xs = c["input"]
    rs = c.get("residual")
    (x, w, r), (tt_x, tt_w, tt_r, cfg, extra) = _inputs(mesh_device, c, c["seed"])
    out = _call(c, tt_x, tt_w, cfg, extra)
    first = out
    if c.get("return_residual_sum"):
        assert isinstance(out, tuple) and len(out) == 2, f"expected (y, t), got {type(out)}"
        out, t_sum = out
        # t = x + residual must be bit-identical to the device's own add of the same two tensors (the option's
        # contract; torch's bf16 add can differ by one ulp from the FPU's rounding). Every element is checked.
        added = ttnn.add(tt_x, tt_r, dtype=getattr(ttnn.DataType, xs["dtype"]))
        t_outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(t_sum)]
        add_outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(added)]
        for d in range(n_dev):
            got_t, want_t = t_outs[d].reshape(xs["shape"]), add_outs[d].reshape(xs["shape"])
            n_diff = int((got_t.view(torch.int16) != want_t.view(torch.int16)).sum())
            assert n_diff == 0, f"dev {d}: residual sum differs from ttnn.add in {n_diff} elements"
            t_ref = x[d : d + 1].reshape(xs["shape"]).double() + r[d : d + 1].reshape(xs["shape"]).double()
            assert _pcc(got_t, t_ref) >= 0.9999, f"dev {d}: residual sum pcc vs torch"
    outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev
    for d in range(n_dev):
        got = outs[d].reshape(xs["shape"]).double()
        want = ref.rms_norm(
            x[d : d + 1].reshape(xs["shape"]),
            epsilon=c["epsilon"],
            weight=w,
            residual=r[d : d + 1].reshape(xs["shape"]) if rs else None,
        )
        pcc = _pcc(got, want)
        err = (got - want).abs()
        bound = c["atol"] + c["rtol"] * want.abs()
        n_bad = int((err > bound).sum())
        print(
            f"dev {d}: pcc {pcc:.7f} max abs err {float(err.max()):.4g} max rel {float((err / want.abs().clamp_min(1e-3)).max()):.4g}"
        )
        assert pcc >= c["pcc"], f"dev {d}: pcc {pcc} < {c['pcc']}"
        assert n_bad == 0, f"dev {d}: {n_bad} elements outside atol {c['atol']} + rtol {c['rtol']}"
    _, (tt_x_b, tt_w_b, _, _, extra_b) = _inputs(mesh_device, c, c["seed"] + 1)
    determinism.assert_deterministic(
        lambda: _call(c, tt_x, tt_w, cfg, extra),
        lambda: _call(c, tt_x_b, tt_w_b, cfg, extra_b),
        first=first,
        label=c["id"],
    )


def _call(c, tt_x, tt_w, cfg, extra):
    return ttnn.bringup.rms_norm(
        tt_x,
        epsilon=c["epsilon"],
        weight=tt_w,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=cfg,
        **extra,
    )


def _inputs(mesh_device, c, seed):
    """Host (x, w, residual) for `seed` and the device (x, w, residual, kernel config, residual kwargs)."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(seed)
    xs, ws = c["input"], c["weight"]

    # Host values in the captured dtype: bf16-rounded for a BFLOAT16 tensor, full fp32 for a FLOAT32 one (the
    # reference then sees exactly what the device holds). BFLOAT16 cases keep their original inputs.
    x = torch.randn([n_dev, *xs["shape"][1:]], generator=g).to(_host_dtype(xs))  # device d gets x[d:d+1]
    w = (1.0 + 0.5 * torch.randn(ws["shape"], generator=g)).to(_host_dtype(ws))
    rs = c.get("residual")  # optional residual_input_tensor, drawn after w so residual-free cases keep their inputs
    r = torch.randn([n_dev, *rs["shape"][1:]], generator=g).to(_host_dtype(rs)) if rs else None

    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        device=mesh_device,
        dtype=getattr(ttnn.DataType, xs["dtype"]),
        layout=getattr(ttnn.Layout, xs["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = ttnn.from_torch(
        w,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
        dtype=getattr(ttnn.DataType, ws["dtype"]),
        layout=getattr(ttnn.Layout, ws["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ck = c["compute_kernel_config"]
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
        dst_full_sync_en=ck["dst_full_sync_en"],
    )

    extra = {}
    if rs:
        tt_r = ttnn.from_torch(
            r,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            device=mesh_device,
            dtype=getattr(ttnn.DataType, rs["dtype"]),
            layout=getattr(ttnn.Layout, rs["layout"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        extra["residual_input_tensor"] = tt_r
        if c.get("return_residual_sum"):
            extra["return_residual_sum"] = True
            extra["residual_sum_memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    return (x, w, r), (tt_x, tt_w, tt_r if rs else None, cfg, extra)


# ---------------------------------------------------------------------------------------------------------------- DS41
_NOISE = [c for c in CASES if "noise" in c]
_DT = {"FLOAT32": (ttnn.float32, torch.float32), "BFLOAT16": (ttnn.bfloat16, torch.bfloat16)}


def _ds41_inputs(mesh_device, c, seed):
    """Host x [n_dev, 1, R, W] (device d its row block d), the bf16-valued fp32 weight, the other TP chip's columns
    (partial_stats: the row's statistic covers stats_width columns, this chip holds W of them) and the device tensors.
    Rows scaled by U(0.1, 3.1), so the per-row statistic matters."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(seed)
    _, _, R, W = c["input"]["shape"]
    x = torch.randn(n_dev, 1, R, W, generator=g) * (torch.rand(n_dev, 1, R, 1, generator=g) * 3 + 0.1)
    x = x.to(_DT[c["input"]["dtype"]][1])
    put = lambda t, dt, mapper: ttnn.from_torch(  # noqa: E731
        t.contiguous(),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    dev = {"x": put(x, _DT[c["input"]["dtype"]][0], shard)}
    w = other = None
    if c["weight"] is not None:
        w = (torch.rand(c["weight"]["shape"], generator=g) + 0.5).bfloat16().float()
        dev["weight"] = put(w, ttnn.float32, ttnn.ReplicateTensorToMesh(mesh_device))
    if "partial_stats" in c:
        # the row's other stats_width - W columns (the other TP chip's), and the gathered partials both chips'
        # stats_out would give: [own partials | other partials], every element a partial sum of squares (B = 2 per
        # chip: 64 partials of W / 64 columns each)
        other = torch.randn(n_dev, 1, R, c["stats_width"] - W, generator=g).to(x.dtype) * x.float().std()
        nb = c["partial_stats"]["shape"][-1]
        cols_all = torch.cat([x.double(), other.double()], -1)
        ps = cols_all.square().reshape(n_dev, 1, R, nb, -1).sum(-1).float()
        dev["partial_stats"] = put(ps, ttnn.float32, shard)
    return x, w, other, dev


def _ds41_call(c, dev, cfg):
    kw = dict(compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if c.get("stats_out"):
        return ttnn.bringup.rms_norm(dev["x"], stats_out=True, **kw)
    kw["epsilon"] = c["epsilon"]
    if "weight" in dev:
        kw["weight"] = dev["weight"]
    if "dtype" in c:
        kw["dtype"] = _DT[c["dtype"]][0]
    if "partial_stats" in c:
        kw["partial_stats"] = dev["partial_stats"]
        kw["stats_width"] = c["stats_width"]
    if "input_window" in c:
        kw["input_window"] = tuple(c["input_window"])
    return ttnn.bringup.rms_norm(dev["x"], **kw)


def _ds41_ref(c, x, w, other):
    """(fn, its inputs): the float64 semantics of the case's call on one chip's host tensors."""
    if c.get("stats_out"):
        return (lambda xx: xx.square().sum(-1)), [x]
    if "input_window" in c:
        c0, c1 = c["input_window"]
        x = x[..., c0:c1]
    if "partial_stats" in c:
        width = c["stats_width"]

        def fn(xx, oo, ww):
            ss = xx.square().sum(-1, keepdim=True) + oo.square().sum(-1, keepdim=True)
            return xx * torch.rsqrt(ss / width + c["epsilon"]) * ww.reshape(-1)

        return fn, [x, other, w]

    def fn(xx, ww):
        return xx * torch.rsqrt(xx.square().mean(-1, keepdim=True) + c["epsilon"]) * ww.reshape(-1)

    return fn, [x, w]


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in _NOISE],
    ids=[c["id"] for c in _NOISE],
    indirect=["mesh_device", "device_params"],
)
def test_rms_norm_ttnn_ds41(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    ck = c["compute_kernel_config"]
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
        dst_full_sync_en=ck["dst_full_sync_en"],
    )
    x, w, other, dev = _ds41_inputs(mesh_device, c, c["seed"])
    out = _ds41_call(c, dev, cfg)
    want_dtype = ttnn.float32 if c.get("stats_out") else _DT[c.get("dtype", c["input"]["dtype"])][0]
    assert out.dtype == want_dtype, (out.dtype, want_dtype)
    outs = [ttnn.to_torch(t).double() for t in ttnn.get_device_tensors(out)]
    for d in range(rows * cols):
        fn, ins = _ds41_ref(c, x[d : d + 1], w, None if other is None else other[d : d + 1])
        want = fn(*[t.double() for t in ins])
        got = outs[d]
        if c.get("stats_out"):
            assert got.shape[-2] == want.shape[-1] and got.shape[-1] % 32 == 0, got.shape
            got = got.sum(-1).reshape(want.shape)  # every element a partial: the row's sum of squares
        got = got.reshape(want.shape)
        lim = noise_limit(fn, ins, c["noise"]["inputs"], c["noise"]["out"])
        a, b = (got - got.mean()).flatten(), (want - want.mean()).flatten()
        pcc = float(a @ b / (a.norm() * b.norm()))
        rel = float((got - want).norm() / want.norm())
        print(
            f"{c['id']} dev {d}: rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e}) "
            f"pcc {pcc:.8f} (min {lim['pcc']:.8f})"
        )
        assert rel <= lim["rel"], f"dev {d}: rel L2 {rel:.3e} > limit {lim['rel']:.3e} (pcc {pcc:.8f}, a diagnostic)"
    dev_b = _ds41_inputs(mesh_device, c, c["seed"] + 1)[3]
    determinism.assert_deterministic(
        lambda: _ds41_call(c, dev, cfg), lambda: _ds41_call(c, dev_b, cfg), first=out, label=c["id"]
    )

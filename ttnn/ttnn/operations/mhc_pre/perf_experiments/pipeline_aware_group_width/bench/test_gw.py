# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""E1 bench: bf16-X group-width selection for mhc_pre (host-only; the REAL op + kernels, make_plan monkeypatched).

Env:
  GW_SHAPES    "TxC,..." (X = (1, 1, T, 4*C))
  GW_VARIANTS  "default,cand,w11,w5,w5d3,..."  (wN = force group_w N; wNdD = + X block depth D)
  GW_WDTYPE    float32 (default) | bfloat16
  GW_REPEAT    calls per (shape, variant) (default 1)
  GW_CHECK     1 = check every call vs the golden reference + golden tolerances (eval/golden_tests/mhc_pre)
Prints: PERF <TxC> <wdtype> <variant> w=<group_w> gx=<groups_x> B=<blocks> d=<depth> kmax=<..> pipe=<0/1> <median>us | samples
"""

import dataclasses
import importlib
import os
import sys

import pytest
import ttnn

torch = importlib.import_module("torch")  # bench-only
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gw_plan  # noqa: E402

import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd  # noqa: E402
from ttnn.operations.mhc_pre import mhc_pre  # noqa: E402
from eval.golden_tests.mhc_pre import helpers as H  # noqa: E402

_KEY = "DEVICE KERNEL DURATION [ns]"
_W_DT = getattr(ttnn, os.environ.get("GW_WDTYPE", "float32"))
_REPEAT = int(os.environ.get("GW_REPEAT", 1))
_CHECK = os.environ.get("GW_CHECK", "0") == "1"
_SHAPES = [tuple(int(v) for v in e.split("x")) for e in os.environ.get("GW_SHAPES", "640x7168").split(",") if e]
_VARIANTS = [v for v in os.environ.get("GW_VARIANTS", "default,cand").split(",") if v]


def _selector(v):
    if v == "default":
        return gw_plan.sel_default, None
    if v == "cand":
        return gw_plan.sel_candidate, None
    assert v.startswith("w")
    w, _, d = v[1:].partition("d")
    return gw_plan.sel_forced(int(w), int(d) if d else None), None


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


@pytest.mark.skipif(os.environ.get("GW_SKIP_PERF") == "1", reason="GW_SKIP_PERF")
def test_gw(device, monkeypatch):
    prof = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    rows, fails = [], []
    n = 4
    for T, C in _SHAPES:
        x_shape, w_shape = (1, 1, T, n * C), (n * C, n * (n + 2))
        x, w, b, scale = H.make_inputs(x_shape, w_shape, dtype=ttnn.bfloat16, weight_dtype=_W_DT, seed=0)
        ref = H.pytorch_mhc_pre(x, w, b, scale=scale) if _CHECK else None
        tx = H.create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tw = H.create_ttnn_input_tensor(w, device, dtype=_W_DT, layout=ttnn.TILE_LAYOUT)
        tb = H.create_ttnn_input_tensor(b, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        for v in _VARIANTS:
            sel, depths = _selector(v)
            mp = gw_plan.make_plan_with(sel, depths)
            with monkeypatch.context() as m:
                m.setattr(pd, "make_plan", mp)
                if prof:
                    _read_ns(device)
                ns = []
                for _ in range(_REPEAT):
                    y, post, comb = mhc_pre(tx, tw, tb, scale=scale, compute_kernel_config=H.make_compute_config())
                    ttnn.synchronize_device(device)
                    if prof:
                        ns += _read_ns(device)
            f = mp.last
            info = (
                f"w={f['group_w']} h={f['group_h']} gx={f['groups_x']} B={f['blocks']} d={f['depth']} "
                f"kmax={f['kmax']} pipe={int(gw_plan.pipelined(f))}"
            )
            if _CHECK:
                y_ref, post_ref, comb_ref = ref
                try:
                    H.check_output(
                        y,
                        y_ref.to(torch.bfloat16),
                        shape=[1, 1, T, C],
                        dtype=ttnn.bfloat16,
                        expected_layout=ttnn.TILE_LAYOUT,
                        tolerance=H.TOLERANCES[("y", ttnn.bfloat16)],
                    )
                    for got, r, k in ((post, post_ref, n), (comb, comb_ref, n * n)):
                        H.check_output(
                            got,
                            r,
                            shape=[1, 1, T, k],
                            dtype=ttnn.float32,
                            expected_layout=ttnn.TILE_LAYOUT,
                            tolerance=H.TOLERANCES[("coeff", ttnn.bfloat16)],
                        )
                    H.check_doubly_stochastic(comb, comb_ref, n=n)
                    print("CORRECT", f"{T}x{C}", v, info, "ok")
                except Exception as e:  # noqa: BLE001
                    print("CORRECT", f"{T}x{C}", v, info, "FAIL", str(e)[:300])
                    fails.append((T, C, v))
            rows.append((f"{T}x{C}", v, info, ns))
            print("ROW", f"{T}x{C}", v, info, " ".join(f"{s / 1000:.1f}" for s in ns), flush=True)
    for s, v, info, ns in rows:
        med = sorted(ns)[len(ns) // 2] / 1000 if ns else float("nan")
        print("PERF", s, _W_DT, v, info, f"{med:.1f}us |", " ".join(f"{e / 1000:.1f}" for e in ns))
    assert not fails, fails


def test_grad_matches_bench(device):
    """The graduated make_plan (grad/) picks exactly the bench candidate's plan on every swept cell, both W dtypes."""
    from grad_loader import grad_pd

    shapes = [
        (t, c) for t in (256, 333, 512, 640, 1024, 1280, 2048, 2560, 4096) for c in (1792, 2560, 4096, 5120, 6144, 7168)
    ]
    n, bad = 4, []
    for wdt in (ttnn.float32, ttnn.bfloat16):
        for T, C in shapes:
            tx = ttnn.from_torch(
                torch.zeros((1, 1, T, n * C)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            tw = ttnn.from_torch(torch.zeros((n * C, 24)), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
            a = grad_pd.make_plan(device, tx, tw, n)
            b = gw_plan.make_plan_with(gw_plan.sel_candidate)(device, tx, tw, n)
            if dataclasses.asdict(a) != dataclasses.asdict(b):  # distinct Plan classes (two modules)
                bad.append((wdt, T, C, a.group_w, b.group_w))
            tx.deallocate()
            tw.deallocate()
    print("GRAD_MATCH", "ok" if not bad else bad)
    assert not bad, bad

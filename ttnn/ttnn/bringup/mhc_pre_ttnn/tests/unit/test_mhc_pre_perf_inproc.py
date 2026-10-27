# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process device-ns probe for mhc_pre (Refinement 4): the knob A/B of test_mhc_pre_perf_sweep with numbers.
Reads each call's device time from the real-time profiler (on in every run: no env var, no Tracy capture) and prints
the per-setting median; perf is never asserted. Selection by env: MHC_PRE_PERF_KNOBS (see the sweep),
MHC_PRE_PERF_DTYPES, MHC_PRE_PERF_SHAPES / _XSHAPES, MHC_PRE_PERF_WDTYPE, MHC_PRE_PERF_REPEAT, MHC_PRE_SINKHORN_ITERS.
Run:  scripts/run_safe_pytest.sh --run-all -s <this file>
"""

import os

import torch
import ttnn

import ttnn.bringup.mhc_pre_ttnn.mhc_pre_program_descriptor as pd
from ttnn.bringup.mhc_pre_ttnn import mhc_pre
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from .test_mhc_pre_perf_sweep import SHAPES, KNOBS

_KDIR = "/mhc_pre_ttnn/"  # the op's programs, by kernel path
_DTYPES = {"xbf16": ttnn.bfloat16, "xf32": ttnn.float32}
_SEL_DT = os.environ.get("MHC_PRE_PERF_DTYPES", "xbf16,xf32").split(",")
_SEL_SH = os.environ.get("MHC_PRE_PERF_SHAPES")
_W_DT = getattr(ttnn, os.environ.get("MHC_PRE_PERF_WDTYPE", "float32"))
_REPEAT = int(os.environ.get("MHC_PRE_PERF_REPEAT", 1))  # calls per setting (the median is printed too)
_SHAPES = [s for s in SHAPES if _SEL_SH is None or f"{s[-2]}x{s[-1] // 4}" in _SEL_SH.split(",")]
# extra "TxC" shapes (1, 1, T, 4*C), e.g. MHC_PRE_PERF_XSHAPES=1x7168,32x128
_SHAPES += [
    (1, 1, int(t), 4 * int(c))
    for t, c in (e.split("x") for e in os.environ.get("MHC_PRE_PERF_XSHAPES", "").split(",") if e)
]


def _device_ns(device, fn, n):
    """Device ns of each of n calls of fn (one mhc_pre program a call; max over chips)."""
    _, per_program = profile_realtime_program_merged(device, lambda: [fn() for _ in range(n)])
    ns = [e["duration_ns"] for e in per_program.values() if any(_KDIR in s for s in e["kernel_sources"])]
    assert len(ns) == n, f"expected {n} mhc_pre programs in the profiler window, got {len(ns)}"
    return ns


def test_mhc_pre_perf_inproc(device, monkeypatch):
    require_realtime_profiler("the mhc_pre device-ns probe")
    rows = []
    for x_shape in _SHAPES:
        for dt_name in _SEL_DT:
            for knob, patch in KNOBS.items():
                with monkeypatch.context() as m:
                    for name, value in patch.items():
                        m.setattr(pd, name, value)
                    torch.manual_seed(0)
                    nc = x_shape[-1]
                    x = torch.randn(x_shape, dtype=torch.float32)
                    w = torch.randn((nc, 24), dtype=torch.float32) / nc**0.5
                    b = torch.randn((1, 24), dtype=torch.float32)
                    tx = ttnn.from_torch(x, dtype=_DTYPES[dt_name], layout=ttnn.TILE_LAYOUT, device=device)
                    tw = ttnn.from_torch(w, dtype=_W_DT, layout=ttnn.TILE_LAYOUT, device=device)
                    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                    outs = []

                    def call():
                        outs.append(
                            mhc_pre(
                                tx,
                                tw,
                                tb,
                                scale=(1.0, 1.0, 1.0),
                                sinkhorn_iters=int(os.environ.get("MHC_PRE_SINKHORN_ITERS", 20)),
                            )
                        )

                    ns = _device_ns(device, call, _REPEAT)
                    y, post, comb = outs[-1]
                    if "ABLATE" not in os.environ.get("MHC_PRE_KERNEL_DEFINES", ""):  # ablated outputs are garbage
                        for t in (y, post, comb):
                            assert torch.isfinite(ttnn.to_torch(t)).all()
                rows.append((f"{x_shape[-2]}x{x_shape[-1] // 4}", dt_name, knob, ns))
    for r in rows:
        med = sorted(r[3])[len(r[3]) // 2] / 1000
        print("PERF", *r[:3], f"median {med:.1f}us |", " ".join(f"{v / 1000:.1f}" for v in r[3]))

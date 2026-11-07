# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""mix_collapse_fuse bake-off: the op forked into ./fork with the collapse's y = sum_j P_j X'_j folded
into the mix window (MCF_VARIANT, see fork/kernels/mhc_post_compute.cpp):
  0 = baseline (identical to the real op), 1 = comb_3j packed (pass3 Mac form) only,
  2 = 1 + y folded into the mix (running fp32 y per c-tile in cb_y_run, X' rounded on the SFPU).
Each test runs the REAL op once (reference) and the fork once, asserts every output bit-identical,
and logs the call order to $MCF_LOG so the --profile ops CSV rows can be labelled (label_csv.py).

Run:  true --profile --run-all <this file> -k "<filter>"
"""
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

torch = importlib.import_module("torch")
import ttnn

from ttnn.operations.mhc_post import mhc_post as mhc_post_real

HERE = Path(__file__).parent
N = 4


def _load_fork():
    name = "mhc_post_mix_collapse_fuse_fork"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / "fork" / "__init__.py", submodule_search_locations=[str(HERE / "fork")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _log(label):
    path = os.environ.get("MCF_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(label) + "\n")


CASES = {
    # id: (mode, T, C, fidelity) -- bf16 F / X, packed mixes, as test_mhc_post_perf.test_perf_bf16_packed
    "focus_collapse_T160": ("collapse", 160, 5120, "HiFi2"),
    "collapse_T640": ("collapse", 640, 5120, "HiFi2"),
    "collapse_T4096_C1792": ("collapse", 4096, 1792, "HiFi2"),
    "collapse_T160_HiFi4": ("collapse", 160, 5120, "HiFi4"),
    "stats_T160": ("stats", 160, 5120, "HiFi2"),
    "plain_T160": ("plain", 160, 5120, "HiFi2"),
}


def _run(fn, device, mode, T, C, fid):
    torch.manual_seed(0)
    dev = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mixes = dev(torch.rand(1, 1, T, N * (N + 2)), ttnn.float32)
    kw = dict(post_col=N, comb_col=2 * N)
    if mode in ("stats", "collapse"):
        kw["stats_weight"] = dev(torch.randn(1, 1, N * C, 32) * 0.01, ttnn.float32)
    if mode == "collapse":
        ng = N * (N + 2)
        kw.update(collapse_pre=mixes, hc_scale=[0.3, 0.2, 0.4], hc_base=[0.1 * (k % 5) for k in range(ng)])
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    f = dev(torch.randn(1, 1, T, C), ttnn.bfloat16)
    x = dev(torch.randn(1, 1, T, N * C), ttnn.bfloat16)
    out = fn(f, x, mixes, mixes, compute_kernel_config=cfg, **kw)
    ttnn.synchronize_device(device)
    outs = out if isinstance(out, tuple) else (out,)
    return [ttnn.to_torch(o) for o in outs if o is not None]


SEL = os.environ.get("MCF_CASES", "")
CASE_IDS = [c for c in CASES if not SEL or c in SEL.split(",")]


VARIANTS = [int(v) for v in os.environ.get("MCF_VARIANTS", "0,1,2").split(",")]
ORDER = os.environ.get("MCF_ORDER", "real_first")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASE_IDS)
def test_mix_collapse_fuse(case, variant, device):
    fork = _load_fork()
    fork.mhc_post_program_descriptor.MCF_VARIANT = variant
    device.clear_program_cache()
    runs = [("real", mhc_post_real), (f"v{variant}", fork.mhc_post)]
    if ORDER == "fork_first":
        runs.reverse()
    res = {}
    for who, fn in runs:
        _log({"case": case, "who": who})
        res[who] = _run(fn, device, *CASES[case])
    ref, got = res["real"], res[f"v{variant}"]
    assert len(ref) == len(got)
    for k, (a, b) in enumerate(zip(ref, got)):
        if not torch.equal(a, b):
            d = (a.float() - b.float()).abs()
            nz = int((d > 0).sum())
            rel = float(d.norm() / a.float().norm())
            pytest.fail(
                f"{case} v{variant}: output {k} differs: {nz}/{a.numel()} elems, max {d.max():.3e}, rel-L2 {rel:.3e}"
            )

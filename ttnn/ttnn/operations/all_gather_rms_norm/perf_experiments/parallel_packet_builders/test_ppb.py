# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""parallel_packet_builders bench entry.

  PPB_VARIANTS     comma list: base, ppbH[_dD][_pP] (default base,ppb4)
  PPB_SIZES        comma list of R_tiles:W_local (default 20:384)
  PPB_PERF         1 (default): eval/ccl_perf protocol after the correctness check; 0: correctness only
  PPB_EXPERIMENTS  ccl_perf experiments (default 5)
  PPB_OUT          result dir (default .eval/ppb)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from eval import ccl_perf

import ttnn  # noqa: E402

# ttnn.operations auto-imports every module under it while ttnn itself initializes (any process importing ttnn):
# touch the environment only when pytest runs this file (ttnn complete by then), never on that auto-import.
_ACTIVE = hasattr(ttnn, "Topology") and "pytest" in sys.modules
PERF = os.environ.get("PPB_PERF", "1") == "1"
ZONES = os.environ.get("PPB_ZONES") == "1"  # zone profile: CSV under TT_METAL_PROFILER_DIR (perf off)
if ZONES:
    PERF = False
    if _ACTIVE:
        os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
if PERF and _ACTIVE:
    for _k, _v in ccl_perf.PROFILER_ENV.items():
        os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from ttnn.operations.all_gather_rms_norm.perf_experiments.parallel_packet_builders import bench  # noqa: E402

VARIANTS = os.environ.get("PPB_VARIANTS", "base,ppb4").split(",")
SIZES = [tuple(int(v) for v in s.split(":")) for s in os.environ.get("PPB_SIZES", "20:384").split(",")]
EXPER = int(os.environ.get("PPB_EXPERIMENTS", "5"))
OUT = Path(os.environ.get("PPB_OUT", ".eval/ppb"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("size", SIZES, ids=[f"R{r}_W{w}" for r, w in SIZES])
@pytest.mark.parametrize("variant", VARIANTS)
def test_ppb(variant, size, _fabric_mesh_cache):
    mesh = _fabric_mesh_cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE if PERF else 0)
    R, Wl = size
    call, out, full, plan = bench.make_call(mesh, variant, R, Wl)
    call()
    ttnn.synchronize_device(mesh)
    bad = bench.check_output(mesh, out, full)
    print(f"PPB check {variant} R{R} W{Wl}: {'OK' if not bad else bad}")
    assert not bad, f"landing mismatch (chip, wrong elements per slot): {bad}"
    for _ in range(3):  # repeat calls: counters re-armed, still correct
        call()
    ttnn.synchronize_device(mesh)
    assert not bench.check_output(mesh, out, full)
    if ZONES:
        for _ in range(6):
            call()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
    if PERF:
        doc = ccl_perf.run(
            mesh, call, experiments=EXPER, out_path=OUT / f"{variant}_R{R}_W{Wl}.json", label=f"{variant} R{R} W{Wl}"
        )
        s = doc["summary"]
        print(
            f"PPB perf {variant} R{R} W{Wl}: fastest {s['fastest']['mean']:.3f} slowest {s['slowest']['mean']:.3f} us"
        )
    assert not bench.check_output(mesh, out, full)

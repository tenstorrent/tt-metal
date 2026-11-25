# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op check of the transport candidate: a COPY of the q_stem package (q_stem_base = the op as on disk,
q_stem_bal = balanced antipodal split) on a loose case, golden-checked twice, then the ccl_perf protocol.

    QSTEM_PKG=q_stem_bal QSTEM_CASES=0 CCL_PERF_EXPERIMENTS=5 scripts/run_safe_pytest.sh --no-precompile \
        --rootdir=<this dir> <this dir>/op_bench.py -s
QSTEM_MODE=check: golden only. QSTEM_BALANCE_ANTIPODAL=0 turns the split off in q_stem_bal.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

from eval import ccl_perf

MODE = os.environ.get("QSTEM_MODE", "perf")
if MODE == "perf":
    for _k, _v in ccl_perf.PROFILER_ENV.items():
        os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.q_stem.conftest import _fabric_mesh_cache, focus_mesh_or_skip  # noqa: E402,F401
from eval.golden_tests.q_stem.helpers import prepare_q_stem  # noqa: E402
from eval.golden_tests.q_stem.test_golden import _loose_cases  # noqa: E402

HERE = Path(__file__).parent
PKG = os.environ.get("QSTEM_PKG", "q_stem_bal")
_spec = importlib.util.spec_from_file_location(
    PKG, HERE / PKG / "__init__.py", submodule_search_locations=[str(HERE / PKG)]
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[PKG] = _mod
_spec.loader.exec_module(_mod)
q_stem = _mod.q_stem
_pd = importlib.import_module(f"{PKG}.q_stem_program_descriptor")
assert Path(_pd.__file__).parent == HERE / PKG

for _kv in filter(None, os.environ.get("QSTEM_KNOBS", "").split(";")):  # e.g. HDRS_PER_DEST_CAP=1
    _k, _v = _kv.split("=", 1)
    assert hasattr(_pd, _k), _k
    setattr(_pd, _k, type(getattr(_pd, _k))(eval(_v)))

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", "5"))
OUT = Path(os.environ.get("CCL_PERF_OUT", str(HERE / "results_op")))
CASES = [int(i) for i in os.environ.get("QSTEM_CASES", "0").split(",")]
TAG = os.environ.get("QSTEM_TAG", "")


@pytest.fixture
def bmesh(request, _fabric_mesh_cache):  # noqa: F811
    extras = request.node.callspec.params["extras"]
    kw = dict(payload_bytes=extras.get("fabric_payload_bytes"))
    if MODE == "perf":
        kw["trace_region_size"] = ccl_perf.TRACE_REGION_SIZE
    return _fabric_mesh_cache.get(focus_mesh_or_skip(), **kw)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras", [_loose_cases()[i] for i in CASES])
def test_op_bench(inputs, axes, extras, bmesh, request):
    call, check = prepare_q_stem(inputs, mesh_device=bmesh, extras=extras, op=q_stem, **axes)
    check(call())
    check(call())  # second call: re-armed counters / reused scratch
    print(f"OP_BENCH {PKG} case {request.node.callspec.id[:60]}: golden check passed x2", flush=True)
    if MODE == "check":
        return
    if MODE == "zone":  # run with TT_METAL_DEVICE_PROFILER=1; then zone_stages.py on profile_log_device.csv
        import ttnn

        for _ in range(3):
            call()
        ttnn.synchronize_device(bmesh)
        ttnn.ReadDeviceProfiler(bmesh)
        return
    cid = CASES[0]
    doc = ccl_perf.run(
        bmesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"case{cid}_{PKG}{TAG}.json",
        label=f"{PKG}{TAG} case{cid}",
        meta={"op": "q_stem", "pkg": PKG, "experiments": EXPERIMENTS},
    )
    s = doc["summary"]
    print(
        f"OP_BENCH_RESULT {PKG}{TAG} case{cid} fastest {s['fastest']['mean']:.2f} slowest {s['slowest']['mean']:.2f} "
        f"per_chip { {k: round(v['mean'], 2) for k, v in s['per_chip'].items()} }",
        flush=True,
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""s3_interleave bake-off harness: the q_stem PERF FOCUS (or another loose case) on a COPY of the op.

    QSTEM_PKG=q_stem_v|q_stem_base  QSTEM_KNOBS="K=V;K=V"  CCL_PERF_EXPERIMENTS=5  CCL_PERF_OUT=<dir>
    QSTEM_CASES=0[,1,2]  QSTEM_MODE=perf|zone|check
    scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/operations/q_stem/perf_experiments/s3_interleave/bench.py -s

perf : golden check once, then the ccl_perf protocol (fastest chip per call) -> <OUT>/<pkg>__<knobs>.json
zone : golden check, 3 calls, ReadDeviceProfiler (run with TT_METAL_DEVICE_PROFILER=1; then zone_stages.py)
check: golden check only (each selected case)
"""

from __future__ import annotations

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
import ttnn  # noqa: E402

from eval.golden_tests.q_stem.conftest import _fabric_mesh_cache, focus_mesh_or_skip  # noqa: E402,F401
from eval.golden_tests.q_stem.helpers import prepare_q_stem  # noqa: E402
from eval.golden_tests.q_stem.test_golden import _loose_cases  # noqa: E402

HERE = Path(__file__).parent
PKG = os.environ.get("QSTEM_PKG", "q_stem_v")
_spec = importlib.util.spec_from_file_location(
    PKG, HERE / PKG / "__init__.py", submodule_search_locations=[str(HERE / PKG)]
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[PKG] = _mod
_spec.loader.exec_module(_mod)
q_stem = _mod.q_stem
_pd = importlib.import_module(f"{PKG}.q_stem_program_descriptor")
assert Path(_pd.__file__).parent == HERE / PKG

KNOBS = os.environ.get("QSTEM_KNOBS", "")
for _kv in filter(None, KNOBS.split(";")):
    _k, _v = _kv.split("=", 1)
    assert hasattr(_pd, _k), _k
    setattr(_pd, _k, type(getattr(_pd, _k))(eval(_v)))

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", "5"))
OUT = Path(os.environ.get("CCL_PERF_OUT", str(HERE / "results")))
CASES = [int(i) for i in os.environ.get("QSTEM_CASES", "0").split(",")]


@pytest.fixture
def bmesh(request, _fabric_mesh_cache):  # noqa: F811
    extras = request.node.callspec.params["extras"]
    kw = dict(payload_bytes=extras.get("fabric_payload_bytes"))
    if MODE == "perf":
        kw["trace_region_size"] = ccl_perf.TRACE_REGION_SIZE
    return _fabric_mesh_cache.get(focus_mesh_or_skip(), **kw)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras", [_loose_cases()[i] for i in CASES])
def test_bench(inputs, axes, extras, bmesh):
    call, check = prepare_q_stem(inputs, mesh_device=bmesh, extras=extras, op=q_stem, **axes)
    if not os.environ.get("QSTEM_ABLATE"):
        check(call())
    else:
        call()
    if not os.environ.get("QSTEM_ABLATE"):
        check(call())  # second call: re-armed counters / reused scratch
    if MODE == "check":
        return
    if MODE == "zone":
        for _ in range(3):
            call()
        ttnn.synchronize_device(bmesh)
        ttnn.ReadDeviceProfiler(bmesh)
        return
    tag = KNOBS.replace(";", "_").replace("=", "").replace(" ", "").replace(",", "-").replace("(", "").replace(")", "")
    ccl_perf.run(
        bmesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"{os.environ.get('QSTEM_ABLATE', '')}{PKG}{'__' + tag if tag else ''}.json",
        label=f"{PKG} {KNOBS or 'default'}",
        meta={"op": "q_stem", "pkg": PKG, "knobs": KNOBS, "experiments": EXPERIMENTS},
    )

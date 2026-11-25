# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""x_supply bake-off harness: a loose case of q_stem on a COPY of the op (q_stem_base = the op as is, q_stem_v =
the candidate).

    QSTEM_PKG=q_stem_v|q_stem_base  QSTEM_KNOBS="K=V;K=V"  QSTEM_ABLATE=<STAGE,...>  CCL_PERF_EXPERIMENTS=5
    CCL_PERF_OUT=<dir>  QSTEM_CASES=0[,1]  QSTEM_MODE=perf|zone|check
    scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/operations/q_stem/perf_experiments/x_supply/op_bench.py -s

perf : golden check (twice), then the ccl_perf protocol -> <OUT>/<pkg>__<knobs>.json
zone : golden check, 3 calls, ReadDeviceProfiler (TT_METAL_DEVICE_PROFILER=1; then zone_stages.py)
check: golden check only
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

import eval.golden_tests.q_stem.helpers as _qh  # noqa: E402

_orig_check_one = _qh._check_one


def _check_one_print(*a, **kw):  # print every device's PCC / rel RMS vs golden (the check itself is unchanged)
    m = _orig_check_one(*a, **kw)
    print(f"XS_OP_METRICS pcc={m.pcc:.7f} rms={m.rms:.4e} tol={kw.get('tolerance')}")
    return m


_qh._check_one = _check_one_print

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
ABLATE = os.environ.get("QSTEM_ABLATE", "")

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
    if not ABLATE:
        check(call())
        check(call())  # second call: re-armed counters / reused scratch
    else:
        call()
    if MODE == "check":
        return
    if MODE == "zone":
        for _ in range(3):
            call()
        ttnn.synchronize_device(bmesh)
        ttnn.ReadDeviceProfiler(bmesh)
        return
    tag = KNOBS.replace(";", "_").replace("=", "").replace(" ", "").replace(",", "-").replace("(", "").replace(")", "")
    abl = ("__abl_" + ABLATE.replace(",", "_")) if ABLATE else ""
    ccl_perf.run(
        bmesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"{PKG}{'__' + tag if tag else ''}{abl}.json",
        label=f"{PKG} {KNOBS or 'default'} {ABLATE}",
        meta={"op": "q_stem", "pkg": PKG, "knobs": KNOBS, "ablate": ABLATE, "experiments": EXPERIMENTS},
    )


# QSTEM_GOLDEN="<regex>;<regex>": correctness of the first SUPPORTED golden cell matching each regex (check only)
def _golden_cells():
    import re

    from eval.golden_tests.q_stem.test_golden import _parametrize_cases

    pats = [p for p in os.environ.get("QSTEM_GOLDEN", "").split(";") if p]
    cells = [c for c in _parametrize_cases() if not c.marks]
    out = []
    for pat in pats:
        hit = next((c for c in cells if re.search(pat, c.id)), None)
        assert hit is not None, pat
        out.append(pytest.param(*hit.values, id=hit.id))
    return out


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes", _golden_cells())
def test_golden_cell(inputs, axes, request, _fabric_mesh_cache):  # noqa: F811
    mesh = _fabric_mesh_cache.get(focus_mesh_or_skip())
    call, check = prepare_q_stem(inputs, mesh_device=mesh, extras={}, op=q_stem, **axes)
    check(call())
    check(call())

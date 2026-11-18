"""eval/golden_tests/all_gather_rms_norm_kimi/perf_measure.py pointed at the experiment's op copy.

    AGRN_NORM_IMPL=<0|1|2> CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile \\
        ttnn/ttnn/operations/all_gather_rms_norm_kimi/perf_experiments/late_chunk_one_pass/perf_measure.py -s
SHAPE env: unset = the PERF FOCUS loose case (golden-checked); "tall" = (1,1,2048,384) (checked against torch)."""

from __future__ import annotations

import os
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.all_gather_rms_norm_kimi.conftest import focus_mesh_or_skip  # noqa: E402
from eval.golden_tests.all_gather_rms_norm_kimi.helpers import prepare_all_gather_rms_norm_kimi  # noqa: E402
from eval.golden_tests.all_gather_rms_norm_kimi.test_golden import _loose_cases  # noqa: E402

from _op import all_gather_rms_norm_kimi  # noqa: E402

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf"))
TALL = os.environ.get("SHAPE") == "tall"


@pytest.fixture
def perf_mesh(_fabric_mesh_cache):
    return _fabric_mesh_cache.get(focus_mesh_or_skip(), trace_region_size=ccl_perf.TRACE_REGION_SIZE)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras", _loose_cases())
def test_perf_measure(inputs, axes, extras, perf_mesh, request):
    case = "tall" if TALL else request.node.callspec.id
    if TALL:
        inputs = [(1, 1, 2048, 384)] + list(inputs[1:])
    call, check = prepare_all_gather_rms_norm_kimi(
        inputs, mesh_device=perf_mesh, extras=extras, op=all_gather_rms_norm_kimi, **axes
    )
    check(call())
    tag = os.environ.get("CCL_PERF_LABEL", f"impl{os.environ.get('AGRN_NORM_IMPL', '0')}")
    ccl_perf.run(
        perf_mesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"{case}.json",
        label=tag,
        meta={"op": "all_gather_rms_norm_kimi[late_chunk_one_pass]", "case": case, "experiments": EXPERIMENTS},
    )

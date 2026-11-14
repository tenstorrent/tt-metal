# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time of mhc_post's PERF FOCUS cases by the eval/ccl_perf.py protocol — the same measurement as
eval/golden_tests/mhc_post/perf_measure.py, but the inputs are uploaded ONCE and `call()` only dispatches the op
(perf_measure.py's prepare_mhc_post re-uploads every input per call, which TT_FATALs inside the trace capture).

Not collected by default (no test_ prefix); run by name:

    CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/mhc_post/perf_feed_measure.py -s

CCL_PERF_CASES: "focus" (default: LOOSE_CASES[0] feed and [1] no feed) or comma-separated LOOSE_CASES indices.
Compare variants with `python -m eval.ccl_perf compare <base.json> <cand.json>`.
"""

from __future__ import annotations

import os
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_harness import parametrize_loose_cases  # noqa: E402
from eval.golden_tests.mhc_post.feature_spec import INVALID, LOOSE_CASES  # noqa: E402
from eval.golden_tests.mhc_post import helpers as H  # noqa: E402
import importlib, sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
_VARIANT = os.environ.get("MHC_CS_VARIANT", "cs")  # cs = candidate copy, base = unchanged copy
_m = importlib.import_module("mhc_post_cs" if _VARIANT == "cs" else "mhc_post_base")
EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, mhc_post = _m.EXCLUSIONS, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.mhc_post

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf_unit"))


def _cases():
    cases = parametrize_loose_cases(
        LOOSE_CASES,
        INPUT_TAGGERS,
        SUPPORTED,
        EXCLUSIONS,
        INVALID,
        shape_id=lambda inp: "X" + "x".join(str(d) for d in inp[1]),
    )
    sel = os.environ.get("CCL_PERF_CASES", "focus")
    if sel == "focus":
        return [c for c in cases if "attention" in c.values[2]]
    return [cases[int(i)] for i in sel.split(",")]


def _prepare(inputs, *, mesh_device, extras, **axes):
    """prepare_mhc_post's call/check, with the dispatch-only call: the op's args are captured from one
    call_mhc_post_mesh (which uploads) and every later call re-dispatches with the same device tensors."""
    captured = {}

    def capture(*a, **kw):
        captured["a"], captured["kw"] = a, kw
        return mhc_post(*a, **kw)

    call0, check = H.prepare_mhc_post(inputs, mesh_device=mesh_device, extras=extras, op=capture, **axes)
    first = call0()  # last capture = the measured call's args (feed kwargs included)
    a, kw = captured["a"], captured["kw"]
    return first, (lambda: mhc_post(*a, **kw)), check


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras", _cases())
def test_perf_feed_measure(inputs, axes, extras, perf_mesh, request):
    case = request.node.callspec.id
    first, call, check = _prepare(inputs, mesh_device=perf_mesh, extras=extras, **axes)
    if not os.environ.get("CCL_PERF_NOCHECK"):  # ablations (stubbed payloads) skip the gates
        check(first)
        check(call())
    meta = {
        "op": "mhc_post",
        "case": case,
        "experiments": EXPERIMENTS,
        "roofline_us": extras.get("roofline_us"),
        "target_us": extras.get("target_us", extras.get("target_ns", 0) / 1000 or None),
    }
    tag = "feed" if axes.get("feed") == "on" else "nofeed"
    ccl_perf.run(
        perf_mesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT / f"{tag}.json",
        label=os.environ.get("CCL_PERF_LABEL", tag),
        meta=meta,
    )


@pytest.fixture(scope="module")
def perf_mesh():
    from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
    from eval.golden_tests.mhc_post.feature_spec import FEED_FABRIC_CONFIG
    from eval.golden_tests.mhc_post.topology import feed_mesh_shape

    shape = feed_mesh_shape()
    if shape is None:
        pytest.skip("no feed mesh on this cluster")
    cache = _FabricMeshCache()
    yield cache.get(shape, FEED_FABRIC_CONFIG, trace_region_size=ccl_perf.TRACE_REGION_SIZE)
    cache.close()

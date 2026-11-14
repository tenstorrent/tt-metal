# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""feed_l1_landing: tests/ttnn/unit_tests/operations/mhc_post/perf_feed_measure.py pointed at MY copies.
MHC_FL_VARIANT=base (unchanged op copy) | l1 (candidate; MHC_FEED_L1=direct|stage|dram picks its mode)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_harness import parametrize_loose_cases  # noqa: E402
from eval.golden_tests.mhc_post.feature_spec import INVALID, LOOSE_CASES  # noqa: E402
from eval.golden_tests.mhc_post import helpers as H  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
_m = importlib.import_module("mhc_post_l1" if os.environ.get("MHC_FL_VARIANT", "l1") == "l1" else "mhc_post_base")
EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, mhc_post = _m.EXCLUSIONS, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.mhc_post

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf_unit"))


def _cases():
    loose = LOOSE_CASES
    ts = os.environ.get("CCL_PERF_DS41_T")  # domain sweep: DS41 collapse + norm + feed at these T (comma list)
    if ts:
        from eval.golden_tests.mhc_post.feature_spec import _ds41_case

        loose = [_ds41_case(int(t), "collapse", "on", "on", f"sweep T{t}") for t in ts.split(",")]
        for c in loose:
            c["extras"]["attention"] = "sweep"
    cases = parametrize_loose_cases(
        loose,
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
    captured = {}

    def capture(*a, **kw):
        captured["a"], captured["kw"] = a, kw
        return mhc_post(*a, **kw)

    call0, check = H.prepare_mhc_post(inputs, mesh_device=mesh_device, extras=extras, op=capture, **axes)
    first = call0()
    a, kw = captured["a"], captured["kw"]
    return first, (lambda: mhc_post(*a, **kw)), check


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras", _cases())
def test_perf_feed_measure(inputs, axes, extras, perf_mesh, request):
    case = request.node.callspec.id
    first, call, check = _prepare(inputs, mesh_device=perf_mesh, extras=extras, **axes)
    if not os.environ.get("CCL_PERF_NOCHECK"):
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

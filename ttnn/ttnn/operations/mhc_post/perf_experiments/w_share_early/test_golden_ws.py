# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The golden suites on the experiment copy (MHC_WS_VARIANT=ws|base): test_golden.py's test_op / test_op_loose
(single device, feed off) and test_feed.py's test_op / test_op_loose (feed on, the 1x4 feed mesh: the 106-worker
grid where the strip placement differs from row-major). WSG_FEED=0|1 selects which half.

    scripts/run_safe_pytest.sh --run-all <this> --import-mode=prepend -o addopts=""
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from eval.golden_harness import parametrize_cases, parametrize_loose_cases  # noqa: E402
from eval.golden_tests.mhc_post import helpers as H  # noqa: E402
from eval.golden_tests.mhc_post.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET  # noqa: E402

_m = importlib.import_module("mhc_post_" + os.environ.get("MHC_WS_VARIANT", "ws"))
H.mhc_post = _m.mhc_post  # helpers' default op
FEED = os.environ.get("WSG_FEED", "0") == "1"


def _sid(inputs):
    return "X" + "x".join(str(d) for d in inputs[1])


def _keep(p):
    """feed half + WSG_FILTER (comma = OR of alternatives, + = AND of substrings of the case id; default all)."""
    if (p.values[1]["feed"] == "on") != FEED:
        return False
    f = os.environ.get("WSG_FILTER", "")
    return not f or any(all(t in p.id for t in alt.split("+")) for alt in f.split(","))


@pytest.fixture(scope="module")
def _mesh():
    if not FEED:
        yield None
        return
    from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
    from eval.golden_tests.mhc_post.feature_spec import FEED_FABRIC_CONFIG
    from eval.golden_tests.mhc_post.topology import feed_mesh_shape

    cache = _FabricMeshCache()
    yield cache.get(feed_mesh_shape(), FEED_FABRIC_CONFIG)
    cache.close()


@pytest.fixture
def dev(request, _mesh):
    return _mesh if FEED else request.getfixturevalue("device")


def _kw(d):
    return {"mesh_device": d} if FEED else {"device": d}


@pytest.mark.parametrize(
    "inputs,axes",
    [
        p
        for p in parametrize_cases(
            TARGET, INPUTS, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.EXCLUSIONS, INVALID, shape_id=_sid
        )
        if _keep(p)
    ],
)
def test_op(inputs, axes, dev):
    H.run_mhc_post(inputs, **_kw(dev), **axes)


@pytest.mark.parametrize(
    "inputs,axes,extras",
    [
        p
        for p in parametrize_loose_cases(
            LOOSE_CASES, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.EXCLUSIONS, INVALID, shape_id=_sid
        )
        if _keep(p)
    ],
)
def test_op_loose(inputs, axes, extras, dev):
    H.run_mhc_post(inputs, **_kw(dev), extras=extras, **axes)

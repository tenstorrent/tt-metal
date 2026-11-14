"""Golden gates of mhc_post against MY copy (MHC_CS_VARIANT=cs|base): the single-device loose cases and the
test_op cartesian cells, both filtered by CS_FILTER (substring of the case id, comma = OR; default all)."""
import importlib, os, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.mhc_post import helpers as H
from eval.golden_tests.mhc_post.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET

_m = importlib.import_module("mhc_post_cs" if os.environ.get("MHC_CS_VARIANT", "cs") == "cs" else "mhc_post_base")
H.mhc_post = _m.mhc_post  # run_mhc_post's single-device path dispatches through helpers.mhc_post


def _sid(inputs):
    return "X" + "x".join(str(d) for d in inputs[1])


def _keep(p):
    if p.values[1]["feed"] == "on":
        return False
    f = os.environ.get("CS_FILTER", "")
    return not f or any(all(t in p.id for t in alt.split("+")) for alt in f.split(","))


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
def test_loose(inputs, axes, extras, device):
    H.run_mhc_post(inputs, device=device, extras=extras, **axes)


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
def test_cells(inputs, axes, device):
    H.run_mhc_post(inputs, device=device, **axes)

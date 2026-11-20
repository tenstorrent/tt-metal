# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Domain sweep: every [perf] focus case and every HEIGHT-sharded perf-file layer (test_conv2d_3x3_perf.py
test_conv2d_3x3_height_sharded_perf: HiFi4, the HEIGHT twins), each run once per variant (KNW_DOM_VARIANTS, default
"base,grad"; KNW_UNZ=1: the zone-stripped twins), in one process. PCC is asserted by the test bodies.
DEVICE KERNEL DURATION rows come back in call order (run_dom.sh joins them with runs/calls_<label>.txt).

Env: KNW_DOM_SET = focus | height | all (default all), KNW_DOM_ORDER=alt (reverse the variant order on every other
shape).
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knw_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as perf  # noqa: E402

HERE = Path(__file__).resolve().parent
VARS = [v for v in os.environ.get("KNW_DOM_VARIANTS", "head,grad").split(",") if v]
SET = os.environ.get("KNW_DOM_SET", "all")
SHAPES = []
if SET in ("focus", "all"):
    SHAPES += [("focus", i, fid) for i, fid in enumerate(perf._FOCUS_IDS)]
if SET in ("height", "all"):
    SHAPES += [("height", i, "hifi4_" + c[0]) for i, c in enumerate(perf.HEIGHT_SHAPES)]
if SET == "height_kwf":
    SHAPES += [
        ("height", i, "hifi4_" + c[0])
        for i, c in enumerate(perf.HEIGHT_SHAPES)
        if c[0] not in ("stem_conv1", "res3_0_conv2")
    ]
if SET == "kwf":
    # The kw-fold cases only: the focus cases of test_knw_ab.ALIAS (by layer prefix), then the HEIGHT perf layers
    # that run kw-fold (stride 1, TILE input: all but stem_conv1 and res3_0_conv2).
    import test_knw_ab as _ab

    for a, prefix in _ab.ALIAS.items():
        i = [k for k, c in enumerate(perf._FOCUS_CASES) if c["extras"]["layer"].startswith(prefix)][0]
        SHAPES.append(("focus", i, a))
    SHAPES += [
        ("height", i, "hifi4_" + c[0])
        for i, c in enumerate(perf.HEIGHT_SHAPES)
        if c[0] not in ("stem_conv1", "res3_0_conv2")
    ]
ONLY = [v for v in os.environ.get("KNW_DOM_ONLY", "").split(";") if v]  # ";": focus ids carry commas
if ONLY:
    SHAPES = [s for s in SHAPES if s[2] in ONLY]
SHAPES = SHAPES * int(os.environ.get("KNW_DOM_REPS", "1"))
PARAMS = []
for n, (kind, i, sid) in enumerate(SHAPES):
    vs = VARS if (os.environ.get("KNW_DOM_ORDER") != "alt" or n % 2 == 0) else list(reversed(VARS))
    PARAMS += [(kind, i, sid, v) for v in vs]
CALLS = HERE / "runs" / os.environ.get("KNW_CALLS", "calls_dom.txt")


@pytest.mark.parametrize("kind,i,sid,variant", PARAMS, ids=[f"{s}-{v}-{n}" for n, (_, _, s, v) in enumerate(PARAMS)])
def test_knw_dom(kind, i, sid, variant, device, monkeypatch):
    import test_knw_ab as ab  # variant names and knob syntax (NAMED, pkg:NAME=VALUE+...)

    device.clear_program_cache()
    pkg, knobs = ab._parse(variant)
    op_mod, pd_mod = copy_modules(pkg)
    for name, val in knobs.items():
        mod = op_mod if name.startswith("@") else pd_mod
        assert hasattr(mod, name.lstrip("@")), name
        monkeypatch.setattr(mod, name.lstrip("@"), val)
    for name in dir(pd_mod):
        m = getattr(pd_mod, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()
    for name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
        monkeypatch.setattr(perf, name, getattr(op_mod, name))
    with open(CALLS, "a") as f:
        f.write(f"{sid} {variant} 0\n")
    if kind == "focus":
        perf.test_conv2d_3x3_perf_focus(perf._FOCUS_CASES[i], device)
    else:
        layer, case, dtype, layout, shard_shape, num_cores = perf.HEIGHT_SHAPES[i]
        perf.test_conv2d_3x3_height_sharded_perf(layer, case, dtype, layout, shard_shape, num_cores, device)

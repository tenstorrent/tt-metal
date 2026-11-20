# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B in one process: the perf focus test body (tests/.../test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus) run
once per (case, variant), the op swapped per call between this experiment's copies (sdm_copy.py: base, copy, grad and
their *_unz twins). A variant may carry descriptor knobs: `copy:NAME=VALUE+NAME=VALUE` (monkeypatched on the copy's
program-descriptor module for that call only).

Env: SDM_CASES (comma list of aliases below), SDM_VARIANTS (comma list, run in that order), SDM_ORDER=alt (reverse the
variant order on every other repetition), SDM_REPS (repetitions, default 1), SDM_UNZ=1 (zone-stripped twins).
Every call's output is compared element-for-element with the first variant of its case (exact.jsonl). PCC vs the torch
reference is asserted by the body. DEVICE KERNEL DURATION rows come back in call order (run.sh joins them with
runs/calls_<label>.txt)."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdm_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as perf  # noqa: E402

HERE = Path(__file__).resolve().parent
ALIAS = {
    "stem1": "stem.conv1",
    "stem2": "stem.conv2",
    "stem3": "stem.conv3",
    "res2": "res2.{0,1,2}.conv2",
    "res30": "res3.0.conv2",
    "res3x": "res3.{1,2,3}.conv2",
    "sdr3f0": "semantic decoder.res3.fuse_conv.0",
    "idr3f0": "instance decoder.res3.fuse_conv.0",
    "sdr3f1": "semantic decoder.res3.fuse_conv.1",
    "sdr2f0": "semantic decoder.res2.fuse_conv.0",
    "idr2f0": "instance decoder.res2.fuse_conv.0",
    "sr2f1": "semantic res2.fuse_conv.1",
    "ir2f1": "instance res2.fuse_conv.1",
    "ch1": "center_head.1",
}
CASES = {}
for a, prefix in ALIAS.items():
    hit = [c for c in perf._FOCUS_CASES if c["extras"]["layer"].startswith(prefix)]
    assert len(hit) == 1, (a, prefix, len(hit))
    CASES[a] = hit[0]
SEL = [c for c in os.environ.get("SDM_CASES", "res3x").split(",") if c]
VARS = [v for v in os.environ.get("SDM_VARIANTS", "base,copy").split(",") if v]
REPS = int(os.environ.get("SDM_REPS", "1"))
PARAMS = []
for r in range(REPS):
    vs = VARS if (os.environ.get("SDM_ORDER") != "alt" or r % 2 == 0) else list(reversed(VARS))
    for c in SEL:
        PARAMS += [(c, v, r) for v in vs]
CALLS = HERE / "runs" / os.environ.get("SDM_CALLS", "calls.txt")
_OUT = {}

# Named variants: knobs of op_copy/ (see that copy's descriptor, SDM_*).
NAMED = {
    "single": "copy:SDM_DUAL=False",
    "dual": "copy:SDM_DUAL=True",
    "noread": 'copy:SDM_DUAL=False+SDM_DEFINES=(("SDM_NOREAD","1"),)',
    "nomcast": 'copy:SDM_DUAL=False+SDM_DEFINES=(("SDM_NOMCAST","1"),)',
    "dual_big": "copy:SDM_DUAL=True+SDM_MIN_RECEIVERS=20",
    "dual_big_a3": 'copy:SDM_DUAL=True+SDM_MIN_RECEIVERS=20+SDM_DEFINES=(("SDM_AHEAD","3"),("SDM_AHEAD_ALL","0"))',
    "dual_big_a2": 'copy:SDM_DUAL=True+SDM_MIN_RECEIVERS=20+SDM_DEFINES=(("SDM_AHEAD","2"),("SDM_AHEAD_ALL","0"))',
    "dual_big_noread": 'copy:SDM_DUAL=True+SDM_MIN_RECEIVERS=20+SDM_DEFINES=(("SDM_NOREAD","1"),)',
    "single_a3": 'copy:SDM_DUAL=False+SDM_DEFINES=(("SDM_AHEAD","3"),("SDM_AHEAD_ALL","0"))',
    "dual_a3": 'copy:SDM_DUAL=True+SDM_DEFINES=(("SDM_AHEAD","3"),("SDM_AHEAD_ALL","0"))',
    "dual_a3_greedy": 'copy:SDM_DUAL=True+SDM_PLACE="greedy"+SDM_DEFINES=(("SDM_AHEAD","3"),("SDM_AHEAD_ALL","0"))',
    "dual_a2": 'copy:SDM_DUAL=True+SDM_DEFINES=(("SDM_AHEAD","2"),("SDM_AHEAD_ALL","0"))',
    "dual_a3all": 'copy:SDM_DUAL=True+SDM_DEFINES=(("SDM_AHEAD","3"),("SDM_AHEAD_ALL","1"))',
    "dual_noread": 'copy:SDM_DUAL=True+SDM_DEFINES=(("SDM_NOREAD","1"),)',
    "dual_nomcast": 'copy:SDM_DUAL=True+SDM_DEFINES=(("SDM_NOMCAST","1"),)',
}
ABLATED = ("noread", "nomcast")  # substring match


def _parse(variant):
    variant = NAMED.get(variant, variant)
    if os.environ.get("SDM_UNZ") == "1":
        pkg, sep, rest = variant.partition(":")
        variant = (pkg if pkg.endswith("_unz") else pkg + "_unz") + sep + rest
    pkg, _, knobs = variant.partition(":")
    kv = {}
    for item in [k for k in knobs.split("+") if k]:
        name, _, val = item.partition("=")
        kv[name] = eval(val)
    return pkg, kv


@pytest.mark.parametrize("case_id,variant,rep", PARAMS, ids=[f"{c}-{v}-{r}" for c, v, r in PARAMS])
def test_sdm_ab(case_id, variant, rep, device, monkeypatch):
    import ttnn

    # Variants can share kernels but differ in runtime-arg layout: never let one hit another's cached program.
    device.clear_program_cache()
    pkg, knobs = _parse(variant)
    op_mod, pd_mod = copy_modules(pkg)
    for name, val in knobs.items():
        assert hasattr(pd_mod, name), name
        monkeypatch.setattr(pd_mod, name, val)
    for name in dir(pd_mod):
        m = getattr(pd_mod, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()
    for name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
        monkeypatch.setattr(perf, name, getattr(op_mod, name))
    ablated = any(a in variant for a in ABLATED)
    if ablated:
        monkeypatch.setattr(perf, "ABLATION", True)
    captured = {}
    real = op_mod.conv2d_3x3

    def capture(*a, **k):
        out = real(*a, **k)
        captured["out"] = ttnn.to_torch(out).float()
        return out

    monkeypatch.setattr(perf, "conv2d_3x3", capture)
    with open(CALLS, "a") as f:
        f.write(f"{case_id} {variant} {rep}\n")
    perf.test_conv2d_3x3_perf_focus(CASES[case_id], device)
    if ablated:
        return  # stubbed variants: no output check
    out = captured["out"]
    first = _OUT.setdefault(case_id, (variant, out))
    diff = int((first[1] != out).sum().item()) if first[1].shape == out.shape else -1
    with open(HERE / "exact.jsonl", "a") as f:
        f.write(
            json.dumps(dict(case=case_id, variant=variant, rep=rep, ref=first[0], differ=diff, numel=out.numel()))
            + "\n"
        )
    assert diff == 0, f"{case_id}: {variant} differs from {first[0]} in {diff} elements"

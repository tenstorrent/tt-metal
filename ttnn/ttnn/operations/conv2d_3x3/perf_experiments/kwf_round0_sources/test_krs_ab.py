# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B in one process: the perf focus test body (tests/.../test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus) run
once per (case, variant), the op swapped per call between this experiment's copies (krs_copy.py: base, copy, grad and
their *_unz twins). A variant may carry descriptor knobs: `copy:NAME=VALUE;NAME=VALUE` (monkeypatched on the copy's
program-descriptor module for that call only; knobs separated by ';'), or be one of the NAMED variants below.

Env: KRS_CASES (comma list of aliases below), KRS_VARIANTS (comma list, run in that order), KRS_ORDER=alt (reverse the
variant order on every other repetition), KRS_REPS (repetitions, default 1), KRS_UNZ=1 (zone-stripped twins).
Every call's output is compared element-for-element with the first variant of its case (exact.jsonl) and must match.
PCC vs the torch reference is asserted by the body. DEVICE KERNEL DURATION rows come back in call order (run.sh joins
them with runs/calls_<label>.txt)."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from krs_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as perf  # noqa: E402

HERE = Path(__file__).resolve().parent
ALIAS = {
    "stem1": "stem.conv1",
    "stem2": "stem.conv2",
    "stem3": "stem.conv3",
    "res2": "res2.{0,1,2}.conv2",
    "res30": "res3.0.conv2",
    "res3x": "res3.{1,2,3}.conv2",
    "res40": "res4.0.conv2",
    "res4x": "res4.{1..5}.conv2",
    "res50": "res5.0.conv2",
    "res51": "res5.1.conv2",
    "res52": "res5.2.conv2",
    "aspp1": "aspp.convs.1",
    "aspp2": "aspp.convs.2",
    "aspp3": "aspp.convs.3",
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
SEL = [c for c in os.environ.get("KRS_CASES", "ir2f1").split(",") if c]
VARS = [v for v in os.environ.get("KRS_VARIANTS", "base,copy").split(",") if v]
REPS = int(os.environ.get("KRS_REPS", "1"))
PARAMS = []
for r in range(REPS):
    vs = VARS if (os.environ.get("KRS_ORDER") != "alt" or r % 2 == 0) else list(reversed(VARS))
    for c in SEL:
        PARAMS += [(c, v, r) for v in vs]
CALLS = HERE / "runs" / os.environ.get("KRS_CALLS", "calls.txt")
_OUT = {}

# Named variants: knobs of op_copy/ (see that copy's descriptor, KRS_*).
NAMED = {
    # Round-0 split: the server's slot rows of every round-0 chunk (cr = rows per chunk, n = receivers).
    "r0q3": "copy:KRS_R0_SERVER_ROWS=lambda cr, n: (3 * cr + 2) // 4",
    "r0m1": "copy:KRS_R0_SERVER_ROWS=lambda cr, n: cr - 1",
    "r0all": "copy:KRS_R0_SERVER_ROWS=lambda cr, n: cr",
    "r0q1": "copy:KRS_R0_SERVER_ROWS=lambda cr, n: cr // 4",
    "f58": "copy:KRS_R0_SERVER_FRAC=7 / 12",
    "f60": "copy:KRS_R0_SERVER_FRAC=0.6",
    "f67": "copy:KRS_R0_SERVER_FRAC=2 / 3",
    "f75": "copy:KRS_R0_SERVER_FRAC=0.75",
    "f50": "copy:KRS_R0_SERVER_FRAC=0.5",
    "pself": "copy:KRS_P_R0_SELF_READ=True",
    "g2": 'copy:KRS_P_R0_SELF_READ=True;KRS_P_SELF_RULE="row";KRS_R0_SERVER_FRAC=7 / 12;KRS_FRAC_ONLY_SELF=True',
    "pselfrow": 'copy:KRS_P_R0_SELF_READ=True;KRS_P_SELF_RULE="row"',
    "pself_sfirst": 'copy:KRS_P_R0_SELF_READ=True;KRS_KERNEL_ORDER="sfirst"',
    "pself_f58": "copy:KRS_P_R0_SELF_READ=True;KRS_R0_SERVER_FRAC=7 / 12",
    "r0ceil": "copy:KRS_R0_SERVER_ROWS=lambda cr, n: (cr + 1) // 2",
    "allceil": "copy:KRS_R0_SERVER_ROWS=lambda cr, n: (cr + 1) // 2;KRS_SERVER_ROWS=lambda cr, n: (cr + 1) // 2",
    # Partners only on rectangles of >= 10 receivers.
    "pmin10": "copy:KRS_PARTNER_MIN_RECEIVERS=10",
    # Kernel list order.
    "pfirst": 'copy:KRS_KERNEL_ORDER="pfirst"',
    "sfirst": 'copy:KRS_KERNEL_ORDER="sfirst"',
    "sfirst_r0q3": 'copy:KRS_KERNEL_ORDER="sfirst";KRS_R0_SERVER_ROWS=lambda cr, n: (3 * cr + 2) // 4',
    "pfirst_r0q3": 'copy:KRS_KERNEL_ORDER="pfirst";KRS_R0_SERVER_ROWS=lambda cr, n: (3 * cr + 2) // 4',
    # Ablations (wrong output, not checked): round 0's DRAM reads / data multicasts skipped on both sources, the
    # partner only (p_) or the first server only (s_).
    "noread": 'copy:KRS_SERVER_DEFINES=(("KRS_NO_R0_READ","1"),)',
    "nomcast": 'copy:KRS_SERVER_DEFINES=(("KRS_NO_R0_MCAST","1"),)',
    "noread_nomcast": 'copy:KRS_SERVER_DEFINES=(("KRS_NO_R0_READ","1"),("KRS_NO_R0_MCAST","1"))',
    "p_noread": 'copy:KRS_PARTNER_DEFINES=(("KRS_NO_R0_READ","1"),)',
    "p_nomcast": 'copy:KRS_PARTNER_DEFINES=(("KRS_NO_R0_MCAST","1"),)',
    "s_noread": 'copy:KRS_FIRST_DEFINES=(("KRS_NO_R0_READ","1"),)',
    "s_nomcast": 'copy:KRS_FIRST_DEFINES=(("KRS_NO_R0_MCAST","1"),)',
}
ABLATED = ("noread", "nomcast")  # substring match


def _fixed_place(table):
    """KRS_PLACE from {rect low corner (logical x, y): (server xy, partner xy)}."""

    def place(rects, spare, virt):
        out = []
        for r in rects:
            s, p = table[(r.start.x, r.start.y)]
            assert s in spare and (p is None or p in spare), (s, p, spare)
            out.append((s, p))
        return out

    return place


# 86-core layers (res3.x, decoder res3): big rectangle (0, 0)-(10, 6), small (0, 7)-(8, 7). HEAD: big server (10, 8),
# partner (0, 9); small server (9, 7), partner (10, 7) (logical).
PLACES = {
    "b_p69": {(0, 0): ((10, 8), (6, 9)), (0, 7): ((9, 7), (10, 7))},
    "b_p68": {(0, 0): ((10, 8), (6, 8)), (0, 7): ((9, 7), (10, 7))},
}
for _n in PLACES:
    NAMED[_n] = f"copy:KRS_PLACE=_fixed_place(PLACES[{_n!r}])"
    NAMED[_n + "_s5"] = f"copy:KRS_PLACE=_fixed_place(PLACES[{_n!r}]);KRS_R0_SERVER_FRAC=5 / 12"
    NAMED[_n + "_s4"] = f"copy:KRS_PLACE=_fixed_place(PLACES[{_n!r}]);KRS_R0_SERVER_FRAC=4 / 12"


def _parse(variant):
    variant = NAMED.get(variant, variant)
    if os.environ.get("KRS_UNZ") == "1":
        pkg, sep, rest = variant.partition(":")
        variant = (pkg if pkg.endswith("_unz") else pkg + "_unz") + sep + rest
    pkg, _, knobs = variant.partition(":")
    kv = {}
    for item in [k for k in knobs.split(";") if k]:
        name, _, val = item.partition("=")
        kv[name] = eval(val)
    return pkg, kv


@pytest.mark.parametrize("case_id,variant,rep", PARAMS, ids=[f"{c}-{v}-{r}" for c, v, r in PARAMS])
def test_krs_ab(case_id, variant, rep, device, monkeypatch):
    import ttnn

    # Variants can share kernels but differ in runtime-arg layout: never let one hit another's cached program.
    device.clear_program_cache()
    pkg, knobs = _parse(variant)
    op_mod, pd_mod = copy_modules(pkg)
    for name, val in knobs.items():
        assert hasattr(pd_mod, name), name
        monkeypatch.setattr(pd_mod, name, val)
    if hasattr(pd_mod, "KRS_PLACEMENT_LOG"):
        pd_mod.KRS_PLACEMENT_LOG.clear()
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
    if getattr(pd_mod, "KRS_PLACEMENT_LOG", None):
        with open(HERE / "placement.jsonl", "a") as f:
            for e in pd_mod.KRS_PLACEMENT_LOG:
                f.write(json.dumps(dict(case=case_id, variant=variant, **e)) + "\n")
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


# HiFi4 HEIGHT perf-file shapes (test_conv2d_3x3_height_sharded_perf): KRS_HS = comma list of layer ids, or "all".
_HS = {c[0]: c for c in perf.HEIGHT_SHAPES}
_HS_SEL = list(_HS) if os.environ.get("KRS_HS") == "all" else [c for c in os.environ.get("KRS_HS", "").split(",") if c]
HS_PARAMS = []
for r in range(REPS if _HS_SEL else 0):
    vs = VARS if (os.environ.get("KRS_ORDER") != "alt" or r % 2 == 0) else list(reversed(VARS))
    for c in _HS_SEL:
        HS_PARAMS += [(c, v, r) for v in vs]


@pytest.mark.parametrize("layer,variant,rep", HS_PARAMS, ids=[f"hs_{c}-{v}-{r}" for c, v, r in HS_PARAMS])
def test_krs_hs(layer, variant, rep, device, monkeypatch):
    import ttnn

    device.clear_program_cache()
    pkg, knobs = _parse(variant)
    op_mod, pd_mod = copy_modules(pkg)
    for name, val in knobs.items():
        assert hasattr(pd_mod, name), name
        monkeypatch.setattr(pd_mod, name, val)
    for name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
        monkeypatch.setattr(perf, name, getattr(op_mod, name))
    captured = {}
    real = op_mod.conv2d_3x3

    def capture(*a, **k):
        out = real(*a, **k)
        captured["out"] = ttnn.to_torch(out).float()
        return out

    monkeypatch.setattr(perf, "conv2d_3x3", capture)
    with open(CALLS, "a") as f:
        f.write(f"hs_{layer} {variant} {rep}\n")
    perf.test_conv2d_3x3_height_sharded_perf(*_HS[layer], device)
    out = captured["out"]
    first = _OUT.setdefault("hs_" + layer, (variant, out))
    diff = int((first[1] != out).sum().item()) if first[1].shape == out.shape else -1
    with open(HERE / "exact.jsonl", "a") as f:
        f.write(
            json.dumps(dict(case="hs_" + layer, variant=variant, rep=rep, ref=first[0], differ=diff, numel=out.numel()))
            + "\n"
        )
    assert diff == 0, f"hs_{layer}: {variant} differs from {first[0]} in {diff} elements"

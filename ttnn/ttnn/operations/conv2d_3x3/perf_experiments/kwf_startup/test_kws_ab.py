# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B in one process: the perf focus test body (tests/.../test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus) run
once per (case, variant), the op swapped per call between this experiment's copies (kws_copy.py PKGS: base, copy,
base_unz, copy_unz). A variant may carry descriptor knobs: `copy:NAME=VALUE+NAME=VALUE` (monkeypatched on the copy's
program-descriptor module for that call only; `@NAME` patches the op module).

Env: KWS_CASES (comma list of aliases below), KWS_VARIANTS (comma list, run in that order), KWS_ORDER=alt (reverse the
variant order on every other repetition), KWS_REPS (repetitions, default 1), KWS_UNZ=1 (zone-stripped twins).
Every call's output is kept per case; each later variant is compared element-for-element with the first variant of its
case (exact.jsonl records the differing-element count). PCC vs the torch reference is asserted by the body.
DEVICE KERNEL DURATION rows come back in call order (run.sh joins them with runs/calls_<label>.txt).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kws_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as perf  # noqa: E402

HERE = Path(__file__).resolve().parent
# Short aliases of the kw-fold HEIGHT focus cases, by layer-name prefix.
ALIAS = {
    "stem2": "stem.conv2",
    "stem3": "stem.conv3",
    "res2": "res2.{0,1,2}.conv2",
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
SEL = [c for c in os.environ.get("KWS_CASES", "res2").split(",") if c]
VARS = [v for v in os.environ.get("KWS_VARIANTS", "base,copy").split(",") if v]
REPS = int(os.environ.get("KWS_REPS", "1"))
PARAMS = []
for r in range(REPS):
    vs = VARS if (os.environ.get("KWS_ORDER") != "alt" or r % 2 == 0) else list(reversed(VARS))
    for c in SEL:
        PARAMS += [(c, v, r) for v in vs]
CALLS = HERE / "runs" / os.environ.get("KWS_CALLS", "calls.txt")
_OUT = {}

# Named variants: knobs of op_copy/ (see that copy's descriptor).
NAMED = {
    "ackp": 'copy:KWS_DEFINES=(("KWS_ACK_PROBE","1"),)',
    "ackp_first": 'copy:KWS_SERVERS_FIRST=True+KWS_SERVER_PICK="first"+KWS_DEFINES=(("KWS_ACK_PROBE","1"),)',
    "nowt": 'copy:KWS_DEFINES=(("KWS_NOWT","1"),)',
    "nowt_nofill": 'copy:KWS_DEFINES=(("KWS_NOWT","1"),("KWS_NOFILL","1"))',
    "nofill": 'copy:KWS_DEFINES=(("KWS_NOFILL","1"),)',
    "noown": "copy:KWS_NOOWN_PLAN=True",
    "early": 'copy:KWS_DEFINES=(("KWS_EARLY_ISSUE","1"),)',
    "early_sf": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_EARLY_ISSUE","1"),)',
    "early_nowt_nofill": 'copy:KWS_DEFINES=(("KWS_EARLY_ISSUE","1"),("KWS_NOWT","1"),("KWS_NOFILL","1"))',
    "ctrl": 'copy:KWS_DEFINES=(("KWS_CTRL_PROBE","1"),)',
    "early_ctrl": 'copy:KWS_DEFINES=(("KWS_EARLY_ISSUE","1"),("KWS_CTRL_PROBE","1"))',
    "r0": 'copy:KWS_DEFINES=(("KWS_R0_LOOP","1"),)',
    "r0_early": 'copy:KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"))',
    "r0_sf": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),)',
    "r0_early_sf": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"))',
    "r0_early_sf_noack": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"))',
    "r0_early_noack": 'copy:KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"))',
    "noack": 'copy:KWS_DEFINES=(("KWS_NO_START_ACK","1"),)',
    "all5": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"))',
    "noack_readall": 'copy:KWS_DEFINES=(("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"))',
    "readall": 'copy:KWS_DEFINES=(("KWS_READ_ALL","1"),)',
    "all5_ack": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_READ_ALL","1"))',
    "all5_nosf": 'copy:KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"))',
    "all5_nor0": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"))',
    "all5_nowt": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"),("KWS_NOWT","1"))',
    "all6": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"),("KWS_R0_ALL","1"))',
    "all5_nora": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_EARLY_ISSUE","1"),("KWS_NO_START_ACK","1"))',
    "all5_noearly": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_NO_START_ACK","1"),("KWS_READ_ALL","1"))',
    "cand": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_NO_START_ACK","1"),("KWS_READ_SMALL","1"))',
    "cand_nobal": 'copy:KWS_SERVERS_FIRST=True+KWS_FILL_BALANCE=False+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_NO_START_ACK","1"),("KWS_READ_SMALL","1"))',
    "cand_noown": 'copy:KWS_SERVERS_FIRST=True+KWS_NOOWN_PLAN=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_NO_START_ACK","1"),("KWS_READ_SMALL","1"))',
    "cand_nors": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_NO_START_ACK","1"))',
    "cand_nosf": 'copy:KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_NO_START_ACK","1"),("KWS_READ_SMALL","1"))',
    "cand_ack": 'copy:KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_R0_LOOP","1"),("KWS_READ_SMALL","1"))',
    "ack0p": 'copy:KWS_ACK_NOC0=True+KWS_DEFINES=(("KWS_ACK_PROBE","1"),)',
    "ack0p_sf": 'copy:KWS_ACK_NOC0=True+KWS_SERVERS_FIRST=True+KWS_DEFINES=(("KWS_ACK_PROBE","1"),)',
    "ack0p_pf": 'copy:KWS_ACK_NOC0=True+KWS_SERVER_PICK="first"+KWS_DEFINES=(("KWS_ACK_PROBE","1"),)',
}


def _parse(variant):
    variant = NAMED.get(variant, variant)
    if os.environ.get("KWS_UNZ") == "1":
        pkg, sep, rest = variant.partition(":")
        variant = (pkg if pkg.endswith("_unz") else pkg + "_unz") + sep + rest
    if os.environ.get("KWS_LZ") == "1":
        # Light-zoned twins (mk_lz.sh): compute waits only.
        pkg, sep, rest = variant.partition(":")
        variant = (pkg if pkg.endswith("_lz") else pkg + "_lz") + sep + rest
    pkg, _, knobs = variant.partition(":")
    kv = {}
    for item in [k for k in knobs.split("+") if k]:
        name, _, val = item.partition("=")
        kv[name] = eval(val)
    return pkg, kv


@pytest.mark.parametrize("case_id,variant,rep", PARAMS, ids=[f"{c}-{v}-{r}" for c, v, r in PARAMS])
def test_kws_ab(case_id, variant, rep, device, monkeypatch):
    import ttnn

    # Variants can share kernels but differ in runtime-arg layout: never let one hit another's cached program.
    device.clear_program_cache()
    pkg, knobs = _parse(variant)
    op_mod, pd_mod = copy_modules(pkg)
    for name, val in knobs.items():
        mod = op_mod if name.startswith("@") else pd_mod
        name = name.lstrip("@")
        assert hasattr(mod, name), name
        monkeypatch.setattr(mod, name, val)
    for name in dir(pd_mod):
        m = getattr(pd_mod, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()
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
        f.write(f"{case_id} {variant} {rep}\n")
    perf.test_conv2d_3x3_perf_focus(CASES[case_id], device)
    if os.environ.get("CONV2D_3X3_PERF_ABLATION") == "1":
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

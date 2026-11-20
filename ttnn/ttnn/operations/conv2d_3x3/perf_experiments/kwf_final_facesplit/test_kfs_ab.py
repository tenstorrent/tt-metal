# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B in one process: the perf focus test body (tests/.../test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus) run
once per (case, variant), the op swapped per call between this experiment's copies (kfs_copy.py PKGS). A variant may
carry descriptor knobs: `copy:NAME=VALUE+NAME=VALUE` (monkeypatched on the copy's program-descriptor module for that
call only; `@NAME` patches the op module), or be one of the NAMED variants below.

Env: KFS_CASES (comma list of aliases below), KFS_VARIANTS (comma list, run in that order), KFS_ORDER=alt (reverse the
variant order on every other repetition), KFS_REPS (repetitions, default 1), KFS_UNZ=1 (zone-stripped twins).
Every call's output is kept per case; each later variant is compared element-for-element with the first variant of its
case (exact.jsonl records the differing-element count, max abs diff and PCC vs that first variant). PCC vs the torch
reference is asserted by the body. With CONV2D_3X3_PERF_ABLATION=1 (stubbed data movement) nothing is checked.
DEVICE KERNEL DURATION rows come back in call order (run.sh joins them with runs/calls_<label>.txt).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfs_copy import copy_modules  # noqa: E402

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
SEL = [c for c in os.environ.get("KFS_CASES", "stem2").split(",") if c]
VARS = [v for v in os.environ.get("KFS_VARIANTS", "base,copy").split(",") if v]
REPS = int(os.environ.get("KFS_REPS", "1"))
PARAMS = []
for r in range(REPS):
    vs = VARS if (os.environ.get("KFS_ORDER") != "alt" or r % 2 == 0) else list(reversed(VARS))
    for c in SEL:
        PARAMS += [(c, v, r) for v in vs]
CALLS = HERE / "runs" / os.environ.get("KFS_CALLS", "calls.txt")
_OUT = {}


def _defs(*names):
    return "copy:KFS_DEFINES=(" + "".join(f'("{n}","1"),' for n in names) + ")"


# Named variants: compute-kernel defines of op_copy/ (see its conv2d_3x3_kw_fold.hpp).
NAMED = {
    "nopack": _defs("KFS_ABL_NOPACK"),
    "nomath1": _defs("KFS_ABL_MATH1"),
    "nomath1_nopack": _defs("KFS_ABL_MATH1", "KFS_ABL_NOPACK"),
    "nomm_base": _defs("KFS_ABL_NOMM").replace("copy:", "base:"),
    "nomm_copy": _defs("KFS_ABL_NOMM"),
    "nomm_remap": _defs("KFS_ABL_NOMM", "KFS_REMAP"),
    "remap": _defs("KFS_REMAP"),
    "lean": _defs("KFS_LEAN_PACK"),
    "zk2": _defs("KFS_FORCE_ZK2"),
}


def _parse(variant):
    variant = NAMED.get(variant, variant)
    if os.environ.get("KFS_UNZ") == "1":
        pkg, sep, rest = variant.partition(":")
        variant = (pkg if pkg.endswith("_unz") else pkg + "_unz") + sep + rest
    pkg, _, knobs = variant.partition(":")
    kv = {}
    for item in [k for k in knobs.split("+") if k]:
        name, _, val = item.partition("=")
        kv[name] = eval(val)
    return pkg, kv


@pytest.mark.parametrize("case_id,variant,rep", PARAMS, ids=[f"{c}-{v}-{r}" for c, v, r in PARAMS])
def test_kfs_ab(case_id, variant, rep, device, monkeypatch):
    import ttnn

    # Variants can share kernels but differ in runtime-arg layout: never let one hit another's cached program.
    device.clear_program_cache()
    # "<variant>@dm": every data-movement stage stubbed (ablation hooks) for this call; no output check.
    base_variant, _, abl = variant.partition("@")
    stub = abl == "dm"
    if stub:
        monkeypatch.setenv("CONV2D_3X3_ABLATE", "gather,fill,weights,scatter")
        monkeypatch.setattr(perf, "ABLATION", True)
    if base_variant.startswith("no") and base_variant in NAMED:
        # final-pass ablations (NAMED): wrong output by construction, no check
        stub = True
        monkeypatch.setattr(perf, "ABLATION", True)
    pkg, knobs = _parse(base_variant)
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
    if stub or os.environ.get("CONV2D_3X3_PERF_ABLATION") == "1":
        return  # stubbed variants: no output check
    import torch

    out = captured["out"]
    first = _OUT.setdefault(case_id, (variant, out))
    same_shape = first[1].shape == out.shape
    diff = int((first[1] != out).sum().item()) if same_shape else -1
    maxabs = float((first[1] - out).abs().max().item()) if same_shape else -1.0
    pcc = float(torch.corrcoef(torch.stack([first[1].flatten(), out.flatten()]))[0, 1].item()) if same_shape else -1.0
    with open(HERE / "exact.jsonl", "a") as f:
        f.write(
            json.dumps(
                dict(
                    case=case_id,
                    variant=variant,
                    rep=rep,
                    ref=first[0],
                    differ=diff,
                    maxabs=maxabs,
                    pcc=pcc,
                    numel=out.numel(),
                )
            )
            + "\n"
        )

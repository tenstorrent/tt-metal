# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Driver for the stateful-NoC (lever B13) isolated bake-off.

Correctness is the ONLY assertion: every variant must reproduce the input
bit-exactly.  Perf is measured and printed, never asserted.

    scripts/run_safe_pytest.sh --profile \\
        ttnn/ttnn/operations/rms_norm/perf_experiments/stateful_noc/test_stateful_noc.py -s

Env:
    SN_SET   which arm set to dispatch:
             "focus"  (default) reader/writer variant menu on the focus geometry
             "domain" the per-regime sweep of baseline vs the recommended pair
             "dtype"  fp32 / bfloat8_b / row-major
             "all"
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

# Loaded by PATH, not by package import: `ttnn/ttnn/operations/__init__.py` walks
# every package under operations/ at `import ttnn` time, so a module-level
# `from ttnn.operations...import` here would execute during ttnn's own init.
_HERE = Path(__file__).parent


def _load_bench():
    name = "sn_bench_stateful_noc"
    spec = importlib.util.spec_from_file_location(name, _HERE / "bench_stateful_noc.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses/typing resolve __module__ through sys.modules
    spec.loader.exec_module(mod)
    return mod


b = _load_bench()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


# An ARM is (regime, rvar, wvar, rskip, wskip).
#
# ISOLATED arms (one half stubbed) are what price the issue loop: on one core a
# simultaneous read+write stream saturates that core's NoC in both directions, so
# in the COPY arms a cheaper issue loop just moves the cost into the barrier.
# The op's focus case has its writer starved for 35 of 44 us, i.e. the reader is
# effectively alone on the NoC — the `rd_only` arms are the faithful predictor.
def _menu(reg, rvars=(0, 1, 2, 3, 4), wvars=(0, 1, 2, 3)):
    arms = [(reg, r, 0, 0, 1) for r in rvars]  # reader menu, writer stubbed
    arms += [(reg, 0, w, 1, 0) for w in wvars]  # writer menu, reader stubbed
    arms += [(reg, 0, 0, 0, 0), (reg, 3, 3, 0, 0)]  # composite copy, both ends
    return arms


FOCUS_ARMS = _menu("focus")

# Domain sweep: baseline vs the recommended pair, isolated (so the per-transaction
# price is comparable across regimes) plus the composite copy for correctness.
def _sweep(reg):
    return [
        (reg, 0, 0, 0, 1),
        (reg, 3, 0, 0, 1),
        (reg, 0, 0, 1, 0),
        (reg, 0, 3, 1, 0),
        (reg, 0, 0, 0, 0),
        (reg, 3, 3, 0, 0),
    ]


DOMAIN_ARMS = [a for reg in ("dram_bound", "wide", "tiny", "tiny4") for a in _sweep(reg)]
DTYPE_ARMS = [a for reg in ("fp32", "bf8b", "rm", "rm_odd") for a in _sweep(reg)]
SMOKE_ARMS = [("focus", 0, 0, 0, 0), ("focus", 3, 3, 0, 0)]

SETS = {"smoke": SMOKE_ARMS, "focus": FOCUS_ARMS, "domain": DOMAIN_ARMS, "dtype": DTYPE_ARMS}

# Round 2: the ORDER-PRESERVING options (one_packet / affine) on the regimes where
# bank-major reordering measurably REGRESSED, plus `bank_rot` (bank-major with a
# per-core rotation) to test whether the regression is DRAM hot-banking.
# Every variant also gets at least one un-stubbed copy arm, so nothing is
# recommended off a stubbed measurement alone.
def _safe(reg):
    return [
        (reg, 1, 0, 0, 1),
        (reg, 2, 0, 0, 1),
        (reg, 5, 0, 0, 1),
        (reg, 0, 1, 1, 0),
        (reg, 0, 2, 1, 0),
        (reg, 0, 4, 1, 0),
        (reg, 1, 1, 0, 0),
        (reg, 2, 2, 0, 0),
        (reg, 5, 4, 0, 0),
    ]


SAFE_ARMS = [a for reg in ("dram_bound", "fp32", "rm", "wide") for a in _safe(reg)]
GATE_ARMS = [
    ("focus", 1, 1, 0, 0),
    ("focus", 2, 2, 0, 0),
    ("focus", 4, 3, 0, 0),
    ("focus", 5, 4, 0, 0),
    ("bf8b", 2, 2, 0, 0),
    ("bf8b", 5, 4, 0, 0),
    ("rm_odd", 2, 2, 0, 0),
    ("rm_odd", 5, 4, 0, 0),
    ("tiny", 2, 2, 0, 0),
    ("tiny", 2, 0, 0, 1),
    ("tiny", 0, 2, 1, 0),
    ("tiny4", 2, 0, 0, 1),
    ("tiny4", 0, 2, 1, 0),
    ("bf8b", 2, 0, 0, 1),
    ("bf8b", 0, 2, 1, 0),
]
# Round 3: (a) `one_packet` at the B0 per-core-overhead end, where `affine`'s
# fixed bank-table build measurably costs; (b) a REPEAT of the focus reader/writer
# menu, because one_packet-vs-affine sits inside the noise band there.
FINAL_ARMS = [
    ("tiny", 1, 0, 0, 1),
    ("tiny", 0, 1, 1, 0),
    ("tiny", 1, 1, 0, 0),
    ("tiny4", 1, 0, 0, 1),
    ("tiny4", 0, 1, 1, 0),
    ("focus", 0, 0, 0, 1),
    ("focus", 1, 0, 0, 1),
    ("focus", 2, 0, 0, 1),
    ("focus", 0, 0, 1, 0),
    ("focus", 0, 1, 1, 0),
    ("focus", 0, 2, 1, 0),
    ("bf8b", 1, 0, 0, 1),
    ("bf8b", 0, 1, 1, 0),
]
SETS["final"] = FINAL_ARMS
SETS["safe"] = SAFE_ARMS
SETS["gate"] = GATE_ARMS
SETS["sweep"] = DOMAIN_ARMS + DTYPE_ARMS
SETS["all"] = FOCUS_ARMS + DOMAIN_ARMS + DTYPE_ARMS


@pytest.mark.timeout(3600)
def test_stateful_noc(device):
    arms = SETS[os.environ.get("SN_SET", "focus")]
    manifest = []
    failures = []
    for reg, rvar, wvar, rskip, wskip in arms:
        if rskip:
            label = f"{reg}/wr_only:{b.WVAR_NAMES[wvar]}"
        elif wskip:
            label = f"{reg}/rd_only:{b.RVAR_NAMES[rvar]}"
        else:
            label = f"{reg}/copy/r:{b.RVAR_NAMES[rvar]}/w:{b.WVAR_NAMES[wvar]}"
        ok, detail = b.dispatch_arm(
            device, manifest, label, reg, rvar, wvar, rskip=rskip, wskip=wskip
        )
        print(f"SN: {label:<44} {'OK' if ok else 'FAIL'}  {detail}")
        if not ok:
            failures.append((label, detail))
    path = b.write_manifest(manifest)
    print(f"\nSN: manifest -> {path} ({len(manifest)} arms)")
    assert not failures, f"variants produced wrong data: {failures}"

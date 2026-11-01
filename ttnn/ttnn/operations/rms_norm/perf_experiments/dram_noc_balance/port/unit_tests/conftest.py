# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Copy of tests/ttnn/unit_tests/operations/rms_norm/ (HEAD) run against the dram_noc_balance PORT (../op_port/).

The port's descriptor replaces the op's rms_norm_program_descriptor module for the whole session (sys.modules entry,
package attribute — so `from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd` and the tests'
monkeypatches hit the port — and rms_norm.py's create_program_descriptor global).

Bit-identity: the package's `rms_norm` is wrapped so EVERY call first runs HEAD's descriptor (the unmodified op, with
every monkeypatched attribute of the port module — knobs and functions such as _choose_k — mirrored onto it), reads the result back and frees it,
then runs the port and asserts its output is bit-identical (NaN-aware) to HEAD's.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.rms_norm as pkg
import ttnn.operations.rms_norm.rms_norm  # noqa: F401  (the package attribute `rms_norm` is the function)

entry = sys.modules["ttnn.operations.rms_norm.rms_norm"]  # the entry MODULE (its create_program_descriptor global)
_head = sys.modules["ttnn.operations.rms_norm.rms_norm_program_descriptor"]  # HEAD's descriptor module

_NAME = "ttnn.operations.rms_norm.rms_norm_program_descriptor"
_PORT = Path(__file__).resolve().parent.parent / "op_port" / "rms_norm_program_descriptor.py"
_spec = importlib.util.spec_from_file_location(_NAME, _PORT)
_port = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = _port
_spec.loader.exec_module(_port)
pkg.rms_norm_program_descriptor = _port
entry.create_program_descriptor = _port.create_program_descriptor
assert "dram_noc_balance/port/op_port" in str(_port.KERNEL_DIR) and _port.NOC_SWAP_COLUMNS == 3
assert _head is not _port and not hasattr(_head, "NOC_SWAP_COLUMNS")

_fn = entry.rms_norm
_NO_MIRROR = {"NOC_SWAP_COLUMNS", "_noc_swapped_cores"}
_HEAD_ORIG = dict(vars(_head))
_PORT_ORIG = dict(vars(_port))  # snapshot before any test runs (before the counting wrapper below)
STATS = {"calls": 0, "identical": 0, "swapped_dispatches": 0, "swapped_cores": 0}

_orig_swapped = _port._noc_swapped_cores


def _counting_swapped(*a, **kw):
    s = _orig_swapped(*a, **kw)
    if s:
        STATS["swapped_dispatches"] += 1
        STATS["swapped_cores"] += len(s)
    return s


_port._noc_swapped_cores = _counting_swapped


def _host(t):
    return ttnn.to_torch(t)


def _dual(*args, **kwargs):
    # mirror the tests' monkeypatches (knobs AND functions, e.g. _choose_k) of the port module onto HEAD's
    for name, val in _HEAD_ORIG.items():
        setattr(_head, name, val)
    for name, val in vars(_port).items():
        if name in _HEAD_ORIG and name not in _NO_MIRROR and val is not _PORT_ORIG.get(name):
            setattr(_head, name, val)
    entry.create_program_descriptor = _head.create_program_descriptor
    try:
        ref_t = _fn(*args, **kwargs)
    finally:
        entry.create_program_descriptor = _port.create_program_descriptor
    ref = _host(ref_t)
    ttnn.deallocate(ref_t)  # free before the port runs: an L1 output would shrink the port's CB budget
    out_t = _fn(*args, **kwargs)
    out = _host(out_t)
    STATS["calls"] += 1
    same = ref.shape == out.shape and bool(((ref == out) | (ref.isnan() & out.isnan())).all())
    assert same, f"port output differs from HEAD: max abs {float((ref.float() - out.float()).abs().max())}"
    STATS["identical"] += 1
    return out_t


pkg.rms_norm = _dual


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


def pytest_report_header(config):
    return f"rms_norm descriptor swapped -> {_PORT} (every call A/B'd bit-exact vs HEAD)"


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"DRAM_NOC_BALANCE PORT: {STATS}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p swx_plugin, this dir on PYTHONPATH): run genify_matmul's own tests (unit / golden) with this dir's
pd_swx.py + kernels/ (BANK_LOCAL_SWIGLU_XCHG) in place of the op's descriptor, without touching the op. Prints one
SWXPLAN line per R1-family op call: test id, xchg (exchange split taken) and the split."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_current = {"id": "?"}
_mod = {}


def _pd_swx():
    if "m" not in _mod:
        spec = importlib.util.spec_from_file_location("_t2p2_pd_swx_plugin", HERE / "pd_swx.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        orig = m.plan

        def _plan(*a, **k):
            geom, grid, blk, gxy = orig(*a, **k)
            print(f"SWXPLAN {_current['id']} xchg={geom.xchg} epi={geom.epilogue} {grid.placement} {grid.grid_n}")
            return geom, grid, blk, gxy

        m.plan = _plan
        _mod["m"] = m
    return _mod["m"]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _current["id"] = item.nodeid.split("::")[-1]


@pytest.fixture(autouse=True)
def _swx_swap(monkeypatch):
    gm = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")
    monkeypatch.setattr(gm, "create_program_descriptor", _pd_swx().create_program_descriptor)

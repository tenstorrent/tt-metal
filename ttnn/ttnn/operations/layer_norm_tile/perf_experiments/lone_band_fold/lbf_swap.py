# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin (-p lbf_swap): drive the real layer_norm_tile op through this dir's descriptor copy.

The copy also replaces the descriptor module in sys.modules, so tests that monkeypatch descriptor knobs
(test_layer_norm_tile_row_stats.py, test_layer_norm_tile_half_band.py) patch the copy. LBF_FOLD=0|1 selects
the base or the folded lone band (one per pytest process: generic_op caches programs).
"""

import importlib.util
import sys
from pathlib import Path

import ttnn.operations.layer_norm_tile as _pkg
from ttnn.operations.layer_norm_tile import layer_norm_tile as _op

_NAME = "ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor"
_spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).parent / "lbf_descriptor.py")
lbf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lbf)
sys.modules[_NAME] = lbf
sys.modules["lbf_descriptor"] = lbf
_pkg.layer_norm_tile_program_descriptor = lbf
_op.select_blocking = lbf.select_blocking
_op.create_program_descriptor = lbf.create_program_descriptor

_calls = {"fold": 0, "total": 0}
_create = lbf.create_program_descriptor


def _counting_create(input_tensor, gamma, beta, residual, output_tensor, blocking, **kwargs):
    _calls["total"] += 1
    _calls["fold"] += int(lbf._fold(blocking))
    return _create(input_tensor, gamma, beta, residual, output_tensor, blocking, **kwargs)


_op.create_program_descriptor = _counting_create


def pytest_sessionfinish(session):
    print(f"\nLBF_SWAP descriptors={_calls['total']} folded={_calls['fold']} FOLD_LONE_BAND={lbf.FOLD_LONE_BAND}")


# LBF_KNOBS="NAME=int,..." overrides descriptor-copy knobs (as test_perf_bench.py's LN_KNOBS).
import os as _os

for _item in filter(None, _os.environ.get("LBF_KNOBS", "").split(",")):
    _knob, _value = _item.split("=")
    setattr(lbf, _knob, type(getattr(lbf, _knob))(int(_value)))

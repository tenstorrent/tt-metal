# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin (-p swap_graduate): run the op through graduate/ (the proposed gate change, op kernels).

Counts dispatches and how many took the residual-on-writer path; prints the tally at session end.
"""

import importlib.util
import sys
from pathlib import Path

from ttnn.operations.layer_norm_tile import layer_norm as _layer_norm

_spec = importlib.util.spec_from_file_location(
    "srm_graduate_pd", Path(__file__).parent / "graduate" / "layer_norm_tile_program_descriptor.py"
)
pd = importlib.util.module_from_spec(_spec)
sys.modules["srm_graduate_pd"] = pd
_spec.loader.exec_module(pd)

_tally = {"calls": 0, "residual_on_writer": 0}


def _select_blocking(*args, **kwargs):
    blocking = pd.select_blocking(*args, **kwargs)
    _tally["calls"] += 1
    _tally["residual_on_writer"] += bool(blocking and blocking.residual_on_writer)
    return blocking


_layer_norm.__globals__["create_program_descriptor"] = pd.create_program_descriptor
_layer_norm.__globals__["select_blocking"] = _select_blocking


def pytest_sessionfinish(session, exitstatus):
    print(f"\nSRM_GRADUATE_TALLY {_tally}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin (-p lbh_swap, PYTHONPATH=<this dir>): run the real layer_norm_tile entry through lbh_descriptor.py
(this dir's kernels). The copy also replaces the op's descriptor module in sys.modules, so the op's unit tests'
knob monkeypatches land on it. LBH_VARIANT=base|half (default half), one per pytest process."""

import importlib.util
import sys
from pathlib import Path

import ttnn.operations.layer_norm_tile as _pkg
from ttnn.operations.layer_norm_tile import layer_norm_tile as _op

_NAME = "ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor"
_spec = importlib.util.spec_from_file_location("lbh_descriptor", Path(__file__).parent / "lbh_descriptor.py")
lbh = importlib.util.module_from_spec(_spec)
sys.modules["lbh_descriptor"] = lbh
_spec.loader.exec_module(lbh)
sys.modules[_NAME] = lbh
sys.modules["layer_norm_tile.layer_norm_tile_program_descriptor"] = lbh
_pkg.layer_norm_tile_program_descriptor = lbh
_op.select_blocking = lbh.select_blocking
_op.create_program_descriptor = lbh.create_program_descriptor

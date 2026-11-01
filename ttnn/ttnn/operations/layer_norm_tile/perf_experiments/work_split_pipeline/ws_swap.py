# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin (-p ws_swap): drive the real layer_norm_tile op through this dir's descriptor copy.

WS_SPLIT selects the row-split strategy (one per pytest process: generic_op caches programs).
"""

import importlib.util
import sys
from pathlib import Path

from ttnn.operations.layer_norm_tile import layer_norm_tile as _op

_spec = importlib.util.spec_from_file_location("ws_descriptor", Path(__file__).parent / "ws_descriptor.py")
ws = importlib.util.module_from_spec(_spec)
sys.modules["ws_descriptor"] = ws
_spec.loader.exec_module(ws)
_op.select_blocking = ws.select_blocking
_op.create_program_descriptor = ws.create_program_descriptor

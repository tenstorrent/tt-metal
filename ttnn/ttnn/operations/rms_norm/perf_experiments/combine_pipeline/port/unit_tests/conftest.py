# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Copy of tests/ttnn/unit_tests/operations/rms_norm/ (HEAD) run against the combine_pipeline PORT.

The port's descriptor (../rms_norm_program_descriptor.py, kernels ../kernels/) replaces the op's
rms_norm_program_descriptor module for the whole session: sys.modules entry, package attribute (what
`from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd` resolves — so the tests' monkeypatches
hit the port), and rms_norm.py's create_program_descriptor global.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

import ttnn.operations.rms_norm as pkg
import ttnn.operations.rms_norm.rms_norm  # noqa: F401  (the package attribute `rms_norm` is the function)

entry = sys.modules["ttnn.operations.rms_norm.rms_norm"]  # the entry MODULE (its create_program_descriptor global)

_NAME = "ttnn.operations.rms_norm.rms_norm_program_descriptor"
_PORT = Path(__file__).resolve().parent.parent / "rms_norm_program_descriptor.py"
_spec = importlib.util.spec_from_file_location(_NAME, _PORT)
_port = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = _port
_spec.loader.exec_module(_port)
pkg.rms_norm_program_descriptor = _port
entry.create_program_descriptor = _port.create_program_descriptor
assert "combine_pipeline/port" in str(_port.KERNEL_DIR) and hasattr(_port, "COMBINE_PIPELINE")
assert entry.create_program_descriptor is _port.create_program_descriptor and hasattr(entry, "validate")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


def pytest_report_header(config):
    return f"rms_norm descriptor swapped -> {_PORT}"


# Evidence that the pipelined path is exercised: count blockings that picked it (per distinct shape / config).
_PIPED = []
_orig_ssb = _port.select_sharded_blocking


def _counting_ssb(geo, sz, input_tensor, budget):
    blk = _orig_ssb(geo, sz, input_tensor, budget)
    if blk.opts.pipe_nb:
        _PIPED.append((tuple(input_tensor.shape), blk.k, blk.block_row_tiles, blk.opts.pipe_nb))
    return blk


_port.select_sharded_blocking = _counting_ssb


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"PIPELINED dispatches: {len(_PIPED)}; distinct: {sorted(set(_PIPED))}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Loads op_copy/ (this experiment's private copy of conv2d_3x3) under a private package name, by path (this directory
has no __init__.py: ttnn.operations walk_packages-imports every package under it on `import ttnn`).
BRF_BANK_RUN=0/1 sets the copy's BANK_RUN_FILL before any blocking is solved (select_blocking is lru-cached)."""

import importlib.util
import os
import sys
from pathlib import Path

OPC = "bank_run_fill_op_copy"


def load():
    if OPC not in sys.modules:
        root = Path(__file__).parent / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
        pd = sys.modules[OPC + ".conv2d_3x3_program_descriptor"]
        pd.BANK_RUN_FILL = os.environ.get("BRF_BANK_RUN", "1") == "1"
        if "BRF_NSTG" in os.environ:
            pd.BRF_NSTG = int(os.environ["BRF_NSTG"])
        pd.BRF_SKIP_DEINT = os.environ.get("BRF_SKIP_DEINT") == "1"
        pd.BRF_LOG = os.environ.get("BRF_LOG") == "1"
        if "BRF_MIN_READ_BYTES" in os.environ:
            pd.BRF_MIN_READ_BYTES = int(os.environ["BRF_MIN_READ_BYTES"])
        if "BRF_CHUNK_BYTES" in os.environ:
            pd.BRF_CHUNK_BYTES = int(os.environ["BRF_CHUNK_BYTES"])
    return sys.modules[OPC], sys.modules[OPC + ".conv2d_3x3_program_descriptor"], sys.modules[OPC + ".conv2d_3x3"]

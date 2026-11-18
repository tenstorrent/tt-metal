# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loads op_copy/ (this experiment's private copy of conv2d_3x3 at 5f23b256d1a) under a private package name.

This directory deliberately has no __init__.py: ttnn.operations walk_packages-executes every module of every package
under it on `import ttnn`, so the copy is loaded by path instead."""

import importlib.util
import sys
from pathlib import Path

import os

# HWM_COPY=grad loads the graduation tree (grad/b, the op at 5f23b256d1a plus graduation.patch) instead of op_copy/.
GRAD = os.environ.get("HWM_COPY") == "grad"
OPC = "height_weight_mcast_grad" if GRAD else "height_weight_mcast_op_copy"


def load():
    if OPC not in sys.modules:
        here = Path(__file__).parent
        root = here / "grad/b/ttnn/ttnn/operations/conv2d_3x3" if GRAD else here / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC], sys.modules[OPC + ".conv2d_3x3_program_descriptor"], sys.modules[OPC + ".conv2d_3x3"]

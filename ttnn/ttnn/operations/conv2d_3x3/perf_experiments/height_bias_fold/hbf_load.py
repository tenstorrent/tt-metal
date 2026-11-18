# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loads op_copy/ (this experiment's private copy of conv2d_3x3 at 5f23b256d1a) under a private package name.

This directory deliberately has no __init__.py: ttnn.operations walk_packages-executes every module of every package
under it on `import ttnn`, so the copy is loaded by path instead."""

import importlib.util
import sys
from pathlib import Path

OPC = "height_bias_fold_op_copy"


def load():
    if OPC not in sys.modules:
        root = Path(__file__).parent / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC], sys.modules[OPC + ".conv2d_3x3_program_descriptor"], sys.modules[OPC + ".conv2d_3x3"]


def set_variant(pd, variant):
    """base: the op unchanged; fold: bias seeded into DEST, last K-block packs the output (partials reloaded through
    SrcA); fold_utd: fold with cb_matmul_partials unpacked straight to DEST (exact fp32 reload); fold_add: no seed, the last
    K-block reloads partials + bias (row-broadcast ELWADD) into DEST; fold_add_c: fold_add with conditional
    (format-change-only) reconfigs around the reload."""
    pd.HBF_FOLD = {"fold": 1, "fold_utd": 1, "fold_add": 2, "fold_add_c": 3}.get(variant, 0)
    pd.HBF_UTD = int(variant == "fold_utd")

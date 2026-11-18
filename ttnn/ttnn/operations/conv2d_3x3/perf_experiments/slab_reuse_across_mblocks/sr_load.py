# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loads op_copy/ (this experiment's private copy of conv2d_3x3) under a private package name.

This directory deliberately has no __init__.py: ttnn.operations walk_packages-executes every module of every package
under it on `import ttnn`, so the copy is loaded by path instead."""

import importlib.util
import os
import sys
from pathlib import Path

# SR_COPY=graduate loads graduate/ (the integration candidate: core slab always on where it fits, no knobs) instead.
COPY = os.environ.get("SR_COPY", "op_copy")
OPC = "slab_reuse_across_mblocks_" + COPY


def load():
    if OPC not in sys.modules:
        root = Path(__file__).parent / COPY
        spec = importlib.util.spec_from_file_location(OPC, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC], sys.modules[OPC + ".conv2d_3x3_program_descriptor"], sys.modules[OPC + ".conv2d_3x3"]


def set_variant(pd, variant):
    """base: every M-block plans, fills and untilizes its own slab (the op's current path). core: the core-wide slab
    wherever it fits the L1 left at the solved per-block blocking (same bm, bk, untilize batch and im2col depth).
    core_first: the core slab at any untilize batch before per-block at any (the ring and batch may shrink for it). select_blocking is lru-cached on its arguments only, so the cache is cleared.
    """
    if COPY != "op_copy":
        assert variant == "core", f"{COPY} has no knobs: it is the core variant"
        return
    pd.CORE_SLAB = variant != "base"
    pd.CORE_SLAB_FIRST = variant == "core_first"
    pd.select_blocking.cache_clear()

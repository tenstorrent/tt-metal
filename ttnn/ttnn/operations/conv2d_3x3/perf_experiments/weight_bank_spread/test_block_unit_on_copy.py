# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The op's BLOCK_SHARDED unit suite (tests/.../test_conv2d_3x3_block_sharded.py), run against this dir's op copy with
one weight_bank_spread variant (correctness only: PCC vs torch as in the suite, every placement / orientation / ragged
row / multi-M-block / writer-only-scatter case it has).

    WBS_UNIT_VARIANT=chunk_diag [WBS_FORCE_KBLOCKED=1] scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_bank_spread/test_block_unit_on_copy.py \\
        -p no:cacheprovider --import-mode=prepend

WBS_FORCE_KBLOCKED=1 skips the N-slice-outer solve, so every K split runs the K-blocked walk (read_weight_k_block).
"""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
# WBS_COPY_DIR=int_check runs the suite on the op with the integration patches applied (no variant knobs there).
_COPY_DIR = os.environ.get("WBS_COPY_DIR", "op_copy")
OPC = f"weight_bank_spread_{_COPY_DIR}"
if OPC not in sys.modules:
    _pkg = types.ModuleType(OPC)
    _pkg.__path__ = [str(HERE / _COPY_DIR)]
    sys.modules[OPC] = _pkg
_pd = importlib.import_module(OPC + ".conv2d_3x3_program_descriptor")
_op = importlib.import_module(OPC + ".conv2d_3x3")

_VARIANT = os.environ.get("WBS_UNIT_VARIANT", "chunk_diag")
_pd.WEIGHT_LAYOUT = {"base": 0, "spread": 1, "chunk": 2}[_VARIANT.removesuffix("_diag")]
_pd.WEIGHT_SENDER_STAGGER = _VARIANT.endswith("_diag")
_pd.FORCE_KBLOCKED = os.environ.get("WBS_FORCE_KBLOCKED") == "1"
_pd.select_blocking.cache_clear()

_unit = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded")
# The suite's functions resolve these through their own module globals: point them at the copy.
_unit.conv2d_3x3 = _op.conv2d_3x3
_unit.prepare_conv2d_weights = _op.prepare_conv2d_weights
_unit.prepare_conv2d_bias = _op.prepare_conv2d_bias
_unit._op_module = _op
_unit._pd = _pd

globals().update({k: v for k, v in vars(_unit).items() if k.startswith("test_")})

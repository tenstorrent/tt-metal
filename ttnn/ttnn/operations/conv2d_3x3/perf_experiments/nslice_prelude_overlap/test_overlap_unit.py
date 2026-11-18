# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the overlapped N-slice prelude on the op's BLOCK_SHARDED unit tests (perf experiment).

Runs every test of tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py against this dir's
op copy with the variant OV_UNIT_VARIANT (a key of test_overlap.VARIANTS; default `ov`) applied. Not for profiling:
it imports the committed op too.
"""

import importlib
import json
import os
from pathlib import Path

import pytest

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded as T
from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded import *  # noqa: F401,F403

pytestmark = pytest.mark.use_module_device

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_prelude_overlap.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == Path(__file__).parent / "op_copy", pd.KERNEL_DIR


@pytest.fixture(autouse=True)
def _overlap_op(monkeypatch):
    from test_overlap import KNOB_DEFAULTS, VARIANTS

    variant = os.environ.get("OV_UNIT_VARIANT", "ov")
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    monkeypatch.setattr(T, "conv2d_3x3", op.conv2d_3x3)
    monkeypatch.setattr(T, "prepare_conv2d_weights", op.prepare_conv2d_weights)
    monkeypatch.setattr(T, "prepare_conv2d_bias", op.prepare_conv2d_bias)
    monkeypatch.setattr(T, "_op_module", op)
    monkeypatch.setattr(T, "_pd", pd)
    # Which dispatches take the overlapped prelude (runs/unit_engaged.jsonl).
    real_create = op.create_program_descriptor

    def recording_create(*args, geometry, blocking, **kwargs):
        g, b = geometry, blocking
        engaged = bool(
            pd.NSLICE_PRELUDE_OVERLAP
            and b.nslice
            and g.tile_input
            and not pd.face_direct_im2col(g, b.block_k_tiles)
            and not pd._row_split(g, b.block_m_tiles, b.block_k_tiles)
        )
        with open(Path(__file__).parent / "runs" / "unit_engaged.jsonl", "a") as f:
            f.write(
                json.dumps(
                    dict(
                        test=os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0],
                        variant=variant,
                        engaged=engaged,
                        nslice=bool(b.nslice),
                        tile=g.tile_input,
                        bm=b.block_m_tiles,
                        bk=b.block_k_tiles,
                        ub=b.untilize_batch_rows,
                        slab=b.slab_tile_rows,
                        mblocks=b.max_m_blocks_per_core,
                        stride=g.stride,
                    )
                )
                + "\n"
            )
        return real_create(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op, "create_program_descriptor", recording_create)
    yield
    pd.select_blocking.cache_clear()

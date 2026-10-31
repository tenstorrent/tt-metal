# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness slice of the row_layout_finalize op copy (perf experiment): every case is checked against the torch
reference (the unit suite's run_case, PCC) AND bitwise against the real op on the same inputs (the finalize must be
bit-identical: same fp32 SFPU ops in the same order).

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/row_layout_finalize/test_slice.py
DRMS_FIN_LAYOUT=dense|packed|col|base selects the candidate layout (read when the op copy is imported).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # test-only; no global torch import under ttnn/
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as real_op
from ttnn.operations.distributed_rms_norm.perf_experiments.row_layout_finalize.op import (
    distributed_rms_norm as cand_op,
)

_UT = (
    Path(__file__).resolve().parents[6]
    / "tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm.py"
)
_spec = importlib.util.spec_from_file_location("drms_unit_suite", _UT)
T = importlib.util.module_from_spec(_spec)
sys.modules["drms_unit_suite"] = T
_spec.loader.exec_module(T)

RING = T.RING_MESHES[0] if T.RING_MESHES else (None, 1)
AX = T.LINEAR_AXES[0] if T.LINEAR_AXES else 1
HS, WS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED

# (id, shape, kwargs for run_case)
CASES = [
    ("focus_ring2", (1, 1, 640, 1792), dict(topology=ttnn.Topology.Ring, num_links=2)),
    ("decode_ring2", (1, 1, 32, 1792), dict(topology=ttnn.Topology.Ring, num_links=2)),
    ("prefill_R1_ring2", (1, 1, 2048, 1280), dict(topology=ttnn.Topology.Ring, num_links=2)),
    ("focus_ring1", (1, 1, 640, 1792), dict(topology=ttnn.Topology.Ring, num_links=1)),
    ("focus_linear", (1, 1, 640, 1792), dict()),
    ("focus_bf16dest", (1, 1, 640, 1792), dict(fp32_dest_acc_en=False)),
    ("prefill_R1_bf16dest", (1, 1, 2048, 1280), dict(fp32_dest_acc_en=False)),
    ("decode_bf8b_bf16dest", (1, 1, 32, 1792), dict(dtype=ttnn.bfloat8_b, fp32_dest_acc_en=False)),
    ("focus_fp32", (1, 1, 640, 1792), dict(dtype=ttnn.float32)),
    ("focus_bf8b", (1, 1, 640, 1792), dict(dtype=ttnn.bfloat8_b)),
    ("focus_no_gamma", (1, 1, 640, 1792), dict(has_gamma=False)),
    ("focus_height", (1, 1, 640, 1792), dict(memory_layout=HS)),
    ("focus_width", (1, 1, 640, 1792), dict(memory_layout=WS)),
    ("w1000_width_masked", (1, 1, 64, 1000), dict(memory_layout=WS)),
    ("batched_height", (2, 4, 128, 512), dict(memory_layout=HS)),
    ("ragged_17x50", (1, 1, 17, 50), dict()),
    ("batched_R1", (4, 1, 512, 512), dict()),
    ("tile_32x32", (1, 1, 32, 32), dict()),
]


_SEL = os.environ.get("DRMS_SLICE_CASES")
_CASES = [c for c in CASES if not _SEL or c[0] in _SEL.split(",")]


REF_DIR = Path(os.environ.get("DRMS_SLICE_REF_DIR", "/tmp/rlf_slice_ref"))


@pytest.mark.parametrize("mesh_shape,cluster_axis", [RING])
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_slice(mesh_shape, cluster_axis, case, mesh_device):
    """DRMS_SLICE_MODE=save: the real op only, outputs saved; =cmp: the candidate only, compared bitwise with the saved
    outputs (one op module per process: two module copies keep separate persistent exchange semaphores / landings
    and plan their CB regions blind to each other's live L1 buffers). Default: both in one process (real first)."""
    name, shape, kw = case
    mode = os.environ.get("DRMS_SLICE_MODE", "both")
    ops = {"save": [("real", real_op)], "cmp": [("cand", cand_op)]}.get(mode, [("real", real_op), ("cand", cand_op)])
    outs = {}
    for tag, op in ops:
        T.distributed_rms_norm = op
        try:
            _, _, out = T.run_case(mesh_device, shape, cluster_axis=cluster_axis, **kw)  # PCC vs torch inside
        finally:
            T.distributed_rms_norm = real_op
        outs[tag] = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
    if mode == "save":
        REF_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(outs["real"], REF_DIR / f"{name}.pt")
        return
    if mode == "cmp":
        outs["real"] = torch.load(REF_DIR / f"{name}.pt")
    for i, (x, y) in enumerate(zip(outs["cand"], outs["real"])):
        assert torch.equal(x, y), f"device {i}: candidate != real op ({(x != y).sum().item()} elements differ)"

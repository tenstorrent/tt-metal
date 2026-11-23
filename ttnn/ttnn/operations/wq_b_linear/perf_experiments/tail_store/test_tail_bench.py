# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tail_store whole-op A/B bench: wq_b_linear with this dir's descriptor copy + kernels.
TAIL="NAME=val;NAME2" -> TAIL_DEFINES; WQB_KNOBS as in _op_bench.py. Run under run_safe_pytest.sh --profile."""
import ast
import os

import pytest
import ttnn

import importlib

importlib.import_module("ttnn.operations.wq_b_linear")
import sys

op_mod = sys.modules["ttnn.operations.wq_b_linear.wq_b_linear"]
from ttnn.operations.wq_b_linear.perf_experiments.tail_store import pd_copy as pd


def _apply_knobs():
    for kv in filter(None, os.environ.get("WQB_KNOBS", "").split(",")):
        k, v = kv.split("=", 1)
        assert hasattr(pd, k), k
        setattr(pd, k, ast.literal_eval(v))
    pd.TAIL_DEFINES = {}
    for kv in filter(None, os.environ.get("TAIL", "").split(";")):
        k, _, v = kv.partition("=")
        pd.TAIL_DEFINES[k] = v or "1"
    print("TAIL_DEFINES", pd.TAIL_DEFINES)


@pytest.mark.use_module_device
def test_tail_bench(device):
    import torch  # local: no global torch import under ttnn/

    _apply_knobs()
    op_mod.create_program_descriptor = pd.create_program_descriptor
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    ref = (x.float() @ ttnn.to_torch(tw).float()).flatten()
    pccs = []
    for i in range(6):
        y = op_mod.wq_b_linear(tx, tw)
        got = ttnn.to_torch(y).float().flatten()
        pccs.append(torch.corrcoef(torch.stack([ref, got]))[0, 1].item())
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
    print("PCC", min(pccs), pccs)
    assert min(pccs) >= 0.9998

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op correctness of the port-spread W read: y from op_kernels/ must equal the op's y bit-exactly
(same W bytes, same math) and pass the focus case's PCC vs an fp64 reference."""

import os
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from eval.golden_tests.wq_b_linear.helpers import WEIGHT_SCALE
from ttnn.operations.wq_b_linear import wq_b_linear

PATCHED_DIR = Path(__file__).parent / os.environ.get("PS_OP_KERNELS", "op_kernels_rot2")
PCC = 0.9998


def _pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("case", ["default", "x_scaled", "w_scaled", "x_offset"])
def test_full_op_port_spread(device, case, monkeypatch):
    import torch

    torch.manual_seed(42)
    x = torch.randn(1, 1, 160, 1536)
    w = torch.randn(1536, 4096) * WEIGHT_SCALE
    if case == "x_scaled":
        x = x * 8.0
    elif case == "w_scaled":
        w = w * 10.0
    elif case == "x_offset":
        x = x + 0.5
    x, w = x.bfloat16(), w.bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    y_base = ttnn.to_torch(wq_b_linear(tx, tw))
    monkeypatch.setattr(pd, "KERNEL_DIR", PATCHED_DIR)
    y_new = ttnn.to_torch(wq_b_linear(tx, tw))
    ref = torch.matmul(x.double(), ttnn.to_torch(tw).double())
    pcc = _pcc(y_new, ref)
    print(f"{case}: bit-exact={torch.equal(y_new, y_base)} pcc={pcc:.6f}")
    assert torch.equal(y_new, y_base)
    assert pcc >= PCC

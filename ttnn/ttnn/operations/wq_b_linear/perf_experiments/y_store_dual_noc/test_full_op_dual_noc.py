# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op copy of wq_b_linear with the dual-NoC y store (kernels/ + dual_noc_descriptor.py in this dir).

YFO_CONFIGS="0:0,4:1" = list of Y_READER_SHARE_TENTHS:Y_SPLIT_ORDER (0:0 = the op's current y store), run in order.
YFO_KNOBS="K_REMAINDER_SPLIT=True" overrides other descriptor knobs. Per config: one call checked against torch
(PCC >= the golden limit), then 3 calls that must equal the first bit for bit.
"""

import os

import pytest
import ttnn

import dual_noc_descriptor as dpd
from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE
from eval.metrics import Tolerance, check_output
from ttnn.operations.wq_b_linear.wq_b_linear import validate

CONFIGS = os.environ.get("YFO_CONFIGS", "0:0,4:1").split(",")
PCC = TOLERANCES[(ttnn.bfloat8_b, False)]

for _kv in filter(None, os.environ.get("YFO_KNOBS", "").split(",")):
    _k, _v = _kv.split("=")
    setattr(dpd, _k, _v == "True" if isinstance(getattr(dpd, _k), bool) else type(getattr(dpd, _k))(_v))


def _run(tx, tw):
    cfg = validate(tx, tw)
    device = tx.device()
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    y = ttnn.generic_op([tx, tw, y], dpd.create_program_descriptor(tx, tw, y, cfg))
    ttnn.ReadDeviceProfiler(device)
    return y


@pytest.mark.parametrize("config", CONFIGS)
def test_full_op(device, config):
    import torch

    share, order = map(int, config.split(":"))
    dpd.Y_READER_SHARE_TENTHS, dpd.Y_SPLIT_ORDER = share, order
    torch.manual_seed(42)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * WEIGHT_SCALE).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch.matmul(x.to(torch.float64), ttnn.to_torch(tw).to(torch.float64))
    y = _run(tx, tw)
    first = ttnn.to_torch(y)
    check_output(
        y,
        expected,
        shape=[1, 1, 160, 4096],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=Tolerance(pcc=PCC),
    )
    pcc = torch.corrcoef(torch.stack([first.flatten().double(), expected.flatten()]))[0, 1].item()
    print(f"YFO config={config} pcc={pcc:.6f}")
    for _ in range(3):
        assert torch.equal(ttnn.to_torch(_run(tx, tw)), first)

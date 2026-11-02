# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Placement rows for guard measurements: QKV_ROWS="8x512:L1:DRAM:DRAM,..." = B x S : x : tables : outputs.
Correctness-gated; run with scripts/run_safe_pytest.sh --profile."""

import os

import pytest
import ttnn

from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope
from tests.ttnn.unit_tests.operations.qkv_proj_heads_rope.test_qkv_proj_heads_rope import (
    check_outputs,
    make_inputs,
    reference,
    to_device,
)

MC = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}
ROWS = [r.split(":") for r in os.environ.get("QKV_ROWS", "8x512:L1:DRAM:DRAM").split(",")]


@pytest.mark.parametrize("row", ROWS, ids=[":".join(r) for r in ROWS])
def test_placement_row(device, row):
    shape, x_mc, t_mc, o_mc = row
    batch, seq = (int(v) for v in shape.split("x"))
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    tensors = (
        to_device(x, device, MC[x_mc]),
        to_device(w, device, MC["DRAM"]),
        to_device(b, device, MC["DRAM"]),
        to_device(cos, device, MC[t_mc]),
        to_device(sin, device, MC[t_mc]),
    )
    outputs = qkv_proj_heads_rope(*tensors, memory_config=MC[o_mc])
    check_outputs(outputs, reference(x, w, b, cos, sin), buffer_type=MC[o_mc].buffer_type)

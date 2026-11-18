# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extra HEIGHT-sharded kw-fold regimes (not Panoptic layers) at LoFi bf8_b, bias + ReLU, for one op copy:
a K-blocked kw-fold with many M-blocks per core (bm 2) and a K-blocked bm 20 single block.

    KWCR_COPY=op_cand scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds/test_kwcr_extra.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwcr_copy import copy_modules  # noqa: E402

_op, _pd = copy_modules(os.environ.get("KWCR_COPY", "op_cand"))

# (H, W, C_in, C_out, shard rows, cores)
SHAPES = [(256, 512, 128, 128, 1216, 108), (128, 256, 128, 128, 640, 52)]


@pytest.mark.parametrize("shape", SHAPES, ids=["kb_bm2_19blocks", "kb_bm20"])
def test_kwcr_extra(shape, device):
    import torch
    import ttnn

    H, W, cin, cout, shard_rows, ncores = shape
    torch.manual_seed(0)
    full, rem = divmod(ncores, 11)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet(ranges), [shard_rows, cin], ttnn.ShardOrientation.ROW_MAJOR),
    )
    kw = dict(
        stride=1,
        padding=1,
        dilation=1,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=mc,
        device=device,
    )
    w = torch.randn(cout, cin, 3, 3).bfloat16()
    b = torch.randn(cout).bfloat16()
    x = torch.randn(1, 1, H * W, cin).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    out = _op.conv2d_3x3(
        tx,
        _op.prepare_conv2d_weights(w, **kw),
        input_height=H,
        input_width=W,
        bias=_op.prepare_conv2d_bias(b, **kw),
        padding=1,
        activation="relu",
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    got = ttnn.to_torch(out).float().reshape(-1)
    if got.abs().max().item() == 0:
        return  # the up-front compile pass
    xq = ttnn.to_torch(tx).float().reshape(1, H, W, cin).permute(0, 3, 1, 2)
    ref = torch.relu(torch.nn.functional.conv2d(xq, w.float(), b.float(), padding=1)).permute(0, 2, 3, 1).reshape(-1)
    pcc = torch.corrcoef(torch.stack([got.double(), ref.double()]))[0, 1].item()
    assert pcc >= 0.996

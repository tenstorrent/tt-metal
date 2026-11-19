# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiled single dispatches of 2-tile n-group kw-fold shapes the focus set does not reach (kw-fold forced, as in
test_conv2d_3x3_kw_fold.py), on op_copy/ (default) or op_base/ (FSR_OP=base). PCC-checked.

    FSR_OP=base scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_fused_side_rounds/test_fsr_extra_perf.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsr_copy import copy_modules  # noqa: E402

# (id, H, W, C_in, C_out, dilation, fidelity, cores)
CASES = [
    ("kblocked_n2_86", 64, 128, 256, 64, 1, "LoFi", 86),
    ("kblocked_n2_hifi4_86", 64, 128, 256, 64, 1, "HiFi4", 86),
    ("n10_kblocked_32", 32, 128, 64, 320, 1, "LoFi", 32),
    ("wholek_n2_d2_86", 64, 128, 64, 64, 2, "LoFi", 86),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fsr_extra_perf(case, device):
    import torch
    import ttnn
    from tests.ttnn.utils_for_testing import assert_with_pcc

    mod, pd = copy_modules()
    _, H, W, c_in, c_out, d, fidelity, cores = case
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    torch.manual_seed(0)
    tx = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tw = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    tb = torch.randn(c_out).to(torch.bfloat16)
    prep = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=mc,
        device=device,
    )
    saved = pd._kw_fold_pays
    pd._kw_fold_pays = lambda *a: True
    try:
        y = mod.conv2d_3x3(
            ttnn.from_torch(tx, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc),
            ttnn.operations.conv2d_3x3.prepare_conv2d_weights(tw, **prep),
            bias=ttnn.operations.conv2d_3x3.prepare_conv2d_bias(tb, **prep),
            input_height=H,
            input_width=W,
            padding=d,
            stride=1,
            dilation=d,
            activation="relu",
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True
            ),
        )
    finally:
        pd._kw_fold_pays = saved
    x = tx.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(torch.nn.functional.conv2d(x, tw.float(), tb.float(), padding=d, dilation=d))
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    assert_with_pcc(ref, ttnn.to_torch(y).float().reshape(ref.shape), 0.99)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Proposed kw-fold unit test (graduation): the final pass's bias placement, exactly. Zero weights, so every output
pixel is ReLU(bias) (or bias); the bias is a distinct small integer per output channel, exact in bf8_b. Catches a swapped
or mis-broadcast srcB row (b_0 .. b_15 must land on faces 0 and 2, b_16 .. b_31 on faces 1 and 3) and a misplaced
half-plane, which a random-weight PCC check can miss. Forced kw-fold; one and two n-groups, the halo exchange and the
multi-M-block recompute. Runs on op_base (HEAD) and op_grad (the graduation) through kfs_copy.py."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfs_copy import copy_modules  # noqa: E402

# (id, H, W, C_in, C_out, cores, relu)
CASES = [
    ("n1_xchg", 64, 128, 64, 32, 52, True),
    ("n2_fused_xchg", 64, 128, 64, 64, 86, True),
    ("n4_two_groups_no_relu", 32, 64, 64, 128, 16, False),
    ("n2_multiblock", 64, 128, 64, 64, 8, True),
]


@pytest.mark.parametrize("variant", os.environ.get("KFS_BIAS_VARIANTS", "base,grad").split(","))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_kfs_bias_layout(case, variant, device, monkeypatch):
    import torch

    import ttnn

    device.clear_program_cache()
    op_mod, pd_mod = copy_modules(variant)
    _, H, W, c_in, c_out, cores, relu = case
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    taken = []
    kw_fold_mode = pd_mod.kw_fold_mode
    monkeypatch.setattr(pd_mod, "_kw_fold_pays", lambda *a: True)
    monkeypatch.setattr(pd_mod, "kw_fold_mode", lambda *a: taken.append(kw_fold_mode(*a)) or taken[-1])
    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    torch_weight = torch.zeros(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = ((torch.arange(c_out) % 32) - 15).to(torch.bfloat16)  # -15 .. 16: exact in bf8_b
    kwargs = dict(
        stride=1,
        padding=1,
        dilation=1,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = op_mod.conv2d_3x3(
        tt_input,
        op_mod.prepare_conv2d_weights(torch_weight, **kwargs),
        input_height=H,
        input_width=W,
        bias=op_mod.prepare_conv2d_bias(torch_bias, **kwargs),
        padding=1,
        stride=1,
        dilation=1,
        activation="relu" if relu else None,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True
        ),
    )
    assert True in taken, f"kw-fold not taken: {taken}"
    got = ttnn.to_torch(tt_output).float().reshape(H * W, c_out)
    want = torch_bias.float().clamp(min=0) if relu else torch_bias.float()
    bad = (got != want.unsqueeze(0)).any(0).nonzero().flatten().tolist()
    assert not bad, f"output channels not equal to their bias: {bad[:16]}"

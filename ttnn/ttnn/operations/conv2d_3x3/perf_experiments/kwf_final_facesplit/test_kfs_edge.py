# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Face-split ACC edge cases, forced kw-fold, base (op_base) vs graduation (op_grad, KFS_EDGE_VARIANT): bit-exact vs
the base output and PCC vs torch. Few cores, many tile-rows per core: the bm search walks candidates whose ACC
half-plane holds more than 64K fp32 datums (the unpacker's 16-bit Tile_x_dim; the final pass then takes two srcA Z
steps of HALF / 8 datums). One M-block of 128 tile-rows (128 x 256 x 32 -> 32 on 8 cores) does not run on HEAD either:
its fill plan trips the descriptor's KWF_FILL_MAX_GROUPS assert, so it is not listed. The two-step path itself is
checked on the focus cases by forcing it (test_kfs_ab.py variant zk2: bit-identical)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfs_copy import copy_modules  # noqa: E402

# (id, H, W, C_in, C_out, dilation, cores)
CASES = [
    ("big_half_n2", 128, 256, 32, 64, 1, 8),
    ("mid_half", 64, 256, 32, 32, 1, 8),
]
_OUT = {}


@pytest.mark.parametrize("variant", ["base", os.environ.get("KFS_EDGE_VARIANT", "grad")])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_kfs_edge(case, variant, device, monkeypatch):
    import torch

    import ttnn
    from tests.ttnn.utils_for_testing import assert_with_pcc

    device.clear_program_cache()
    op_mod, pd_mod = copy_modules(variant)
    cid, H, W, c_in, c_out, d, cores = case
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
    taken, planes = [], []
    kw_fold_mode = pd_mod.kw_fold_mode
    monkeypatch.setattr(pd_mod, "_kw_fold_pays", lambda *a: True)
    monkeypatch.setattr(pd_mod, "kw_fold_mode", lambda *a: taken.append(kw_fold_mode(*a)) or taken[-1])
    plane_bytes = pd_mod._kw_fold_plane_bytes
    monkeypatch.setattr(pd_mod, "_kw_fold_plane_bytes", lambda g, bm: planes.append(plane_bytes(g, bm)) or planes[-1])
    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    torch_weight = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    kwargs = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True
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
        padding=d,
        stride=1,
        dilation=d,
        activation="relu",
        compute_kernel_config=compute_config,
    )
    assert True in taken, f"kw-fold not taken: {taken}"
    out = ttnn.to_torch(tt_output).float()
    print(
        f"EDGE {cid} {variant}: plane bytes {sorted(set(planes))} (half / 4 = {[p // 8 for p in sorted(set(planes))]})"
    )
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=d, dilation=d))
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    assert_with_pcc(ref, out.reshape(ref.shape), 0.99)
    first = _OUT.setdefault(cid, (variant, out))
    if first[0] != variant:
        diff = int((first[1] != out).sum())
        print(f"EDGE {cid} {variant} vs {first[0]}: {diff} differing elements")
        assert diff == 0

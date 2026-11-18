# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""kw-fold with more than one n-group (N > sbw output column tiles), forced on, on this experiment's copy: the weight
sub-rounds ((n-group, kw) order), one ACC plane per pass, a final pass per n-group, and the bias read before the first
final pass, in each weight-stream form (servers, in-grid multicast sender, own stream) and both weight residencies
(K-blocked sub-rounds double-buffered, whole-K resident), several M-blocks per core, a ragged last core, dilation 2,
plane widths 1 / 3 / 4 (C_out 160 / 192 / 256), HiFi4 with bf16 activations. Checked against a fp32 torch conv (PCC)
and bit-exactly against the base op's forced kw-fold where that fits L1 (same per-element accumulation order).

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_ngroup_outer/test_kno_kwf_extra.py
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kno_copy import copy_modules  # noqa: E402

# (id, H, W, C_in, C_out, dilation, act dtype, weight dtype, fidelity, relu, cores, weight stream)
CASES = [
    ("n8_kblocked_servers", 32, 64, 128, 256, 1, "bf8", "bf8", "LoFi", True, 16, "servers"),
    ("n8_kblocked_sender", 32, 64, 128, 256, 1, "bf8", "bf8", "LoFi", True, 16, "sender"),
    ("n8_whole_k", 32, 64, 32, 256, 1, "bf8", "bf8", "LoFi", False, 16, "servers"),
    ("n5_sbw1_whole_k", 32, 64, 64, 160, 1, "bf8", "bf8", "LoFi", True, 16, "servers"),
    ("n6_sbw3_kblocked", 32, 64, 128, 192, 1, "bf8", "bf8", "LoFi", True, 16, "servers"),
    ("n8_multiblock", 64, 128, 64, 256, 1, "bf8", "bf8", "LoFi", True, 8, "servers"),
    ("n8_multiblock_sender", 64, 128, 64, 256, 1, "bf8", "bf8", "LoFi", True, 8, "sender"),
    ("n8_ragged", 48, 64, 64, 256, 1, "bf8", "bf8", "LoFi", True, 20, "servers"),
    ("n8_d2", 64, 64, 64, 256, 2, "bf8", "bf8", "LoFi", True, 32, "servers"),
    ("n8_one_core_own_stream", 32, 32, 64, 256, 1, "bf8", "bf8", "LoFi", True, 1, "servers"),
    ("n8_bf16_hifi4", 32, 64, 64, 256, 1, "bf16", "bf8", "HiFi4", False, 16, "servers"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_kno_kwf_extra(case, device, monkeypatch):
    import torch
    import ttnn
    from tests.ttnn.utils_for_testing import assert_with_pcc
    from ttnn.operations.conv2d_3x3 import conv2d_3x3 as base_op, prepare_conv2d_bias, prepare_conv2d_weights

    dtypes = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    _, H, W, c_in, c_out, d, act, wdt, fidelity, relu, cores, stream = case
    dtype, weight_dtype = dtypes[act], dtypes[wdt]
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
    op_mod, pd_mod = copy_modules()
    base_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    plans = {}
    for name, pd in (("copy", pd_mod), ("base", base_pd)):
        monkeypatch.setattr(pd, "_kw_fold_pays", lambda *a: True)
        if stream == "sender":
            monkeypatch.setattr(pd, "WEIGHT_SERVERS", False)
        real_create = pd.create_program_descriptor

        def create(*a, _real=real_create, _name=name, _pd=pd, **k):
            desc = _real(*a, **k)
            defines = [dict(kd.defines) for kd in desc.kernels]
            plans[_name] = any("CONV2D_3X3_KW_FOLD" in df for df in defines)
            if _name == "copy":
                comp = [df for df in defines if "KWF_SBW" in df]
                plans["copy_shape"] = (
                    {k: comp[0][k] for k in ("KWF_SBW", "KWF_SBH", "KWF_KBLOCKED", "KWF_NOINLINE")} if comp else None
                )
                plans["kernels"] = len(desc.kernels)
                plans["blocks"] = (k["blocking"].block_m_tiles, k["blocking"].core_m_tiles)
            return desc

        mod = op_mod if name == "copy" else importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
        monkeypatch.setattr(mod, "create_program_descriptor", create)

    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    torch_weight = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    kwargs = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True
    )
    outs = {}
    for name, fn in (("copy", op_mod.conv2d_3x3), ("base", base_op)):
        tt_input = ttnn.from_torch(
            torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        y = fn(
            tt_input,
            prepare_conv2d_weights(torch_weight, **kwargs),
            input_height=H,
            input_width=W,
            bias=prepare_conv2d_bias(torch_bias, **kwargs),
            padding=d,
            stride=1,
            dilation=d,
            activation="relu" if relu else None,
            compute_kernel_config=compute_config,
        )
        outs[name] = ttnn.to_torch(y).float()
        # One call's shards live at a time (the L1 budget counts the call's own input + output only).
        ttnn.deallocate(y)
        ttnn.deallocate(tt_input)
    print("KNO_EXTRA", case[0], plans)
    assert plans["copy"], "kw-fold not taken by the copy"

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    pcc = 0.997 if dtype == weight_dtype == ttnn.bfloat16 else 0.99
    assert_with_pcc(ref, outs["copy"].reshape(ref.shape), pcc)
    if plans["base"]:
        diff = (outs["copy"] - outs["base"]).abs()
        print(
            "KNO_EXTRA_VS_BASE_KWF",
            case[0],
            "max diff",
            diff.max().item(),
            "identical",
            (diff == 0).float().mean().item(),
        )
        assert torch.equal(outs["copy"], outs["base"]), "copy differs from the base op's kw-fold"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact gate of the padded slab (raw untilize destinations + raw tilize sources) against the base op path.

Each BLOCK-sharded TILE shape runs twice on this dir's op copy, PADDED_SLAB False (the committed tap-direct prelude)
and True, and the two outputs must be identical bit for bit (plus a PCC sanity check vs torch). The shapes cover every
untilize mode the padded path drives (1-wide, whole-row ct 2..4, half-row ct 5..8, chunked ct > 8), image rows of 1,
2, 3 and 4 tile-rows, gaps of 1..40 pixels (taps wholly inside the padding), ragged last cores and bf16 / bf8_b.
A shape whose blocking does not take the padded slab is recorded and skipped (the run log says which).

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tapdirect_cheap_publish/test_tdcp_bitexact.py
"""

import importlib
import json
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.ablation_r8.merge_gates.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
LOG = HERE / "runs" / "bitexact.jsonl"

# ((H, W, C_in), C_out, p == d), (shard rows, shard width)
SHAPES = [
    # res4.x itself, and a 2-row-tile image (tpr 1)
    (((32, 64, 256), 256, 1), (256, 32)),
    (((16, 32, 256), 128, 1), (64, 32)),
    # tpr 3 / 4, 1-wide
    (((16, 96, 128), 128, 2), (192, 32)),
    (((8, 128, 128), 64, 1), (128, 32)),
    # whole-row ct 2, 3, 4
    (((32, 64, 256), 256, 2), (224, 64)),
    (((16, 64, 192), 128, 3), (128, 96)),
    (((16, 64, 256), 128, 1), (128, 128)),
    # half-row ct 5, 6, 8
    (((16, 64, 320), 64, 2), (128, 160)),
    (((32, 64, 384), 128, 6), (256, 192)),
    (((16, 64, 512), 64, 4), (128, 256)),
    # chunked ct 9
    (((8, 64, 576), 64, 1), (64, 288)),
    # gaps wider than a tap block (d = 40): taps wholly in the padding; and d = 18 (aspp.convs.3-like)
    (((16, 64, 128), 64, 40), (128, 32)),
    (((32, 64, 256), 128, 18), (256, 64)),
    # ragged: 7 shard rows over 44 image rows
    (((44, 64, 128), 128, 1), (416, 32)),
]


def _id(case):
    ((H, W, c), co, d), (sr, sw) = case
    return f"H{H}_W{W}_Ci{c}_Co{co}_d{d}_sh{sr}x{sw}"


def _run(device, case, dtype, padded, monkeypatch):
    import torch

    ((H, W, c_in), c_out, d), (shard_rows, shard_width) = case
    monkeypatch.setattr(pd, "PADDED_SLAB", padded)
    pd.select_blocking.cache_clear()
    torch.manual_seed(0)
    num_px = H * W
    rows, cols = -(-num_px // shard_rows), -(-c_in // shard_width)
    grid = device.compute_with_storage_grid_size()
    if cols > grid.x or rows > grid.y:
        pytest.skip(f"{cols}x{rows} shard grid does not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))]),
            [shard_rows, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    kwargs = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs)
    torch_input = torch.randn(1, 1, num_px, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    taken = {}
    real_create = pd.create_program_descriptor

    def create(*a, **k):
        desc = real_create(*a, **k)
        taken["on"] = any("CONV2D_3X3_PADDED_SLAB" in dict(kd.defines) for kd in desc.kernels)
        return desc

    monkeypatch.setattr(op, "create_program_descriptor", create)
    tt_output = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=d,
        stride=1,
        dilation=d,
        activation="relu",
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    out = ttnn.to_torch(tt_output).float().reshape(1, 1, num_px, c_out)
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=d, dilation=d))
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return out, ref, taken


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8b", "bf16"])
@pytest.mark.parametrize("case", SHAPES, ids=[_id(c) for c in SHAPES])
def test_tdcp_bitexact(case, dtype, device, monkeypatch):
    import torch

    from tests.ttnn.utils_for_testing import comp_pcc

    out_base, ref, _ = _run(device, case, dtype, False, monkeypatch)
    out_pad, _, info = _run(device, case, dtype, True, monkeypatch)
    taken = bool(info and info.get("on"))
    same = bool(torch.equal(out_base, out_pad))
    _, pcc = comp_pcc(ref, out_pad, 0.99)
    rec = dict(case=_id(case), dtype=str(dtype), padded=info, taken=taken, bit_identical=same, pcc=float(pcc))
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if not taken:
        pytest.skip(f"padded slab not taken: {info}")
    assert same, rec
    assert pcc >= 0.99, rec

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: split the HEIGHT-sharded TILE slab fill of conv2d_3x3 by owner direction (perf experiment).

Runs a private copy of conv2d_3x3 (op_copy/), switched by module knobs:
  base    the committed TILE fill: the reader (NCRISC, NoC0) reads every owner shard's input tile-rows of each
          cb_input_tiles batch, including the higher-index shards', whose NoC0 read responses wrap around the ring.
  split   the reader reads its own and lower shards' tile-rows; for a batch holding higher shards' rows it first
          hands the reserved batch to the writer (fill_go), which reads those rows on NoC1 (short way back) into the
          batch's ring slot and raises fill_done; the reader pushes the batch once both halves landed. The writer
          serves hand-overs while it polls for a free cb_weights slot and before each (blocking) multicast receive.
  split_poll  split, and a weight-multicast receiver also serves hand-overs while it waits for a round's data (raw
          receive: ack the fixed sender, poll the data-ready semaphore); the sender polls its acks the same way.
  split_poll_rx  split_poll on the pure multicast receivers only: the sender and the own-stream cores (whose writer
          blocks in weight DRAM reads) keep the whole fill on the reader.

Precision contract is each case's own (fp32 DEST, its fidelity, bf8_b/bf16): every variant runs it unchanged, and the
two variants move the same bytes into the same L1 slots, so their outputs must be bit-identical.

One dispatch per test_tfs_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  TFS_CASES=focus TFS_VARIANTS=base,split scripts/run_safe_pytest.sh --profile --run-all <this file> -k test_tfs_perf
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import tfs_load  # noqa: E402

opc, pd, op_mod = tfs_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"

KNOB_DEFAULTS = dict(TILE_FILL_SPLIT_NOCS=False, TILE_FILL_POLL_RECEIVE=False, TILE_FILL_RECEIVERS_ONLY=False)
VARIANTS = {
    "base": {},
    "split": dict(TILE_FILL_SPLIT_NOCS=True),
    "split_poll": dict(TILE_FILL_SPLIT_NOCS=True, TILE_FILL_POLL_RECEIVE=True),
    "split_poll_rx": dict(TILE_FILL_SPLIT_NOCS=True, TILE_FILL_POLL_RECEIVE=True, TILE_FILL_RECEIVERS_ONLY=True),
}

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _grid(num_cores, grid_x=11):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _height(shard, num_cores):
    return ttnn.MemoryConfig(
        HS, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(num_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    )


# The HEIGHT perf harness's TILE layers (test_conv2d_3x3_perf.py HEIGHT_SHAPES), in their harness config: bf8_b
# weights, bias + ReLU, the op's default compute config.
# (layer, ((H, W, C_in), C_out, padding, stride), dtype, shard_shape, num_cores)
HEIGHT_TILE = [
    ("stem_conv3", ((256, 512, 64), 128, 1, 1), ttnn.bfloat8_b, [1216, 64], 110),
    ("res2_conv2", ((128, 256, 64), 64, 1, 1), ttnn.bfloat8_b, [320, 64], 103),
    ("res3_0_conv2", ((128, 256, 128), 128, 1, 2), ttnn.bfloat8_b, [384, 128], 86),
    ("res3_conv2", ((64, 128, 128), 128, 1, 1), ttnn.bfloat8_b, [96, 128], 86),
    ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), ttnn.bfloat8_b, [320, 288], 103),
    ("stem_conv2", ((256, 512, 64), 64, 1, 1), ttnn.bfloat8_b, [1216, 64], 110),
    ("sem_res3_fuse_conv0", ((64, 128, 320), 256, 1, 1), ttnn.bfloat8_b, [96, 320], 86),
    ("inst_res3_fuse_conv0", ((64, 128, 320), 128, 1, 1), ttnn.bfloat8_b, [96, 320], 86),
    ("sem_res3_fuse_conv1", ((64, 128, 256), 256, 1, 1), ttnn.bfloat8_b, [96, 256], 86),
    ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), ttnn.bfloat8_b, [320, 160], 103),
    ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), ttnn.bfloat8_b, [320, 256], 103),
    ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), ttnn.bfloat8_b, [320, 128], 103),
    ("center_head1", ((128, 256, 128), 32, 1, 1), ttnn.bfloat8_b, [320, 128], 103),
]


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    (focus,) = [
        c
        for c in LOOSE_CASES
        if c["extras"]["layer"].startswith("semantic res2.fuse_conv.1") and "[perf]" in c["extras"]["layer"]
    ]
    (H, W, c_in), c_out, padding, stride, dilation = focus["inputs"]
    out["focus"] = dict(
        shape=((H, W, c_in), c_out, padding, stride, dilation),
        dtype=focus["dtype"],
        layout=focus["layout"],
        weight_dtype=focus["weight_dtype"],
        mem=lambda: _height(list(focus["extras"]["shard_shape"]), 103),
        bias=focus["bias_mode"] == "with_bias",
        relu=focus["activation"] == "relu",
        fidelity=focus["extras"].get("math_fidelity"),
        fp32_dest=focus["fp32_dest_acc_en"],
        pcc=focus["extras"]["tolerance"][0],
        golden=True,
    )
    for layer, ((H, W, c_in), c_out, padding, stride), dtype, shard, cores in HEIGHT_TILE:
        out[layer] = dict(
            shape=((H, W, c_in), c_out, padding, stride, 1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            weight_dtype=ttnn.bfloat8_b,
            mem=(lambda shard=shard, cores=cores: _height(shard, cores)),
            bias=True,
            relu=True,
            fidelity=None,
            fp32_dest=None,
            pcc=0.99,
            golden=False,
        )
    # Interleaved TILE (the split never applies: no owner shards).
    out["dram_tile_bf8b"] = dict(
        shape=((64, 128, 128), 128, 1, 1, 1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
        bias=True,
        relu=False,
        fidelity=None,
        fp32_dest=None,
        pcc=0.99,
        golden=False,
    )
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)


def run_case(case, device, seed=0):
    import torch

    torch.manual_seed(seed)
    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    memory_config = case["mem"]()
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=memory_config,
        device=device,
    )
    if case["golden"]:
        from eval.golden_tests.conv2d_3x3 import helpers

        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if case["bias"] else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    else:
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias"] else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if case["bias"] else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=memory_config
    )
    config = None
    if case["fidelity"] is not None:
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fidelity"], fp32_dest_acc_en=case["fp32_dest"], math_approx_mode=False
        )
    tt_output = opc.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if case["relu"] else None,
        compute_kernel_config=config,
    )
    if case["golden"]:
        ref = helpers.pytorch_conv2d_3x3(
            torch_input,
            torch_weight,
            input_height=H,
            input_width=W,
            bias=torch_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation="relu" if case["relu"] else None,
        ).float()
        return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if case["relu"]:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref


def _pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


PERF_CASES = _selected("TFS_CASES", list(CASES))
PERF_VARIANTS = _selected("TFS_VARIANTS", list(VARIANTS))


@pytest.mark.parametrize("variant", PERF_VARIANTS)
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_tfs_perf(case_id, variant, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch."""
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    out, ref = run_case(case, device)
    pcc = _pcc(out, ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(dict(kind="perf", case=case_id, variant=variant, pcc=pcc, tag=os.environ.get("TFS_TAG", "")))
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != "base"])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_tfs_identical(case_id, variant, device, monkeypatch):
    """Same inputs through base and the candidate: the outputs must be bit-identical (same bytes, same slots)."""
    import torch

    case = CASES[case_id]
    outs = {}
    for v in ("base", variant):
        apply_variant(v, monkeypatch)
        outs[v], ref = run_case(case, device, seed=1)
    same = bool(torch.equal(outs["base"], outs[variant]))
    pcc = _pcc(outs[variant], ref)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(dict(kind="identical", case=case_id, variant=variant, identical=same, pcc=pcc)) + "\n")
    assert (
        same
    ), f"{case_id}/{variant}: output differs from base (max |diff| {(outs['base'] - outs[variant]).abs().max()})"
    assert pcc >= case["pcc"]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the TILE halo slab as a ring across M-blocks (perf experiment ring_slab_reuse, round 4).

Runs a private copy of conv2d_3x3 (op_copy/), switched by the descriptor knob RING_SLAB:
  base  the op's current TILE slab: per M-block the reader plans the block's three bands (tile_slab_plan, bands that
        do not share tile-rows are compacted next to each other), streams slab_tile_rows (a whole number of untilize
        batches, unread rows past the plan included) through cb_input_tiles, compute untilizes all of them into
        cb_halo_slab, and the reader pops the whole slab after the block's gather.
  ring  cb_halo_slab is a FIFO ring of consecutive input tile-rows across the core's M-blocks: M-block m streams and
        untilizes only its window's rows past M-block m-1's window (the first block: its whole window), the reader
        pops only the rows below M-block m+1's window, every band gathers at one base and a slab read that crosses
        the ring end is split. Ring capacity = the largest M-block window, taken only from the L1 left at the solved
        blocking (same bm, bk, untilize batch, im2col depth and center-tap mode), else base.

Precision contract is each case's own (fp32 DEST, its fidelity, bf8_b/bf16): every variant runs it unchanged. Both
variants untilize the same input tiles into slab pixels the gather reads at the same image pixel, so their outputs must
be bit-identical (test_rs_identical, torch.equal).

One dispatch per test_rs_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  RS_CASES=focus RS_VARIANTS=base,ring RS_TAG=t1 scripts/run_safe_pytest.sh --profile --run-all \\
      <this file> -k test_rs_perf -p no:cacheprovider --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import rs_load  # noqa: E402

opc, pd, op_mod = rs_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"
_OFF = dict(RING_SLAB=False, RING_SLAB_SINGLE=False, RING_IN_SOLVE=False, TEST_BM_CAP=0)
VARIANTS = {
    "base": dict(_OFF),
    "ring": dict(_OFF, RING_SLAB=True),
    # Menu: the ring also where every core runs one M-block.
    "ring_all": dict(_OFF, RING_SLAB=True, RING_SLAB_SINGLE=True),
    # Menu: the ring counted inside the L1 solve (no fallback to the slab; the solve shrinks other buffers for it).
    "ring_solve": dict(_OFF, RING_SLAB=True, RING_SLAB_SINGLE=True, RING_IN_SOLVE=True),
    # Correctness coverage only: bm capped at 1 (many M-blocks per core, image-edge windows).
    "base_bm1": dict(_OFF, TEST_BM_CAP=1),
    "ring_bm1": dict(_OFF, RING_SLAB=True, RING_SLAB_SINGLE=True, TEST_BM_CAP=1),
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


# The HEIGHT perf harness's TILE layers (tests/.../test_conv2d_3x3_perf.py HEIGHT_SHAPES) in its config: bf8_b act and
# weights, bias + ReLU, the op's default compute config.
HEIGHT_TILE = [
    ("stem_conv3", ((256, 512, 64), 128, 1, 1), [1216, 64], 110),
    ("res2_conv2", ((128, 256, 64), 64, 1, 1), [320, 64], 103),
    ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), [320, 288], 103),
    ("stem_conv2", ((256, 512, 64), 64, 1, 1), [1216, 64], 110),
    ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), [320, 160], 103),
    ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), [320, 256], 103),
    ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), [320, 128], 103),
    ("center_head1", ((128, 256, 128), 32, 1, 1), [320, 128], 103),
]

# Interleaved (DRAM) TILE shapes: the perf harness's PERF_SHAPES, plus shapes picked to land on more than one M-block
# per core. bf16 weights, bias, no ReLU (harness config).
INTERLEAVED = [
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
    ((128, 256, 512), 256, 1),
    ((256, 512, 128), 128, 1),
    ((64, 128, 1024), 512, 1),
]
TILE_DTYPES = [("bf16", ttnn.bfloat16), ("bf8b", ttnn.bfloat8_b)]


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
    for layer, ((H, W, c_in), c_out, padding, stride), shard, cores in HEIGHT_TILE:
        out[layer] = dict(
            shape=((H, W, c_in), c_out, padding, stride, 1),
            dtype=ttnn.bfloat8_b,
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
    for (H, W, c_in), c_out, padding in INTERLEAVED:
        for tag, dtype in TILE_DTYPES:
            out[f"dram_H{H}_W{W}_Ci{c_in}_Co{c_out}_{tag}"] = dict(
                shape=((H, W, c_in), c_out, padding, 1, 1),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                weight_dtype=ttnn.bfloat16,
                mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
                bias=True,
                relu=False,
                fidelity=None,
                fp32_dest=None,
                pcc=0.995 if dtype == ttnn.bfloat16 else 0.99,
                golden=False,
            )
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


LAST = {}


def apply_variant(variant, monkeypatch):
    for k, v in VARIANTS[variant].items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nb=blocking.max_m_blocks_per_core,
            ub=blocking.untilize_batch_rows,
            slab_rows=blocking.slab_tile_rows,
            im2col_depth=blocking.im2col_rm_depth_rows,
            center=int(blocking.center_mode),
            ring=bool(blocking.ring),
            foot_kb=round(blocking.footprint_bytes / 1024, 1),
        )
        return real(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op_mod, "create_program_descriptor", capture)


def run_case(case, device, seed=0, dispatch=True):
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
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


PERF_CASES = _selected("RS_CASES", ["focus"])
PERF_VARIANTS = _selected("RS_VARIANTS", ["base", "ring"])


@pytest.mark.parametrize("variant", PERF_VARIANTS)
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_rs_perf(case_id, variant, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch."""
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    out, ref = run_case(case, device)
    pcc = _pcc(out, ref)
    if os.environ.get("RS_FLUSH_PROFILER") == "1":
        # Long profiled sessions: flush the device markers per dispatch.
        ttnn.ReadDeviceProfiler(device)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(kind="perf", case=case_id, variant=variant, pcc=pcc, tag=os.environ.get("RS_TAG", ""), **LAST)
            )
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


BASE_VARIANT = os.environ.get("RS_BASE", "base")


@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != BASE_VARIANT])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_rs_identical(case_id, variant, device, monkeypatch):
    """Same inputs through the base (RS_BASE) and the candidate: the outputs must be bit-identical (and PCC-gated)."""
    import torch

    case = CASES[case_id]
    outs, blk = {}, {}
    for v in (BASE_VARIANT, variant):
        apply_variant(v, monkeypatch)
        outs[v], ref = run_case(case, device, seed=1)
        blk[v] = dict(LAST)
    same = bool(torch.equal(outs[BASE_VARIANT], outs[variant]))
    pcc = _pcc(outs[variant], ref)
    diff = float((outs[BASE_VARIANT] - outs[variant]).abs().max())
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(
                    kind="identical",
                    case=case_id,
                    variant=variant,
                    identical=same,
                    pcc=pcc,
                    max_diff=diff,
                    blocking=blk,
                )
            )
            + "\n"
        )
    assert same, f"{case_id}/{variant}: output differs from base (max |diff| {diff}) {blk}"
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc}"

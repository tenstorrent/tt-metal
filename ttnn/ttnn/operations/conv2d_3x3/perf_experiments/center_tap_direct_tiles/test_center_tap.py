# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: center-tap direct in0 tiles for conv2d_3x3 (perf experiment, round 3).

Runs a private copy of conv2d_3x3 (op_copy/), switched by the descriptor's module knobs:
  base    the committed path: every K-block's in0 goes untilize (input tiles -> bf16 slab) -> tap-major gather
          (slab -> cb_im2col_rm) -> tilize (-> cb_im2col_tiles).
  center  stride 1, padding == dilation, W % 32 == 0 on a TILE input: the kw = 1 taps are whole input tile-rows
          shifted by (kh*d - p) * W / 32 tile-rows (zero tile-rows outside the image).
            one tap per K-block (mode 1): the reader copies the kw = 1 K-blocks into cb_center_tiles, compute skips
              their tilize and matmul_block's In0SourceFn reads them from there;
            one kernel row per K-block, bm >= 2 (mode 2): the reader writes each tile-row's kw = 1 tiles straight into
              their cb_im2col_tiles slots, compute's tilize reserves + pushes those slots unpacked.
  center_kb  mode 1 only.

Precision contract is each case's own (fp32 DEST, its fidelity, bf8_b act + weights, bias + ReLU): every variant runs
it unchanged. Bar: bit-identical output to base (torch.equal), not just PCC.

One dispatch per test_ctd_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  CTD_CASES=focus CTD_VARIANTS=base,center CTD_TAG=t1 \\
    scripts/run_safe_pytest.sh --profile --run-all <this file> -k test_ctd_perf -p no:cacheprovider --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import ctd_load  # noqa: E402

opc, pd, op_mod = ctd_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"

KNOB_DEFAULTS = dict(
    CENTER_TAP_DIRECT=False, CENTER_TAP_SLOTS=True, CENTER_SLOTS_SINGLE=False, CENTER_KB_IN_SOLVE=False
)
VARIANTS = {
    "base": {},
    # Recommended: both modes (center K-blocks with one tap per K-block, center slots with one kernel row per K-block
    # and bm >= 2), each only where bm * bk more in0 tiles fit the L1 left at the solved blocking.
    "center": dict(CENTER_TAP_DIRECT=True),
    # Menu: center slots also single-buffered where no L1 is left.
    "center_single": dict(CENTER_TAP_DIRECT=True, CENTER_SLOTS_SINGLE=True),
    # Menu: mode 1's cb_center_tiles counted inside the solve (may shrink the im2col ring).
    "center_kb_solve": dict(CENTER_TAP_DIRECT=True, CENTER_KB_IN_SOLVE=True),
}

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
LOFI = ttnn.MathFidelity.LoFi


def _grid(num_cores, grid_x=11):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _block(shard, grid_xy):
    gx, gy = grid_xy
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# The BLOCK perf-focus layers (feature_spec LOOSE_CASES "[perf]", LoFi) and the LoFi res4 guard layer.
# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y))
BLOCK_LOFI = [
    ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10)),
    ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10)),
    ("aspp_convs_1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8)),
    ("aspp_convs_2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8)),
    ("res4_conv2_lofi", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8)),
]


def _height(shard, num_cores):
    return ttnn.MemoryConfig(
        HS, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(num_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    )


# The HEIGHT guard harness's TILE layers (tests/.../test_conv2d_3x3_perf.py HEIGHT_SHAPES), in its config: bf8_b act
# and weights, bias + ReLU, the op's default compute config.
# (layer, ((H, W, C_in), C_out, padding, stride), shard_shape, num_cores)
HEIGHT_TILE = [
    ("stem_conv3", ((256, 512, 64), 128, 1, 1), [1216, 64], 110),
    ("res2_conv2", ((128, 256, 64), 64, 1, 1), [320, 64], 103),
    ("res3_0_conv2", ((128, 256, 128), 128, 1, 2), [384, 128], 86),
    ("res3_conv2", ((64, 128, 128), 128, 1, 1), [96, 128], 86),
    ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), [320, 288], 103),
    ("stem_conv2", ((256, 512, 64), 64, 1, 1), [1216, 64], 110),
    ("sem_res3_fuse_conv0", ((64, 128, 320), 256, 1, 1), [96, 320], 86),
    ("inst_res3_fuse_conv0", ((64, 128, 320), 128, 1, 1), [96, 320], 86),
    ("sem_res3_fuse_conv1", ((64, 128, 256), 256, 1, 1), [96, 256], 86),
    ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), [320, 160], 103),
    ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), [320, 256], 103),
    ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), [320, 128], 103),
    ("center_head1", ((128, 256, 128), 32, 1, 1), [320, 128], 103),
]


def _cases():
    out = {
        # semantic res2.fuse_conv.1 @ LoFi [perf] (feature_spec LOOSE_CASES): golden builders (randn), LoFi.
        "focus": dict(
            shape=((128, 256, 256), 256, 1, 1, 1),
            mem=lambda: _height([320, 256], 103),
            fidelity=LOFI,
            pcc=0.996,
        ),
        # The focus layer at LoFi with C_in 288 (9 tiles): the other one-tap-K-block HEIGHT layer.
        "sem_res2_fuse_conv0_lofi": dict(
            shape=((128, 256, 288), 256, 1, 1, 1),
            mem=lambda: _height([320, 288], 103),
            fidelity=LOFI,
            pcc=0.996,
        ),
    }
    # Interleaved (DRAM) TILE input: the center tiles come page by page through the accessor. bf8_b and bf16.
    for tag, dtype in (("bf8b", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)):
        out[f"dram_tile_{tag}"] = dict(
            shape=((64, 128, 128), 128, 1, 1, 1),
            mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
            fidelity=None,
            pcc=0.99,
            dtype=dtype,
        )
    for layer, ((H, W, c_in), c_out, padding, dilation), shard, grid_xy in BLOCK_LOFI:
        out[layer] = dict(
            shape=((H, W, c_in), c_out, padding, 1, dilation),
            mem=(lambda shard=shard, grid_xy=grid_xy: _block(shard, grid_xy)),
            fidelity=LOFI,
            pcc=0.996,
        )
    for layer, ((H, W, c_in), c_out, padding, stride), shard, cores in HEIGHT_TILE:
        out[layer] = dict(
            shape=((H, W, c_in), c_out, padding, stride, 1),
            mem=(lambda shard=shard, cores=cores: _height(shard, cores)),
            fidelity=None,
            pcc=0.99,
        )
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


LAST = {}


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    if os.environ.get("CTD_NOZONE") == "1":
        # make_nozone.sh output: the same kernels with every MaybeDeviceZoneScope removed.
        monkeypatch.setattr(pd, "KERNEL_DIR", pd.KERNEL_DIR.parent / "kernels_nozone")
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nkb=blocking.num_k_blocks,
            ring=blocking.im2col_rm_depth_rows,
            ub=blocking.untilize_batch_rows,
            foot=blocking.footprint_bytes,
            center=int(blocking.center_mode),
            slots=int(blocking.center_slot_blocks),
        )
        return real(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op_mod, "create_program_descriptor", capture)


def make_inputs(case, seed, wide):
    """randn activation / weight / bias; `wide` scales every activation element by 2^U(-24, 24), so the bf8_b
    16-element blocks mix magnitudes and many members underflow their block's shared exponent (stresses the
    bf8_b -> bf16 -> bf8_b round trip the base path makes)."""
    import torch

    torch.manual_seed(seed)
    (H, W, c_in), c_out, *_ = case["shape"]
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    x = torch.randn(1, 1, H * W, c_in)
    if wide:
        x = x * torch.pow(2.0, torch.randint(-24, 25, x.shape).float())
    return x.to(torch.bfloat16), torch_weight, torch_bias


def run_case(case, device, seed=0, wide=False):
    import torch

    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    torch_input, torch_weight, torch_bias = make_inputs(case, seed, wide)
    memory_config = case["mem"]()
    dtype = case.get("dtype", ttnn.bfloat8_b)
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    config = None
    if case["fidelity"] is not None:
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fidelity"], fp32_dest_acc_en=True, math_approx_mode=False
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
        activation="relu",
        compute_kernel_config=config,
    )
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(
        torch.nn.functional.conv2d(
            x, torch_weight.float(), torch_bias.float(), stride=stride, padding=padding, dilation=dilation
        )
    )
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


PERF_CASES = _selected("CTD_CASES", ["focus"])
PERF_VARIANTS = _selected("CTD_VARIANTS", list(VARIANTS))


@pytest.mark.parametrize("variant", PERF_VARIANTS)
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_ctd_perf(case_id, variant, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch."""
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    out, ref = run_case(case, device)
    pcc = _pcc(out, ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(
                    kind="perf",
                    case=case_id,
                    variant=variant,
                    pcc=pcc,
                    tag=os.environ.get("CTD_TAG", ""),
                    nozone=os.environ.get("CTD_NOZONE") == "1",
                    **LAST,
                )
            )
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


@pytest.mark.parametrize("wide", [False, True], ids=["randn", "wide"])
@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != "base"])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_ctd_identical(case_id, variant, wide, device, monkeypatch):
    """Same inputs through base and the candidate: the outputs must be bit-identical."""
    import torch

    case = CASES[case_id]
    outs = {}
    blk = {}
    for v in ("base", variant):
        apply_variant(v, monkeypatch)
        outs[v], ref = run_case(case, device, seed=1, wide=wide)
        blk[v] = dict(LAST)
    same = bool(torch.equal(outs["base"], outs[variant]))
    pcc = _pcc(outs[variant], ref)
    diff = float((outs["base"] - outs[variant]).abs().max())
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(
                    kind="identical",
                    case=case_id,
                    variant=variant,
                    wide=wide,
                    identical=same,
                    pcc=pcc,
                    max_diff=diff,
                    blocking=blk,
                )
            )
            + "\n"
        )
    assert same, f"{case_id}/{variant}: output differs from base (max |diff| {diff})"
    if not wide:
        assert pcc >= case["pcc"]

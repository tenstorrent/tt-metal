# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the K split's owner-side combine tail (perf round 4, idea `ksplit_combine_tail`).

Runs this dir's private copy of conv2d_3x3 (op_copy/), switched by the descriptor's COMBINE_TAIL knob (a define on the
writer and compute kernels; see op_copy/kernels/conv2d_3x3_combine_tail.hpp):
  base        the committed tail: own slice packed to cb_matmul_partials, NoC1-copied by the writer into the own slot,
              ckl::reduce (AccumulateViaAdd, Skip) of k_cols slots into fp32 cb_combined, add_bias_bcast_rows (+ ReLU).
  fused       sum + bias + ReLU in one DEST pass (DEST-limit tiles per section), bias folded into the first pair.
  fused_x     as fused, but the bias goes in after the whole sum by dest reuse (committed rounding / order).
  own         fused + the own slice packed straight into cb_combined (no NoC1 self-copy, k_cols - 1 arrivals).
  own_x       fused_x + own.
  far*        the N-slice walk visits the remote owners farthest first (the last remote sends are one hop), alone and
              with the variants above.
  near*       the N-slice walk keeps its ascending order but moves the +-1 owners to the end (just before the own
              slice), so the last remote sends are the one-hop ones and the d = k/2 send has two slice periods.
  rfar*       only the reader's +d sends run farthest first (+k/2 at step 0 ... +1 last); the writer's order is kept.
  rmid*       the reader sends +1 first, then farthest first (+1, +k/2, ..., +2); the writer's order is kept.
  split*      the walk order is kept; the last remote slice (d = k/2, even k_cols, shared-L1 walk) goes half on the
              reader's NoC0 and half on the writer's NoC1 (idle at that step), each half its own arrival.

Precision contract is each case's own (fp32 DEST, its fidelity, bf8_b act + weights, bias + ReLU as the case says):
no variant touches it. Gate: PCC vs torch at the case's threshold, and (test_ct_identical) the output vs base.

One dispatch per test_ct_perf, for `run_safe_pytest.sh --profile`; each appends a JSON line to runs/<CT_LOG>.jsonl:
  CT_CASES=res5_1_conv2 CT_VARIANTS=base,own CT_LOG=r1 \\
    scripts/run_safe_pytest.sh --profile --run-all <this file> -k test_ct_perf -p no:cacheprovider --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ct_load  # noqa: E402

opc, pd, op_mod = ct_load.load()

VARIANTS = {
    "base": 0,
    "fused": 2,
    "fused_x": 6,
    "own": 3,
    "own_x": 7,
    "far": 8,
    "far_fused_x": 14,
    "far_own": 11,
    "far_own_x": 15,
    "near": 16,
    "near_fused_x": 22,
    "near_own_x": 23,
    "near_own": 19,
    "rfar": 32,
    "rfar_fused_x": 38,
    "rfar_own_x": 39,
    "rfar_own": 35,
    # diagnostic: rfar_own_x with the combine held ~1.1 us after its waits (UNPACK spin)
    "rfar_own_x_hold": 39 | 64,
    "rmid": 128,
    "rmid_own_x": 128 | 7,
    "rmid_own": 128 | 3,
    "split": 256,
    "split_own_x": 256 | 7,
    "split_own": 256 | 3,
}

LOFI, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4

# tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py BLOCK_SHAPES (the BLOCK guard harness).
# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
BLOCK_SHAPES = [
    ("res4_0_conv2", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, HIFI4, 2),
    ("res4_conv2", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, HIFI4, 1),
    ("res4_conv2_lofi", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, LOFI, 1),
    ("res5_0_conv2", ((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, HIFI4, 1),
    ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, HIFI4, 1),
    ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, HIFI4, 1),
    ("aspp_convs1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, HIFI4, 1),
    ("aspp_convs2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, HIFI4, 1),
    ("aspp_convs3", ((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, HIFI4, 1),
    # Not in the harness: res4.0's input at stride 1, so each core runs several M-blocks (the non-shared N-slice walk).
    ("res4_s1_lofi", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, LOFI, 1),
]


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = dict(kind="focus", spec=c)
    return out


def _cases():
    out = _focus_cases()
    for layer, ((H, W, c_in), c_out, padding, dilation), shard, grid_xy, bias_relu, fid, stride in BLOCK_SHAPES:
        out["block:" + layer] = dict(
            kind="block",
            shape=((H, W, c_in), c_out, padding, stride, dilation),
            shard=shard,
            grid_xy=grid_xy,
            bias_relu=bias_relu,
            fidelity=fid,
        )
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


LAST = {}


def apply_variant(variant, monkeypatch):
    monkeypatch.setattr(pd, "COMBINE_TAIL", VARIANTS[variant])
    pd.select_blocking.cache_clear()
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            k_cols=geometry.k_cols,
            k_split=bool(geometry.k_split),
            ns=geometry.out_n_tiles,
            bm=blocking.block_m_tiles,
            nkb=blocking.num_k_blocks,
            nslice=bool(blocking.nslice),
            shared=bool(blocking.shared_slab_bytes),
            shared_zero=bool(blocking.shared_zero_bytes),
            mblocks=blocking.max_m_blocks_per_core,
            foot=blocking.footprint_bytes,
            csb=(blocking.combine_subblock_h, blocking.combine_subblock_w),
        )
        return real(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op_mod, "create_program_descriptor", capture)


def _block_mem(shard, grid_xy):
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


def run_case(case, device, seed=0):
    """Returns (tt output as float torch, torch reference, pcc threshold)."""
    import torch

    torch.manual_seed(seed)
    if case["kind"] == "focus":
        from eval.golden_tests.conv2d_3x3 import helpers

        spec = case["spec"]
        (H, W, c_in), c_out, padding, stride, dilation = spec["inputs"]
        extras = spec["extras"]
        dtype, weight_dtype, layout = spec["dtype"], spec["weight_dtype"], spec["layout"]
        memory_config = helpers.build_input_memory_config(
            (1, 1, H * W, c_in), spec["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
        )
        if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
            pytest.skip("pinned shard geometry does not fit this device's grid")
        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        has_bias = spec["bias_mode"] == "with_bias"
        activation = None if spec["activation"] == "none" else spec["activation"]
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
        config = helpers.make_compute_kernel_config(
            spec["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        )
        pcc_min = extras.get("tolerance", (0.99, None))[0]
    else:
        (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
        dtype, weight_dtype, layout = ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT
        memory_config = _block_mem(case["shard"], case["grid_xy"])
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias_relu"] else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        activation = "relu" if case["bias_relu"] else None
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fidelity"], fp32_dest_acc_en=True, math_approx_mode=False
        )
        pcc_min = 0.99
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    tt_output = opc.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=config,
    )
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if activation == "relu":
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref, pcc_min


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _log(rec):
    log = HERE / "runs" / (os.environ.get("CT_LOG", "default") + ".jsonl")
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")


PERF_CASES = _selected("CT_CASES", ["res5_1_conv2"])
PERF_VARIANTS = _selected("CT_VARIANTS", list(VARIANTS))


@pytest.mark.parametrize("variant", PERF_VARIANTS)
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_ct_perf(case_id, variant, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch."""
    apply_variant(variant, monkeypatch)
    out, ref, pcc_min = run_case(CASES[case_id], device)
    pcc = _pcc(out, ref)
    digest = hashlib.sha1(out.numpy().tobytes()).hexdigest()[:12]
    _log(dict(kind="perf", case=case_id, variant=variant, pcc=pcc, hash=digest, **LAST))
    if os.environ.get("CT_FLUSH") == "1":
        # Long profiled sessions: flush the device profiler per dispatch (marker buffers).
        ttnn.ReadDeviceProfiler(device)
    assert pcc >= pcc_min, f"{case_id}/{variant}: pcc {pcc} < {pcc_min}"


@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != "base"])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_ct_identical(case_id, variant, device, monkeypatch):
    """Same inputs through base and the candidate: report bit-identity, mismatch count and max |diff|."""
    import torch

    outs = {}
    for v in ("base", variant):
        apply_variant(v, monkeypatch)
        outs[v], ref, pcc_min = run_case(CASES[case_id], device, seed=1)
    a, b = outs["base"], outs[variant]
    diff = (a - b).abs()
    rec = dict(
        kind="identical",
        case=case_id,
        variant=variant,
        identical=bool(torch.equal(a, b)),
        mismatches=int((diff > 0).sum()),
        numel=int(diff.numel()),
        max_diff=float(diff.max()),
        pcc_base=_pcc(a, ref),
        pcc_var=_pcc(b, ref),
        maxerr_base=float((a - ref).abs().max()),
        maxerr_var=float((b - ref).abs().max()),
        meanerr_base=float((a - ref).abs().mean()),
        meanerr_var=float((b - ref).abs().mean()),
        **LAST,
    )
    _log(rec)
    assert rec["pcc_var"] >= pcc_min, f"{case_id}/{variant}: pcc {rec['pcc_var']} < {pcc_min}"

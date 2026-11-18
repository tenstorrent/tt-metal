# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off: tap-aligned block_k_tiles in the conv2d_3x3 L1 solve.

Every variant runs the op COPY in op_copy/ (frozen at Perf 1, + experiment knobs in its descriptor) with one
dispatch, checks PCC / rel-RMS against torch, records the blocking it ran with to runs.jsonl, and saves its
output under OUT_DIR so tap-major vs row-gather outputs can be compared bit for bit (compare_outputs.py).

    scripts/run_safe_pytest.sh --profile --run-all --precompile \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tap_aligned_blocking/test_tap_aligned.py -k "<ids>"
    python3 -I .../tap_aligned_blocking/report.py <ops_perf_results csv>

Variants (`knobs`): TAP_ALIGNED_BK = rule A; FORCED = a menu point (bm, bk[, weight_depth]); TAP_MAJOR_IM2COL False =
the row gather at the same blocking (bit-identity reference). Every variant runs with the user's config unchanged
(LoFi, fp32_dest_acc_en, math_approx_mode False, bf8_b act / weights). Zones are compiled out (NO_ZONES) unless
TAB_ZONES=1.
"""

import json
import os
import time
from pathlib import Path

import pytest
import ttnn

from ttnn.operations.conv2d_3x3.perf_experiments.tap_aligned_blocking.op_copy import conv2d_3x3 as op_mod_fn  # noqa
import sys

OP = sys.modules["ttnn.operations.conv2d_3x3.perf_experiments.tap_aligned_blocking.op_copy.conv2d_3x3"]
PD = sys.modules[
    "ttnn.operations.conv2d_3x3.perf_experiments.tap_aligned_blocking.op_copy.conv2d_3x3_program_descriptor"
]

HERE = Path(__file__).parent
LOG = HERE / "runs.jsonl"
OUT_DIR = Path(os.environ.get("TAB_OUT_DIR", "/tmp/tap_aligned_blocking_outputs"))
ZONES = os.environ.get("TAB_ZONES") == "1"


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    return {c["extras"]["layer"].split(" ")[0].replace(".", "_"): c for c in cases}


FOCUS = _focus_cases()

# (case id, variant id, knobs)
FOCUS_VARIANTS = []
for res5 in ("res5_1_conv2", "res5_2_conv2"):
    FOCUS_VARIANTS += [
        (res5, "base", {}),
        (res5, "ruleA_bm7_bk2", {"TAP_ALIGNED_BK": True}),
        (res5, "bm7_bk6_wd1", {"FORCED": dict(bm=7, bk=6, weight_depth=1)}),
        (res5, "bm1_bk6", {"FORCED": dict(bm=1, bk=6)}),
        (res5, "rowgather_bm7_bk2", {"FORCED": dict(bm=7, bk=2), "TAP_MAJOR_IM2COL": False}),
    ]
FOCUS_VARIANTS += [
    ("semantic", "base", {}),
    ("semantic", "ruleA_bm5_bk8", {"TAP_ALIGNED_BK": True}),
    ("semantic", "bm2_bk24", {"FORCED": dict(bm=2, bk=24)}),
    ("semantic", "bm1_bk24", {"FORCED": dict(bm=1, bk=24)}),
    ("semantic", "bm5_bk24_wd1", {"FORCED": dict(bm=5, bk=24, weight_depth=1)}),
    ("semantic", "bm10_bk8_wd1", {"FORCED": dict(bm=10, bk=8, weight_depth=1)}),
    ("semantic", "rowgather_bm10_bk6", {"FORCED": dict(bm=10, bk=6)}),
    ("semantic", "rowgather_bm5_bk8", {"FORCED": dict(bm=5, bk=8), "TAP_MAJOR_IM2COL": False}),
]

KNOB_DEFAULTS = dict(TAP_ALIGNED_BK=False, TAP_ALIGNED_SINGLE_WEIGHTS=False, FORCED=None, TAP_MAJOR_IM2COL=True)


def _set_knobs(knobs):
    for name, default in KNOB_DEFAULTS.items():
        setattr(PD, name, knobs.get(name, default))
    PD.NO_ZONES = not ZONES
    PD.select_blocking.cache_clear()


def _capture_blocking():
    """Wrap the copy's select_blocking as conv2d_3x3.py sees it, to record what the op ran with."""
    seen = {}
    real = PD.select_blocking

    def spy(g, grid, budget):
        b = real(g, grid, budget)
        seen.update(g=g, b=b, budget=budget)
        return b

    return seen, spy


def _metrics(expected, actual):
    e = expected.double().flatten()
    a = actual.double().flatten()
    ec, ac = e - e.mean(), a - a.mean()
    pcc = float((ec * ac).sum() / (ec.norm() * ac.norm()))
    rel_rms = float(((a - e) ** 2).mean().sqrt() / e.std())
    return pcc, rel_rms


def _log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _run(case_id, case, variant, knobs, device):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    _set_knobs(knobs)
    seen, spy = _capture_blocking()
    OP.select_blocking = spy
    try:
        torch.manual_seed(0)
        (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
        extras = case.get("extras", {})
        dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
        if case["memory_layout"] == ttnn.TensorMemoryLayout.INTERLEAVED:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            memory_config = helpers.build_input_memory_config(
                (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
            )
            if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
                pytest.skip("pinned shard geometry does not fit this device's grid")
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
        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        has_bias = case["bias_mode"] == "with_bias"
        activation = None if case["activation"] == "none" else case["activation"]
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
        tt_weight = OP.prepare_conv2d_weights(torch_weight, **kwargs)
        tt_bias = OP.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

        tt_output = OP.conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=activation,
            compute_kernel_config=helpers.make_compute_kernel_config(
                case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
            ),
        )
        out = ttnn.to_torch(tt_output)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
        ref = helpers.pytorch_conv2d_3x3(
            torch_input,
            torch_weight,
            input_height=H,
            input_width=W,
            bias=torch_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=activation,
        )
        out = out.reshape(ref.shape)
        pcc, rel_rms = _metrics(ref.float(), out.float())
        g, b = seen["g"], seen["b"]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(out.to(torch.bfloat16), OUT_DIR / f"{case_id}__{variant}.pt")
        _log(
            dict(
                ts=time.time(),
                case=case_id,
                variant=variant,
                zones=ZONES,
                bm=b.block_m_tiles,
                bk=b.block_k_tiles,
                num_k_blocks=b.num_k_blocks,
                m_blocks=b.max_m_blocks_per_core,
                weight_depth=b.weight_depth,
                ub=b.untilize_batch_rows,
                footprint=b.footprint_bytes,
                budget=seen["budget"],
                tap_major=bool(PD.tap_major_im2col(g, b)),
                pcc=pcc,
                rel_rms=rel_rms,
            )
        )
        pcc_gate, rms_gate = extras.get("tolerance", (0.99, 0.12))
        assert pcc >= pcc_gate and rel_rms <= rms_gate, f"{case_id}/{variant}: pcc {pcc:.5f} rel_rms {rel_rms:.4f}"
    finally:
        OP.select_blocking = PD.select_blocking
        _set_knobs({})


@pytest.mark.parametrize("case_id,variant,knobs", FOCUS_VARIANTS, ids=[f"{c}-{v}" for c, v, _ in FOCUS_VARIANTS])
def test_focus(case_id, variant, knobs, device):
    _run(case_id, FOCUS[case_id], variant, knobs, device)


# Domain: the other perf-harness shapes rule A changes (host solve over test_conv2d_3x3_perf.py's HEIGHT / BLOCK /
# interleaved lists): only the interleaved 8x16x512 -> 256 (bk 24 -> 16), in its three layouts. Weights bf16, bias,
# no activation, HiFi4, fp32 DEST, as in test_conv2d_3x3_perf.
def _interleaved(H, W, c_in, c_out, dtype, layout):
    return {
        "inputs": ((H, W, c_in), c_out, 1, 1, 1),
        "dtype": dtype,
        "weight_dtype": ttnn.bfloat16,
        "layout": layout,
        "memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED,
        "fp32_dest_acc_en": True,
        "bias_mode": "with_bias",
        "activation": "none",
        "extras": {"tolerance": (0.995 if dtype == ttnn.bfloat16 else 0.99, 0.12)},
    }


DOMAIN = {
    "i8x16x512_rm_bf16": _interleaved(8, 16, 512, 256, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    "i8x16x512_tile_bf16": _interleaved(8, 16, 512, 256, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "i8x16x512_tile_bf8b": _interleaved(8, 16, 512, 256, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
}
DOMAIN_VARIANTS = [(c, v, k) for c in DOMAIN for v, k in (("base", {}), ("ruleA", {"TAP_ALIGNED_BK": True}))]
# Bit-identity reference: the row gather at rule A's blocking (bk 16), so a base-vs-ruleA difference can be split into
# the K-block regrouping (bk 24 -> 16) and the gather layout (tap-major vs row).
DOMAIN_VARIANTS += [
    (c, "rowgather_bk16", {"FORCED": dict(bm=1, bk=16), "TAP_MAJOR_IM2COL": False})
    for c in ("i8x16x512_rm_bf16", "i8x16x512_tile_bf16")
]


@pytest.mark.parametrize("case_id,variant,knobs", DOMAIN_VARIANTS, ids=[f"{c}-{v}" for c, v, _ in DOMAIN_VARIANTS])
def test_domain(case_id, variant, knobs, device):
    _run(case_id, DOMAIN[case_id], variant, knobs, device)

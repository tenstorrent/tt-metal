# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-op bake-off: conv2d_3x3's TILE-slab untilize, helper vs narrow_untilize (this dir's op copy only).

Variants (module knobs of the copy's program descriptor, compiled in as compute defines):
  base      compute_kernel_lib::untilize, exactly the op's path (the honest baseline; same copy, same zones)
  nu        1-tile slices: narrow_untilize::untilize_1wide, contiguous-stream pack, 4 tile-rows per DEST section,
            CB sync per section. Wider slices: the helper.
  nu_bsync  as nu, CB sync once per untilize batch
  nu_s2     as nu, 2 tile-rows per DEST section
  wide      wide slices (2+ tiles): the helper's fast_untilize with CB sync per untilize batch. 1-tile: the helper.
  all       nu + wide

One test per (case, order): the body dispatches the variants of NU_VARIANTS in that order (order "ba" reverses it),
checks every output against torch (case PCC) and bit-equality against the first variant's output (only the untilize
implementation differs and it is lossless). Each dispatch appends a record to runs/run_<pid>.jsonl.

    NU_CASES=res4_0_conv2_perf NU_VARIANTS=base,nu scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/narrow_untilize/test_narrow_untilize_op.py
    python3 .../narrow_untilize/report.py <ops_perf_results csv>

Cases: every TILE Panoptic LOOSE_CASE (id: layer's first token, dots -> '_', '_perf' for the [perf] rows, '_lofi' for
the LoFi twin), plus interleaved 1-tile-slice shapes (il_*).
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
PKG = "narrow_untilize_op_copy"
RUNS = HERE / "runs"


def _load_copy():
    # No __init__.py anywhere in this experiment (ttnn walk_packages executes every package module on import): the
    # op copy is loaded as a private namespace package.
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(HERE / "op")]
        sys.modules[PKG] = pkg
    return importlib.import_module(PKG + ".conv2d_3x3"), importlib.import_module(PKG + ".conv2d_3x3_program_descriptor")


op_mod, pd = _load_copy()

VARIANTS = {
    "base": dict(NARROW_UNTILIZE=0, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=4, NU_CB_SYNC=0),
    "nu": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=4, NU_CB_SYNC=0),
    "nu_bsync": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=4, NU_CB_SYNC=1),
    "nu_s2": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=2, NU_CB_SYNC=0),
    "nu_rowmop": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=4, NU_CB_SYNC=0, NU_PACK=0),
    "nu_rowmop_s1": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=1, NU_CB_SYNC=0, NU_PACK=0),
    "nu_s1": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=0, NU_ROWS_PER_SECTION=1, NU_CB_SYNC=0),
    "wide": dict(NARROW_UNTILIZE=0, WIDE_UNTILIZE_BATCH=1, NU_ROWS_PER_SECTION=4, NU_CB_SYNC=0),
    "all": dict(NARROW_UNTILIZE=1, WIDE_UNTILIZE_BATCH=1, NU_ROWS_PER_SECTION=4, NU_CB_SYNC=0),
}


def _case_id(c):
    layer = c["extras"]["layer"]
    cid = layer.split(" ")[0].replace(".", "_").replace("{", "").replace("}", "").replace(",", "")
    if "[perf]" in layer:
        cid += "_perf"
    elif c["extras"].get("math_fidelity") == ttnn.MathFidelity.LoFi:
        cid += "_lofi"
    if layer.startswith(("semantic", "instance")):
        cid = layer.split(" ")[0][:4] + "_" + layer.split(" ")[1].replace(".", "_") + cid[len(layer.split(" ")[0]) :]
    return cid


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = {}
    for c in LOOSE_CASES:
        if c["layout"] != ttnn.TILE_LAYOUT:
            continue
        cid = _case_id(c)
        while cid in cases:
            cid += "_b"
        cases[cid] = c
    # Interleaved 1-tile slices (C_in <= 32): the only non-BLOCK cases with a 1-tile-wide slab.
    for dtype, tag in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8b")):
        for (H, W, c_in), c_out, stride in (((32, 64, 32), 64, 1), ((64, 64, 16), 32, 2)):
            cases[f"il_c{c_in}_s{stride}_{tag}"] = {
                "inputs": ((H, W, c_in), c_out, 1, stride, 1),
                "dtype": dtype,
                "weight_dtype": ttnn.bfloat16,
                "layout": ttnn.TILE_LAYOUT,
                "memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED,
                "fp32_dest_acc_en": True,
                "bias_mode": "with_bias",
                "activation": "relu",
                "extras": {"layer": f"interleaved c_in {c_in}", "tolerance": (0.99, None)},
            }
    # HEIGHT_SHARDED 32-channel TILE input (1-tile slab rows on the R2 path), 32 cores x [128, 32].
    for dtype, tag in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8b")):
        for stride in (1, 2):
            cases[f"hs_c32_s{stride}_{tag}"] = {
                "inputs": ((64, 64, 32), 64, 1, stride, 1),
                "dtype": dtype,
                "weight_dtype": ttnn.bfloat8_b,
                "layout": ttnn.TILE_LAYOUT,
                "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "fp32_dest_acc_en": True,
                "bias_mode": "with_bias",
                "activation": "relu",
                "extras": {"layer": "height c_in 32", "tolerance": (0.99, None), "_hs_cores": (8, 4)},
            }
    return cases


CASES = _cases()
SELECTED = [c for c in os.environ.get("NU_CASES", "res4_0_conv2_perf").split(",") if c]
ORDERS = [o for o in os.environ.get("NU_ORDERS", "ab").split(",") if o]


def _run(case, device, variant):
    setattr(pd, "NU_PACK", 1)
    for k, v in VARIANTS[variant].items():
        setattr(pd, k, v)
    torch_input, torch_weight, torch_bias, kw = case["_tensors"]
    tt_weight = op_mod.prepare_conv2d_weights(torch_weight, **kw["prep"])
    tt_bias = op_mod.prepare_conv2d_bias(torch_bias, **kw["prep"]) if torch_bias is not None else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=kw["mem"]
    )
    out = op_mod.conv2d_3x3(tt_input, tt_weight, bias=tt_bias, **kw["call"])
    result = ttnn.to_torch(out).float()
    for t in (tt_input, tt_weight, tt_bias, out):
        if t is not None:
            t.deallocate()
    return result


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("case_id", SELECTED)
def test_narrow_untilize_op(case_id, order, device):
    import torch

    from eval.golden_tests.conv2d_3x3 import helpers
    from tests.ttnn.utils_for_testing import comp_pcc

    case = CASES[case_id]
    variants = [v for v in os.environ.get("NU_VARIANTS", "base,nu").split(",") if v]
    if order == "ba":
        variants = variants[::-1]
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    if "_hs_cores" in extras:
        gx, gy = extras["_hs_cores"]
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
                [H * W // (gx * gy), c_in],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    else:
        mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if case["memory_layout"] == ttnn.TensorMemoryLayout.INTERLEAVED
            else helpers.build_input_memory_config(
                (1, 1, H * W, c_in),
                case["memory_layout"],
                layout=case["layout"],
                dtype=case["dtype"],
                device=device,
                extras=extras,
            )
        )
    if "shard_shape" in extras and (
        mem.shard_spec is None or list(mem.shard_spec.shape) != list(extras["shard_shape"])
    ):
        pytest.skip(f"{extras['layer']}: pinned shard geometry does not fit this device's grid")
    has_bias = case["bias_mode"] == "with_bias"
    activation = None if case["activation"] == "none" else case["activation"]
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    prep = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=mem,
        device=device,
    )
    call = dict(
        input_height=H,
        input_width=W,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    case = dict(case, _tensors=(torch_input, torch_weight, torch_bias, dict(prep=prep, mem=mem, call=call)))
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
    ).float()
    pcc_thr = extras.get("tolerance", (0.99, None))[0]

    RUNS.mkdir(exist_ok=True)
    log = RUNS / f"run_{os.getpid()}.jsonl"
    first = None
    failures = []
    for v in variants:
        out = _run(case, device, v).reshape(ref.shape)
        ok, pcc = comp_pcc(ref, out, pcc_thr)
        pcc = float(pcc) if not isinstance(pcc, str) else pcc
        equal = None if first is None else bool(torch.equal(out, first[1]))
        if first is None:
            first = (v, out)
        if not ok:
            failures.append(f"{case_id}/{v}: pcc {pcc} < {pcc_thr}")
        if equal is False:
            failures.append(f"{case_id}/{v}: output differs from {first[0]}")
        with log.open("a") as f:
            f.write(
                json.dumps(dict(case=case_id, order=order, variant=v, pcc=pcc, equal_first=equal, first=first[0]))
                + "\n"
            )
        print(f"NU_RESULT {case_id} {order} {v}: pcc={pcc} equal_first={equal}")
    # NU_NO_CHECK=1: ablation runs (CONV2D_3X3_ABLATE) are wrong by construction; record, do not gate.
    assert os.environ.get("NU_NO_CHECK") == "1" or not failures, failures


@pytest.mark.skipif(os.environ.get("NU_DEBUG") != "1", reason="debug: NU_DEBUG=1")
@pytest.mark.parametrize("case_id", SELECTED)
def test_nu_determinism(case_id, device):
    """Runs NU_VARIANTS twice each (a, b, a, b) and prints which output elements differ between any pair."""
    import torch

    from eval.golden_tests.conv2d_3x3 import helpers

    case = CASES[case_id]
    variants = [v for v in os.environ.get("NU_VARIANTS", "base,nu").split(",") if v]
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    mem = (
        ttnn.DRAM_MEMORY_CONFIG
        if case["memory_layout"] == ttnn.TensorMemoryLayout.INTERLEAVED
        else helpers.build_input_memory_config(
            (1, 1, H * W, c_in),
            case["memory_layout"],
            layout=case["layout"],
            dtype=case["dtype"],
            device=device,
            extras=extras,
        )
    )
    has_bias = case["bias_mode"] == "with_bias"
    activation = None if case["activation"] == "none" else case["activation"]
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    prep = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=mem,
        device=device,
    )
    call = dict(
        input_height=H,
        input_width=W,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    case = dict(case, _tensors=(torch_input, torch_weight, torch_bias, dict(prep=prep, mem=mem, call=call)))
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
    ).float()
    outs = []
    for rep in range(2):
        for v in variants:
            outs.append((f"{v}#{rep}", _run(case, device, v).reshape(ref.shape)))
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            a, b = outs[i][1].reshape(-1, ref.shape[-1]), outs[j][1].reshape(-1, ref.shape[-1])
            d = a != b
            n = int(d.sum())
            msg = f"NU_DIFF {case_id} {outs[i][0]} vs {outs[j][0]}: {n} differing elements"
            if n:
                px = d.any(dim=1).nonzero().flatten()
                ch = d.any(dim=0).nonzero().flatten()
                mx = float((a - b).abs().max())
                ra = float(((a - ref.reshape(a.shape)).abs().max()))
                rb = float(((b - ref.reshape(b.shape)).abs().max()))
                msg += (
                    f"; pixels {px.numel()} first {px[:12].tolist()} last {px[-4:].tolist()}; channels {ch.numel()}"
                    f" {ch[:8].tolist()}; max|a-b| {mx:.4g}; max|a-ref| {ra:.4g} max|b-ref| {rb:.4g}"
                )
            print(msg)
            if n and os.environ.get("NU_DEBUG_DETAIL") == "1" and i == 0:
                rows = d.any(dim=1).reshape(-1, 32)
                for t in range(rows.shape[0]):
                    if rows[t].any():
                        offs = rows[t].nonzero().flatten().tolist()
                        print(f"NU_TILEROW {outs[j][0]} out tile-row {t}: pixel offsets {offs}")

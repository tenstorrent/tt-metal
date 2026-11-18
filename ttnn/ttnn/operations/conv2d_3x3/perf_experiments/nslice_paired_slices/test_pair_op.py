# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op A/B on a private copy of conv2d_3x3 (op_copy/): N-slice-outer walk, one owner slice per matmul call
(baseline = the committed op) vs two consecutive slices per call (pair rounds; see NSLICE_PAIR in the copy's
descriptor).

    PO_CASES=res5_1_conv2_perf,res5_2_conv2_perf PO_VARIANTS=baseline,pair \\
        scripts/run_safe_pytest.sh --profile --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_paired_slices/test_pair_op.py --import-mode=prepend
    python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_paired_slices/report.py <ops_perf_results.csv>

Variants (all on the copy, so instrumentation is identical; never profile the copy next to the real op):
  baseline  the committed op: one slice per call (1x2 subblocks on res5), weight round per slice, depth 4 slices.
  pair      pair rounds: [K, 2 * out_n] interleaved weight rounds, depth 3 pairs; paired raw walk (1x4 on res5);
            slice CBs 2 deep.
  pair_sd1  pair with single-depth slice CBs (the shared-L1 walk's current depth): less L1, the next pair waits on
            both sends.
  pair_d4   pair with depth 4 pair rounds and two rounds of lookahead.
  hyb2      pairs for all but the last two walk steps, which run as single-slice rounds (compact [K, out_n] weights),
            depth 4 slots, two rounds of lookahead.
  pair_mf / hyb2_mf  pair / hyb2 with the weight sender multicasting each round before it issues the next round's
            prefetch reads.
  base_mf   control: baseline (no pairing) with only that reorder.

The precision contract is the case's own (fp32 DEST, its fidelity, bf8_b): every variant runs it unchanged. Each
dispatch appends a JSON record (blocking, PCC, output hash) to runs/<PO_LOG>.jsonl; report.py joins it with the
profiler CSV's GenericOp rows in call order and checks pair outputs are bit-identical to baseline.
"""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
# No __init__.py anywhere under perf_experiments (ttnn walk_packages would execute it on `import ttnn`): the copy is a
# namespace package.
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_paired_slices.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / "op_copy", pd.KERNEL_DIR

KNOB_DEFAULTS = dict(
    NSLICE_PAIR=False,
    NSLICE_PAIR_WEIGHT_DEPTH=3,
    NSLICE_PAIR_SLICE_DEPTH=2,
    NSLICE_PAIR_LOOKAHEAD=1,
    NSLICE_PAIR_TAIL=0,
    NSLICE_PAIR_MCAST_FIRST=False,
)
VARIANTS = {
    "baseline": {},
    "pair": dict(NSLICE_PAIR=True),
    "pair_sd1": dict(NSLICE_PAIR=True, NSLICE_PAIR_SLICE_DEPTH=1),
    # Every pair round in flight early: depth 4 pair rounds (all of res5's weights), two rounds ahead.
    "pair_d4": dict(NSLICE_PAIR=True, NSLICE_PAIR_WEIGHT_DEPTH=4, NSLICE_PAIR_LOOKAHEAD=2),
    # Hybrid: the last two walk steps as single-slice rounds (the weight-gated tail), depth 4 slots, two rounds ahead.
    "hyb2": dict(NSLICE_PAIR=True, NSLICE_PAIR_WEIGHT_DEPTH=4, NSLICE_PAIR_LOOKAHEAD=2, NSLICE_PAIR_TAIL=2),
    # The sender multicasts each round before issuing the next round's prefetch reads.
    "pair_mf": dict(NSLICE_PAIR=True, NSLICE_PAIR_MCAST_FIRST=True),
    # Attribution control: the committed per-slice walk with only the mcast-before-prefetch reorder.
    "base_mf": dict(NSLICE_PAIR_MCAST_FIRST=True),
    "hyb2_mf": dict(
        NSLICE_PAIR=True,
        NSLICE_PAIR_WEIGHT_DEPTH=4,
        NSLICE_PAIR_LOOKAHEAD=2,
        NSLICE_PAIR_TAIL=2,
        NSLICE_PAIR_MCAST_FIRST=True,
    ),
}


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()


def _block_cases():
    """Every BLOCK LOOSE case, id from the layer name (+ _perf for the LoFi focus rows, _lofi for the LoFi twin)."""
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if c["memory_layout"] != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            continue
        layer = c["extras"]["layer"]
        cid = layer.split(" ")[0].replace(".", "_").replace("{", "").replace("}", "")
        if "[perf]" in layer:
            cid += "_perf"
        elif "@ LoFi" in layer:
            cid += "_lofi"
        out.setdefault(cid, c)
    return out


CASES = _block_cases()
FOCUS = ["res5_1_conv2_perf", "res5_2_conv2_perf"]
SEL_CASES = [c for c in os.environ.get("PO_CASES", ",".join(FOCUS)).split(",") if c]
if SEL_CASES == ["all"]:
    SEL_CASES = list(CASES)
SEL_VARIANTS = [v for v in os.environ.get("PO_VARIANTS", "baseline,pair").split(",") if v]
RUN_LOG = HERE / "runs" / os.environ.get("PO_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def run_case(case, device, seed=0):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(seed)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    memory_config = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip(f"{extras['layer']}: pinned shard geometry does not fit this device's grid")
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
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    tt_output = op.conv2d_3x3(
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
    return tt_output, ref


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            ub=b.untilize_batch_rows,
            nslice=bool(b.nslice),
            pair=bool(b.nslice_pair),
            out_n=g.out_n_tiles,
            k_cols=g.k_cols,
            k_tiles=g.tensor_k_tiles,
            slice_sb=list(pd._out_subblock(b.block_m_tiles, g.out_n_tiles)),
            weight_depth=b.weight_depth,
            tail=pd.NSLICE_PAIR_TAIL,
            lookahead=pd.NSLICE_PAIR_LOOKAHEAD,
            slice_depth=b.slice_depth,
            alias=b.shared_slab_bytes > 0,
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("variant", SEL_VARIANTS)
@pytest.mark.parametrize("case_id", SEL_CASES)
def test_pair_op(case_id, variant, device, monkeypatch):
    import torch

    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    tt_output, ref = run_case(case, device)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    pcc_threshold = (case["extras"].get("tolerance") or (0.99, None))[0]
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert info["finite"] and passed, info

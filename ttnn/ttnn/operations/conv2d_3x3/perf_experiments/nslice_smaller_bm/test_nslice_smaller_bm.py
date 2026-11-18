# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: run aspp's K split N-slice-outer instead of the K-blocked walk (perf experiment, not part of the op).

At the K-blocked solve's bm (8) aspp.convs.1/2's resident N-slice-outer footprint is 1.77 MB against a 1.28 MB CB
budget, so the op falls back to the K-blocked walk (9 fp32 L1-accumulation passes + a post-walk scatter). Variants
(all on this dir's op copy, op_copy/, whose knobs default to the committed op's behaviour):

  baseline   the committed solve: K-blocked walk, bm 8, bk 6, ub 3.
  A_bm4      N-slice-outer at a smaller bm when it does not fit at the K-blocked bm (bm 4, bk 18: two M-blocks per
             core, weights streamed twice, two combine rounds).
  B_bk18     bm 8, N-slice-outer, cb_combine_slots sharing cb_halo_slab's L1 (the slab is dead once the core's gather
             landed; owners hand out their combine credits only after that), slice CB depth 1; solver's pick
             (bk 18, untilize batch 1).
  B_bk6      as B_bk18 at bk 6 (untilize batch 5).
  B_*_zc     as B_*, with cb_combined also sharing cb_zero_src's L1 (one more untilize batch row fits).
  rec        the recommended solver rule (B, cb_combined in cb_zero_src, slice CB depth 1).

The precision contract is the case's own (fp32 DEST, its fidelity, bf8_b): every variant runs it unchanged.

Env: NSBM_CASES / NSBM_VARIANTS (comma lists; default the two focus cases x every variant). One dispatch per test, for
`run_safe_pytest.sh --profile`; each dispatch appends a JSON record (blocking, PCC, output hash) to runs/<log>.jsonl,
which report.py joins with the profiler CSV's GenericOp rows in call order.
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
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_smaller_bm.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / "op_copy", pd.KERNEL_DIR

KNOB_DEFAULTS = dict(
    NSLICE_SMALLER_BM=False,
    NSLICE_ALIAS_SLAB_COMBINE=False,
    NSLICE_ALIAS_ALWAYS=False,
    NSLICE_ALIAS_CONFIGS=((2, False), (1, False)),
    FORCE_BLOCKING=None,
)
VARIANTS = {
    "baseline": {},
    "A_bm4": dict(NSLICE_SMALLER_BM=True),
    "B_bk18": dict(NSLICE_ALIAS_SLAB_COMBINE=True),
    "B_bk6": dict(NSLICE_ALIAS_SLAB_COMBINE=True, FORCE_BLOCKING=(8, 6, True, (1, False))),
    # The B solve with cb_combined in cb_zero_src's L1 as well (one more untilize batch row may fit).
    "B_bk6_zc": dict(NSLICE_ALIAS_SLAB_COMBINE=True, FORCE_BLOCKING=(8, 6, True, (1, True))),
    "B_bk18_zc": dict(NSLICE_ALIAS_SLAB_COMBINE=True, FORCE_BLOCKING=(8, 18, True, (1, True))),
    # Recommended solver rule: B with cb_combined in cb_zero_src too and single-buffered slice CBs (A measured slower, so
    # no smaller-bm fallback). What graduation would run.
    "rec": dict(NSLICE_ALIAS_SLAB_COMBINE=True, NSLICE_ALIAS_CONFIGS=((1, True),)),
}


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()


def _block_cases():
    """Every BLOCK LOOSE case, id from the layer name (+ _perf for the LoFi focus rows)."""
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
FOCUS = ["aspp_convs_1_perf", "aspp_convs_2_perf"]
SEL_CASES = [c for c in os.environ.get("NSBM_CASES", ",".join(FOCUS)).split(",") if c]
if SEL_CASES == ["all"]:
    SEL_CASES = list(CASES)
SEL_VARIANTS = [v for v in os.environ.get("NSBM_VARIANTS", ",".join(VARIANTS)).split(",") if v]
RUN_LOG = HERE / "runs" / os.environ.get("NSBM_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


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
            alias=bool(b.alias_slab_combine),
            alias_combined=b.alias_combined_region_bytes > 0,
            slice_depth=b.slice_depth,
            tap_major=bool(pd.tap_major_im2col(g, b)),
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("variant", SEL_VARIANTS)
@pytest.mark.parametrize("case_id", SEL_CASES)
def test_nsbm_perf(case_id, variant, device, monkeypatch):
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

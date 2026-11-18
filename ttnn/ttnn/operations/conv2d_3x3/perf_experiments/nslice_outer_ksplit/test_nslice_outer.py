# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: N-slice-outer K walk for the R4 BLOCK K split (perf experiment, not part of the op).

Runs a private copy of conv2d_3x3 (op_copy/) whose R4 path is switched by module knobs:
  baseline          the committed R4 walk: K-blocked matmul with fp32 packer-L1 accumulation of all [bm, C_out]
                    partials, then one post-walk row reduce-scatter (writer NoC1 + reader NoC0 halves).
  nslice            in0 tilized once for all of K (resident [bm, K]); per owner N-slice one whole-K matmul call
                    (DEST accumulation, one fp32 pack of [bm, out_n]); the writer scatters slice y to owner y on
                    NoC1 while compute multiplies the next slice. Same bm / bk as the baseline.
  nslice_split      as nslice, the higher owner columns' slices sent by the reader on NoC0 from a second slice CB.
  nslice_split_alt  as nslice_split, slice order my_col+1, my_col-1, my_col+2, ... (own last).
  nslice_free       as nslice_split, but the L1 solve may pick the coarsest bk the smaller footprint allows.

Precision contract is the case's own (LoFi, fp32 DEST, bf8_b): every variant runs it unchanged.

One dispatch per test, for `run_safe_pytest.sh --profile`. Each test appends a JSON line (metrics + blocking +
L1 footprint) to results.jsonl next to this file; the profiler CSV gives DEVICE KERNEL DURATION.
`ablate` = CONV2D_3X3_ABLATE tokens of the copy's ablation hooks ("gather" stubs the im2col gather reads; the
output is then wrong by construction and no PCC is checked).
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.metrics import CheckOutputError, check_output

# This directory deliberately has no __init__.py: ttnn.operations walk_packages-executes every module of every
# package under it on `import ttnn`, so the op copy is loaded by path under a private package name instead.
OPC = "nslice_outer_ksplit_op_copy"


def _load_op_copy():
    if OPC not in sys.modules:
        root = Path(__file__).parent / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC]


opc = _load_op_copy()
pd = sys.modules[OPC + ".conv2d_3x3_program_descriptor"]
op_mod = sys.modules[OPC + ".conv2d_3x3"]

RESULTS = Path(__file__).parent / "results.jsonl"

KNOB_DEFAULTS = dict(
    NSLICE_OUTER=False,
    NSLICE_SPLIT_NOCS=False,
    NSLICE_DEPTH=2,
    NSLICE_PIN_BASELINE_BLOCKING=True,
    NSLICE_ORDER="rotate",
    NSLICE_SENDER=None,
    NSLICE_WEIGHT_DEPTH=2,
    NSLICE_WEIGHT_PREFETCH=False,
    NSLICE_WEIGHT_LOOKAHEAD=1,
    NSLICE_FALLBACK=False,
    FORCE_BLOCKING=None,
)
VARIANTS = {
    "baseline": {},
    # Attribution: the baseline scheme at the blocking the unpinned candidate picks on aspp.convs.1/2.
    "baseline_bm4bk18": dict(FORCE_BLOCKING=(4, 18)),
    "nslice": dict(NSLICE_OUTER=True),
    "nslice_split": dict(NSLICE_OUTER=True, NSLICE_SPLIT_NOCS=True),
    "nslice_split_alt": dict(NSLICE_OUTER=True, NSLICE_SPLIT_NOCS=True, NSLICE_ORDER="alternate"),
    "nslice_free": dict(NSLICE_OUTER=True, NSLICE_SPLIT_NOCS=True, NSLICE_PIN_BASELINE_BLOCKING=False),
    "nslice_reader": dict(NSLICE_OUTER=True, NSLICE_SENDER="reader"),
    "nslice_reader_w3": dict(NSLICE_OUTER=True, NSLICE_SENDER="reader", NSLICE_WEIGHT_DEPTH=3),
    "nslice_split_w3": dict(NSLICE_OUTER=True, NSLICE_SENDER="split", NSLICE_WEIGHT_DEPTH=3),
    "nslice_reader_pf": dict(
        NSLICE_OUTER=True, NSLICE_SENDER="reader", NSLICE_WEIGHT_DEPTH=3, NSLICE_WEIGHT_PREFETCH=True
    ),
    "nslice_split_pf": dict(
        NSLICE_OUTER=True, NSLICE_SENDER="split", NSLICE_WEIGHT_DEPTH=3, NSLICE_WEIGHT_PREFETCH=True
    ),
    # Weights two rounds ahead of the writer's next send (depth 4 keeps the prefetch's extra slot free).
    "nslice_split_la2": dict(
        NSLICE_OUTER=True,
        NSLICE_SENDER="split",
        NSLICE_WEIGHT_DEPTH=4,
        NSLICE_WEIGHT_PREFETCH=True,
        NSLICE_WEIGHT_LOOKAHEAD=2,
    ),
    "nslice_dir_la2": dict(
        NSLICE_OUTER=True,
        NSLICE_SENDER="direction",
        NSLICE_ORDER="alternate",
        NSLICE_WEIGHT_DEPTH=4,
        NSLICE_WEIGHT_PREFETCH=True,
        NSLICE_WEIGHT_LOOKAHEAD=2,
    ),
    # The recommended config with the L1 solve free (aspp.convs.1/2 do not fit at the baseline's bm).
    "nslice_dir_la2_free": dict(
        NSLICE_OUTER=True,
        NSLICE_SENDER="direction",
        NSLICE_ORDER="alternate",
        NSLICE_WEIGHT_DEPTH=4,
        NSLICE_WEIGHT_PREFETCH=True,
        NSLICE_WEIGHT_LOOKAHEAD=2,
        NSLICE_PIN_BASELINE_BLOCKING=False,
    ),
    # Recommended: the candidate at the baseline's bm with the coarsest bk that fits; the baseline scheme where the
    # resident in0 does not fit at that bm.
    "nslice_rec": dict(
        NSLICE_OUTER=True,
        NSLICE_SENDER="direction",
        NSLICE_ORDER="alternate",
        NSLICE_WEIGHT_DEPTH=4,
        NSLICE_WEIGHT_PREFETCH=True,
        NSLICE_WEIGHT_LOOKAHEAD=2,
        NSLICE_PIN_BASELINE_BLOCKING="bm",
        NSLICE_FALLBACK=True,
    ),
    "nslice_reader_la2": dict(
        NSLICE_OUTER=True,
        NSLICE_SENDER="reader",
        NSLICE_WEIGHT_DEPTH=4,
        NSLICE_WEIGHT_PREFETCH=True,
        NSLICE_WEIGHT_LOOKAHEAD=2,
    ),
}


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()


def _block_cases():
    """Every BLOCK LOOSE case (focus "[perf]" rows first), id from the layer name."""
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if c["memory_layout"] != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            continue
        layer = c["extras"]["layer"]
        cid = layer.split(" ")[0].replace(".", "_").replace("{", "").replace("}", "")
        if "[perf]" not in layer:
            # The HiFi4 golden rows (and the one non-perf LoFi twin) of the same layers.
            cid += "_lofi" if "@ LoFi" in layer else "_hifi4"
        out.setdefault(cid, c)
    return out


CASES = _block_cases()
FOCUS = ["res5_1_conv2", "aspp_convs_1", "res4_0_conv2"]


def _record_blocking(monkeypatch, info):
    real = op_mod.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["bm"], info["bk"], info["bn"] = b.block_m_tiles, b.block_k_tiles, b.block_n_tiles
        info["num_k_blocks"], info["k_cols"], info["out_n"] = b.num_k_blocks, g.k_cols, g.out_n_tiles
        info["l1_budget"] = l1_budget
        info["footprint"] = b.footprint_bytes
        info["tap_major"] = bool(pd.tap_major_im2col(g, b))
        info["sb"] = (b.out_subblock_h, b.out_subblock_w)
        info["nslice"] = bool(b.nslice)
        # The same (bm, bk) under each scheme, for the before/after ledger.
        saved = pd.NSLICE_OUTER
        for name, flag in (("footprint_baseline_scheme", False), ("footprint_nslice_scheme", True)):
            pd.NSLICE_OUTER = flag
            info[name] = pd._footprint(g, b.core_m_tiles, b.block_m_tiles, b.block_k_tiles, b.block_n_tiles)[
                "footprint_bytes"
            ]
        pd.NSLICE_OUTER = saved
        return b

    monkeypatch.setattr(op_mod, "select_blocking", recording)


def run_case(case, device, monkeypatch, seed=0):
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
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
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


ABLATIONS = ["none", "gather", "gather+matmul", "gather+weights", "gather+scatter", "gather+weights+scatter"]


@pytest.mark.parametrize("ablate", ABLATIONS)
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case_id", list(CASES))
def test_nslice_perf(case_id, variant, ablate, device, monkeypatch):
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    monkeypatch.setenv("CONV2D_3X3_ABLATE", "" if ablate == "none" else ablate.replace("+", ","))
    info = dict(case=case_id, variant=variant, ablate=ablate)
    _record_blocking(monkeypatch, info)
    tt_output, ref = run_case(case, device, monkeypatch)
    error = None
    if ablate == "none":
        tolerance = case["extras"].get("tolerance") or (0.99, 0.12)
        try:
            m = check_output(
                tt_output,
                ref.float(),
                shape=list(ref.shape),
                dtype=case["dtype"],
                expected_layout=ttnn.TILE_LAYOUT,
                tolerance=tolerance,
            )
        except CheckOutputError as e:
            m, error = e.metrics, str(e)
        info.update(pcc=m.pcc, rms=m.rms, max_abs=m.max_abs_diff, tolerance=list(tolerance), error=error)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert error is None, error


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "baseline"])
@pytest.mark.parametrize("case_id", list(CASES))
def test_nslice_vs_baseline(case_id, variant, device, monkeypatch):
    """Same inputs through the baseline and the candidate: max |diff| between the two outputs (summation order is
    the only legitimate difference) plus both against torch."""
    import torch

    case = CASES[case_id]
    outs = {}
    for v in ("baseline", variant):
        apply_variant(v, monkeypatch)
        monkeypatch.setenv("CONV2D_3X3_ABLATE", "")
        tt_output, ref = run_case(case, device, monkeypatch, seed=1)
        outs[v] = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    diff = (outs["baseline"] - outs[variant]).abs()
    info = dict(
        case=case_id,
        variant=variant,
        kind="vs_baseline",
        max_abs_vs_baseline=float(diff.max()),
        frac_differing=float((diff > 0).float().mean()),
        ref_absmax=float(ref.abs().max()),
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(info) + "\n")
    tolerance = case["extras"].get("tolerance") or (0.99, 0.12)
    assert torch.isfinite(outs[variant]).all()
    # Summation order only: allow a few fp32/bf8 output ulps relative to the output scale.
    assert float(diff.max()) <= 0.02 * float(ref.abs().max()) + 1e-3, info

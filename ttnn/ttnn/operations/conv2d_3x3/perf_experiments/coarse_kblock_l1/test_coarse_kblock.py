# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: coarser K-blocks for the K-blocked HEIGHT walk (perf experiment, not part of the op).

Runs a private copy of conv2d_3x3 (op_copy/) whose L1 quanta and K walk are switched by module knobs of the copied
program descriptor (CKB_*). Variants:
  baseline      the committed op (bm 5, bk 8 on the focus shape: 18 K-blocks per core, 720 fp32 L1-acc tile packs).
  ckb           (the L1 solve takes the CKB quanta only where the committed quanta do not fit) weights streamed in N-chunks of one output-subblock column (bk x out_subblock_w tiles per multicast
                round, CB depth n_chunks + 1 chunks), cb_im2col_rm in tap quanta (depth 2 taps), a 16-pixel zero source,
                and a raw K walk (N-chunk outer, M-subblock inner, caller-owned fp32 partials, pack format and L1-acc
                state set once per K-block). The L1 solve is unchanged apart from those terms, so it is free to pick a
                coarser block: the focus shape gets bm 10, bk 8 (one M-block per core).
  ckb_bm5bk24   the same machinery, blocking forced to bm 5, bk 24 (one K-block per kernel row, 6 per core).
  ckb_bm5bk8    the same machinery at the baseline blocking (attribution: the walk rewrite alone).

Precision contract is the case's own (LoFi on the focus case, fp32 DEST, bf8_b): every variant runs it unchanged.
One dispatch per test, for `run_safe_pytest.sh --profile`. Each test appends a JSON line (blocking, L1 footprint, PCC)
to results.jsonl next to this file; the profiler CSV gives DEVICE KERNEL DURATION.
`CONV2D_3X3_ABLATE` = the ablation hooks' tokens (output wrong by construction, no PCC checked).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.metrics import CheckOutputError, check_output

# No __init__.py anywhere here: ttnn.operations walk_packages-executes every package module on `import ttnn`, so the
# copy is loaded by path under a private package name.
OPC = "coarse_kblock_l1_op_copy"


def _load_op_copy():
    if OPC not in sys.modules:
        root = Path(__file__).parent / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "_pkg.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC]


opc = _load_op_copy()
pd = sys.modules[OPC + ".conv2d_3x3_program_descriptor"]
op_mod = sys.modules[OPC + ".conv2d_3x3"]

RESULTS = Path(__file__).parent / "results.jsonl"

KNOB_DEFAULTS = dict(
    CKB_ENABLE=False,
    CKB_FORCE_BLOCKING=None,
    CKB_ZERO_SRC_PX=32,
    CKB_INPUT_PIPELINE=False,
    CKB_INPUT_DEPTH=None,
    CKB_FORCE_MODE=False,
)
VARIANTS = {
    "baseline": {},
    "ckb": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=16),
    "ckb_bm5bk24": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=16, CKB_FORCE_BLOCKING=(5, 24)),
    "ckb_bm5bk8": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=16, CKB_FORCE_BLOCKING=(5, 8), CKB_FORCE_MODE=True),
    # The CKB quanta wherever they apply (sweep 1 ran this as "ckb", before the solve gated them).
    "ckb_always": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=16, CKB_FORCE_MODE=True),
    "ckb_pipe": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=16, CKB_INPUT_PIPELINE=True),
    "ckb_bm5bk24_pipe": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=16, CKB_FORCE_BLOCKING=(5, 24), CKB_INPUT_PIPELINE=True),
    # A three-batch input ring (two batches in flight), paid for by an 8-pixel zero source.
    "ckb_pipe3": dict(CKB_ENABLE=True, CKB_ZERO_SRC_PX=8, CKB_INPUT_PIPELINE=True, CKB_INPUT_DEPTH=3),
    "ckb_bm5bk24_pipe3": dict(
        CKB_ENABLE=True, CKB_ZERO_SRC_PX=8, CKB_FORCE_BLOCKING=(5, 24), CKB_INPUT_PIPELINE=True, CKB_INPUT_DEPTH=3
    ),
    # The baseline's own blocking forced to bm 2, bk 24 (fits as the CBs are sized today; 5 M-blocks per core).
    "baseline_bm2bk24": dict(CKB_FORCE_BLOCKING=(2, 24)),
}


# Variants the sweep tests run (tracy's `python -m tracy` loses -k quoting, so the selection is an env var).
SWEEP_VARIANTS = [v for v in os.environ.get("CKB_SWEEP_VARIANTS", "").split(",") if v] or list(VARIANTS)


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    pd._block_extents.cache_clear()


def _case_id(layer):
    import re

    name = layer.split(" @")[0].split(" + ")[0].split(" (")[0]
    return (
        re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        + ("_perf" if "[perf]" in layer else "")
        + ("_lofi" if "@ LoFi" in layer and "[perf]" not in layer else "")
    )


def _loose_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        out.setdefault(_case_id(c["extras"]["layer"]), c)
    return out


CASES = _loose_cases()
FOCUS = "semantic_res2_fuse_conv_1_perf"
HEIGHT_IDS = [k for k, c in CASES.items() if c["memory_layout"] == ttnn.TensorMemoryLayout.HEIGHT_SHARDED]
BLOCK_IDS = [k for k, c in CASES.items() if c["memory_layout"] == ttnn.TensorMemoryLayout.BLOCK_SHARDED]


def _record_blocking(monkeypatch, info):
    real = op_mod.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info.update(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            bn=b.block_n_tiles,
            num_k_blocks=b.num_k_blocks,
            m_blocks=b.max_m_blocks_per_core,
            ub=b.untilize_batch_rows,
            slab_rows=b.slab_tile_rows,
            sb=[b.out_subblock_h, b.out_subblock_w],
            l1_budget=l1_budget,
            footprint=b.footprint_bytes,
            tap_major=bool(pd.tap_major_im2col(g, b)),
            nslice=bool(b.nslice),
            ckb=bool(pd.ckb_walk(g, b)),
            weight_chunks=pd.weight_n_chunks(g, b),
        )
        return b

    monkeypatch.setattr(op_mod, "select_blocking", recording)


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


def _run_and_record(case_id, variant, device, monkeypatch):
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    ablate = os.environ.get("CONV2D_3X3_ABLATE", "")
    info = dict(case=case_id, variant=variant, ablate=ablate)
    _record_blocking(monkeypatch, info)
    tt_output, ref = run_case(case, device)
    error = None
    if not ablate:
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


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_focus(variant, device, monkeypatch):
    _run_and_record(FOCUS, variant, device, monkeypatch)


@pytest.mark.parametrize("variant", SWEEP_VARIANTS)
@pytest.mark.parametrize("case_id", HEIGHT_IDS + BLOCK_IDS)
def test_sweep(case_id, variant, device, monkeypatch):
    _run_and_record(case_id, variant, device, monkeypatch)


@pytest.mark.parametrize("variant", [v for v in SWEEP_VARIANTS if v != "baseline"])
@pytest.mark.parametrize("case_id", HEIGHT_IDS + BLOCK_IDS)
def test_vs_baseline(case_id, variant, device, monkeypatch):
    """Same inputs through the baseline and the candidate: the max |diff| between the two outputs (only where the fp32
    partial sums are rounded differs) and both against torch."""
    import torch

    case = CASES[case_id]
    outs, pccs = {}, {}
    for v in ("baseline", variant):
        apply_variant(v, monkeypatch)
        tt_output, ref = run_case(case, device, seed=1)
        outs[v] = ttnn.to_torch(tt_output).float().reshape(ref.shape)
        m = check_output(
            tt_output,
            ref.float(),
            shape=list(ref.shape),
            dtype=case["dtype"],
            expected_layout=ttnn.TILE_LAYOUT,
            tolerance=(0.0, 1e9),
        )
        pccs[v] = m.pcc
    diff = (outs["baseline"] - outs[variant]).abs()
    info = dict(
        case=case_id,
        variant=variant,
        kind="vs_baseline",
        pcc_baseline=pccs["baseline"],
        pcc_variant=pccs[variant],
        max_abs_vs_baseline=float(diff.max()),
        frac_differing=float((diff > 0).float().mean()),
        ref_absmax=float(ref.abs().max()),
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert torch.isfinite(outs[variant]).all()
    tolerance = case["extras"].get("tolerance") or (0.99, 0.12)
    assert pccs[variant] >= tolerance[0], info


# DRAM-interleaved perf shapes (the committed perf harness's test_conv2d_3x3_perf): ((H, W, C_in), C_out, padding).
INTERLEAVED_SHAPES = [
    ((32, 64, 128), 128, 1),
    ((16, 16, 64), 128, 1),
    ((8, 16, 512), 256, 1),
    ((8, 16, 2048), 256, 1),
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
]
INTERLEAVED_LAYOUTS = [
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="rm_bf16"),
    pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="tile_bf16"),
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="tile_bf8b"),
]


@pytest.mark.parametrize("variant", SWEEP_VARIANTS)
@pytest.mark.parametrize("dtype,layout", INTERLEAVED_LAYOUTS)
@pytest.mark.parametrize(
    "shape", INTERLEAVED_SHAPES, ids=[f"H{s[0][0]}_W{s[0][1]}_Ci{s[0][2]}_Co{s[1]}" for s in INTERLEAVED_SHAPES]
)
def test_interleaved(shape, dtype, layout, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    info = dict(
        case="interleaved_" + "_".join(map(str, shape[0])) + f"_{shape[1]}", variant=variant, layout=str(layout)
    )
    info["dtype"] = str(dtype)
    _record_blocking(monkeypatch, info)
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding = shape
    kwargs = dict(
        stride=1,
        padding=padding,
        dilation=1,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_output = opc.conv2d_3x3(tt_input, tt_weight, input_height=H, input_width=W, bias=tt_bias, padding=padding)
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), padding=padding)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    error = None
    try:
        m = check_output(
            tt_output,
            ref,
            shape=list(ref.shape),
            dtype=dtype,
            expected_layout=ttnn.TILE_LAYOUT,
            tolerance=(0.995 if dtype == ttnn.bfloat16 else 0.99, 1e9),
        )
    except CheckOutputError as e:
        m, error = e.metrics, str(e)
    info.update(pcc=m.pcc, max_abs=m.max_abs_diff, error=error)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert error is None, error

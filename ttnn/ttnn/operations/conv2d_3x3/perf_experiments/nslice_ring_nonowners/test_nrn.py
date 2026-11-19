# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: N-slice ring forwarding granularity (perf experiment, round 10, `nslice_ring_nonowners`).

Every variant runs on this dir's op copy (op_copy/: HEAD 6981db2a54c + this experiment's knobs; NRN_PKG=op_copy_nz
selects the zone-stripped copy made by mk_nz.sh), whose knobs default to the committed op's behaviour (variant
`baseline`). Never profile it in one process with the real op (zone-name hash collisions).

Env: NRN_CASES (focus ids or `block:<layer>`, comma list; `domain` = the brief's sweep), NRN_VARIANTS (comma list, run in
that order per case), NRN_ORDER=alt (reverse the order on every other case), NRN_LOG (run-log name), NRN_NOCHECK=1. One
dispatch per test, for `run_safe_pytest.sh --profile`; each dispatch appends a JSON record (blocking, PCC, mean / max
abs error vs a fp32 torch conv, output hash, and the output itself as a .pt for max-abs-diff vs the baseline) to
runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp rows in call order. The precision contract is
the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it. Inputs are seeded per case.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_ring_nonowners."
_PKG = _BASE + os.environ.get("NRN_PKG", "op_copy")
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / os.environ.get("NRN_PKG", "op_copy"), pd.KERNEL_DIR
# Variant `grad`: the graduation patch applied to HEAD (grad/, or NRN_GRAD_PKG, e.g. grad_nz), knob-free.
_GRAD_PKG = _BASE + os.environ.get("NRN_GRAD_PKG", "grad")
_MODS = {"op": op, "pd": pd}

KNOB_DEFAULTS = dict(
    RING_ROW_TOKENS=False, RING_TOKEN_PAGES=1, RING_INFLIGHT=2, RING_EXTRA_DEFINES=[], RING_SPREAD_OWNERS=False
)
VARIANTS = {
    "baseline": {},
    # Group units (HEAD's forwarding granularity), a 2-page token CB.
    "p2": dict(RING_TOKEN_PAGES=2),
    # One token per tile-row where a row group is one subblock, 1-page token CB.
    "row1": dict(RING_ROW_TOKENS=True),
    # Tile-row tokens, 2-page token CB (the packer runs a unit ahead of the reader's forwarding).
    "row2": dict(RING_ROW_TOKENS=True, RING_TOKEN_PAGES=2),
    "row3": dict(RING_ROW_TOKENS=True, RING_TOKEN_PAGES=3),
    # Row gate: per-row gating of the L1-acc packs by cumulative reserves, one push burst (one drain) per group; the
    # reader forwards each row as its own write and arrival (4-page token CB, all dummies before the walk).
    "rowgate": dict(RING_ROW_TOKENS=2, RING_TOKEN_PAGES=4),
    "rowgate_if4": dict(RING_ROW_TOKENS=2, RING_TOKEN_PAGES=4, RING_INFLIGHT=4),
    "rowgate_if8": dict(RING_ROW_TOKENS=2, RING_TOKEN_PAGES=4, RING_INFLIGHT=8),
    "row2_if4": dict(RING_ROW_TOKENS=True, RING_TOKEN_PAGES=2, RING_INFLIGHT=4),
    # Diagnostics on the unit protocol with group units (p2 = the same protocol, HEAD's writes):
    # 4 KB packets under one transaction id and one arrival per group,
    "split4k": dict(RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_PACKET_BYTES", "4096")]),
    # one write, transaction id and arrival per tile-row (group tokens),
    "rowarr": dict(RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_ROW_ARRIVALS", "1")]),
    "rowarr_if4": dict(RING_TOKEN_PAGES=2, RING_INFLIGHT=4, RING_EXTRA_DEFINES=[("RING_ROW_ARRIVALS", "1")]),
    # Inline absolute-count arrival writes instead of NoC atomic increments.
    "p2_inl": dict(RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_INLINE_ARRIVAL", "1")]),
    "p1_inl": dict(RING_TOKEN_PAGES=1, RING_EXTRA_DEFINES=[("RING_INLINE_ARRIVAL", "1")]),
    "rowarr_inl": dict(
        RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_ROW_ARRIVALS", "1"), ("RING_INLINE_ARRIVAL", "1")]
    ),
    "rowgate_inl": dict(RING_ROW_TOKENS=2, RING_TOKEN_PAGES=4, RING_EXTRA_DEFINES=[("RING_INLINE_ARRIVAL", "1")]),
    "row2_inl": dict(RING_ROW_TOKENS=True, RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_INLINE_ARRIVAL", "1")]),
    # Arrival signalled once the unit's write has left the core (not once its ack returned).
    "p1_sent": dict(RING_TOKEN_PAGES=1, RING_EXTRA_DEFINES=[("RING_SENT_SIGNAL", "1")]),
    "p2_sent": dict(RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_SENT_SIGNAL", "1")]),
    "rowgate_sent": dict(RING_ROW_TOKENS=2, RING_TOKEN_PAGES=4, RING_EXTRA_DEFINES=[("RING_SENT_SIGNAL", "1")]),
    "row2_sent": dict(RING_ROW_TOKENS=True, RING_TOKEN_PAGES=2, RING_EXTRA_DEFINES=[("RING_SENT_SIGNAL", "1")]),
    # NOT bit-identical: non-owning columns spread round the ring (HEAD's protocol otherwise); remote output store only.
    "spread": dict(RING_SPREAD_OWNERS=True),
    "grad": {},
}


def apply_variant(variant, monkeypatch):
    global op, pd
    if variant == "grad":
        op = importlib.import_module(_GRAD_PKG + ".conv2d_3x3")
        pd = importlib.import_module(_GRAD_PKG + ".conv2d_3x3_program_descriptor")
        assert "nslice_ring_nonowners/grad" in str(pd.KERNEL_DIR), pd.KERNEL_DIR
    else:
        op, pd = _MODS["op"], _MODS["pd"]
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        if hasattr(pd, k):
            monkeypatch.setattr(pd, k, v)
        elif k in VARIANTS[variant]:
            raise AttributeError(f"op copy has no knob {k}")
    if hasattr(pd.select_blocking, "cache_clear"):
        pd.select_blocking.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = c
    return out


FOCUS = _focus_cases()
DOMAIN = [
    "aspp_convs_3",
    "aspp_convs_1",
    "aspp_convs_2",
    "res4_{1__5}_conv2",
    "res5_0_conv2",
]
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride): the HiFi4
# BLOCK shapes of test_conv2d_3x3_block_sharded_perf.
BLOCK_SHAPES = {
    "res4_0_conv2": (((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    "res4_conv2": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    "res5_0_conv2": (((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_1_conv2": (((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_2_conv2": (((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, "HiFi4", 1),
    "aspp_convs1": (((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs2": (((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs3": (((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, "HiFi4", 1),
    # Not Panoptic: an odd k_cols (7 columns) ring with C_out 224 (7 owners) and bias, LoFi.
    "plain_k7": (((32, 64, 224), 224, 1, 1), [256, 32], (7, 8), True, "LoFi", 1),
    # Not Panoptic: 11 columns, C_out 160 = 5 owners of 1 tile (6 non-owning columns), no bias, LoFi.
    "plain_k11_o5": (((32, 64, 352), 160, 2, 2), [256, 32], (11, 8), False, "LoFi", 1),
}


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("NRN_CASES", "aspp_convs_3").split(",") if c]:
        if c == "domain":
            out += DOMAIN
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("NRN_VARIANTS", "baseline,row2").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("NRN_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("NRN_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("NRN_NOCHECK") == "1"


def _run_focus(case, device):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
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
    pcc = (extras.get("tolerance") or (0.99, None))[0]
    return tt_output, ref, pcc, memory_config


def _run_block(layer, device):
    import torch

    torch.manual_seed(0)
    ((H, W, c_in), c_out, padding, dilation), shard_shape, (gx, gy), bias_relu, fidelity, stride = BLOCK_SHAPES[layer]
    grid = device.compute_with_storage_grid_size()
    if gx > grid.x or gy > grid.y:
        pytest.skip(f"{layer}: pinned {gx}x{gy} shard grid does not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if bias_relu else None
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if bias_relu else None
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if bias_relu else None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True, math_approx_mode=False
        ),
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
    if bias_relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return tt_output, ref, 0.99, memory_config


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            nkb=b.num_k_blocks,
            nslice=bool(b.nslice),
            k_cols=g.k_cols,
            out_n=g.out_n_tiles,
            n_tiles=g.tensor_n_tiles,
            remote=bool(g.remote_output),
            ranges=[list(r) for r in g.m_ranges] if g.m_ranges else None,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_nrn(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold, in_mc = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold, in_mc = _run_focus(FOCUS[case_id], device)
    out_mc = tt_output.memory_config()
    assert out_mc.memory_layout == in_mc.memory_layout
    assert str(out_mc.shard_spec.grid) == str(in_mc.shard_spec.grid)
    assert list(out_mc.shard_spec.shape) == list(out_mc.shard_spec.shape)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    sha = hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()[:16]
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        mae=float((out - ref.float()).abs().mean()),
        max_err=float((out - ref.float()).abs().max()),
        out_sha=sha,
        out_shard=list(out_mc.shard_spec.shape),
    )
    outs = HERE / "runs" / "outs"
    outs.mkdir(parents=True, exist_ok=True)
    torch.save(out.to(torch.bfloat16), outs / f"{case_id.replace(':', '_')}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"

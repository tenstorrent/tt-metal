# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: overlap the BLOCK TILE prelude's slab fill with its untilize (perf experiment block_fill_untilize_overlap,
round Perf 9).

Every variant runs on this dir's op copy (op_copy/: HEAD a51305feccb + the r9 ablation hooks + this experiment's
knobs), whose knobs default to the committed op's behaviour (variant `baseline`). Never profile it in one process with
the real op (zone-name hash collisions).

Env: FUO_CASES (focus ids, `block:<layer>`), FUO_VARIANTS (comma list, run in that order per case), FUO_ORDER=alt
(alternate the variant order per case), FUO_LOG. One dispatch per test, for `run_safe_pytest.sh --profile`; each
dispatch appends a JSON record (blocking, PCC, output hash) to runs/<log>.jsonl, which report.py joins with the profiler
CSV's GenericOp rows in call order. The precision contract is fixed per case (its fidelity, fp32 DEST, its dtypes): no
variant touches it. A pure data-movement change, so every variant must be bit-identical to `baseline`.
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
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.block_fill_untilize_overlap." + os.environ.get("FUO_PKG", "op_copy")
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent.parent == HERE, pd.KERNEL_DIR

KNOB_DEFAULTS = dict(
    FUO_DEFINES=(),
    FUO_UB=None,
    FILL_WINDOW_BATCHES=4,
    FILL_WRITER_WINDOW_BATCHES=4,
    FUO_KSPLIT_WINDOW=0,
    FUO_KSPLIT_WRITER_WINDOW=0,
)
VARIANTS = {
    "baseline": {},
    "diag": dict(FUO_DEFINES=("FUO_DIAG",)),
}
# K-split-only candidates: batch cap u, reader window w, writer window ww (HEIGHT / interleaved keep the op's knobs).
for _ub in (2, 3, 4):
    for _w in (4, 8):
        for _ww in (4, 8):
            _knobs = dict(FUO_UB=_ub, FUO_KSPLIT_WINDOW=_w, FUO_KSPLIT_WRITER_WINDOW=_ww)
            VARIANTS[f"k{_ub}w{_w}ww{_ww}"] = _knobs
            VARIANTS[f"dk{_ub}w{_w}ww{_ww}"] = dict(_knobs, FUO_DEFINES=("FUO_DIAG",))
            VARIANTS[f"ownk{_ub}w{_w}ww{_ww}"] = dict(_knobs, FUO_DEFINES=("FUO_OWN_NOC1",))
# Bounds (diagnostic, wrong output): the padded-slab center copy and gap zeroing stubbed.
VARIANTS["stubcenter"] = dict(FUO_DEFINES=("FUO_STUB_CENTER",))
VARIANTS["stubcenterk4"] = dict(
    FUO_DEFINES=("FUO_STUB_CENTER",), FUO_UB=4, FUO_KSPLIT_WINDOW=8, FUO_KSPLIT_WRITER_WINDOW=4
)
# Reader window w (batches in flight), batch cap ub, writer window ww; "d" prefix = with the diagnostic zones.
for _ub in (None, 1, 2, 3, 4, 6):
    for _w in (1, 2, 3, 4, 6, 8):
        for _ww in (1, 2, 4):
            _name = ("" if _ub is None else f"ub{_ub}") + f"w{_w}" + ("" if _ww == 4 else f"ww{_ww}")
            _knobs = dict(FUO_UB=_ub, FILL_WINDOW_BATCHES=_w, FILL_WRITER_WINDOW_BATCHES=_ww)
            VARIANTS[_name] = _knobs
            VARIANTS["d" + _name] = dict(_knobs, FUO_DEFINES=("FUO_DIAG",))
            VARIANTS["l" + _name] = dict(_knobs, FUO_DEFINES=("FUO_DIAG", "FUO_DIAG2"))
            # The split fill's writer also reads the core's own shard (NoC1 self-reads).
            VARIANTS["own" + _name] = dict(_knobs, FUO_DEFINES=("FUO_OWN_NOC1",))
            VARIANTS["down" + _name] = dict(_knobs, FUO_DEFINES=("FUO_DIAG", "FUO_OWN_NOC1"))
            VARIANTS["lown" + _name] = dict(_knobs, FUO_DEFINES=("FUO_DIAG", "FUO_DIAG2", "FUO_OWN_NOC1"))


def apply_variant(variant, monkeypatch):
    # FUO_PKG=grad: the real op + graduation.patch, no knobs (variant `baseline` only).
    assert os.environ.get("FUO_PKG") != "grad" or variant == "baseline"
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        if hasattr(pd, k):
            monkeypatch.setattr(pd, k, v)
    # select_blocking is lru_cached on the geometry only: a knob must not hit a cached solve.
    for _f in vars(pd).values():
        if hasattr(_f, "cache_clear"):
            _f.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = c
    return out


FOCUS = _focus_cases()
# The BLOCK guard set of tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py (copied, not imported).
# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
BLOCK_SHAPES = {
    "res4_0_conv2": (((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    "res4_conv2": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    "res4_conv2_lofi": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "LoFi", 1),
    "res5_0_conv2": (((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_1_conv2": (((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_2_conv2": (((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, "HiFi4", 1),
    "aspp_convs1": (((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs2": (((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs3": (((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, "HiFi4", 1),
}
FOCUS_DEFAULT = ["aspp_convs_2", "aspp_convs_1"]
DOMAIN = [
    "aspp_convs_1",
    "aspp_convs_2",
    "aspp_convs_3",
    "res5_0_conv2",
    "res5_1_conv2",
    "res5_2_conv2",
    "res4_{1__5}_conv2",
    "res4_0_conv2",
]


def _sel_cases():
    raw = [c for c in os.environ.get("FUO_CASES", "aspp_convs_2").split(",") if c]
    out = []
    for c in raw:
        if c == "focus":
            out += FOCUS_DEFAULT
        elif c == "domain":
            out += DOMAIN
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("FUO_VARIANTS", "baseline").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("FUO_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("FUO_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


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
    return tt_output, ref, pcc


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
    return tt_output, ref, 0.99


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            nkb=b.num_k_blocks,
            nslice=bool(b.nslice),
            ub=b.untilize_batch_rows,
            slab=b.slab_tile_rows,
            mblk=b.max_m_blocks_per_core,
            ring=bool(b.ring),
            center=b.center_mode,
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)
    real_ranges = pd.core_m_ranges

    def ranges(g, core_m_tiles):
        r = real_ranges(g, core_m_tiles)
        info["m_ranges"] = [list(x) for x in r]
        return r

    monkeypatch.setattr(pd, "core_m_ranges", ranges)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_fuo(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold = _run_focus(FOCUS[case_id], device)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
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
    if os.environ.get("CONV2D_3X3_PERF_ABLATION") != "1":
        assert info["finite"] and passed, info

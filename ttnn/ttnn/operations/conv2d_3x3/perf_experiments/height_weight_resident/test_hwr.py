# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: keep every K-block's weights resident in cb_weights (perf experiment height_weight_resident,
round 7).

Runs a private copy of conv2d_3x3 (op_copy/, the op at e40c435b625) whose L1 solve takes the HWR_* knobs of
op_copy/conv2d_3x3_program_descriptor.py. Variant `base` leaves them off and is the op unchanged.

The precision contract is each case's own (fp32 DEST, its fidelity, its dtypes): every variant runs it unchanged. A
deeper cb_weights only moves data-movement work in time, so the outputs must be bit-identical to base
(test_hwr_identical).

One dispatch per test_hwr_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  HWR_CASES=res3_x HWR_SEQ=base,last HWR_TAG=t1 scripts/run_safe_pytest.sh --profile --run-all \\
      <this file>::test_hwr_perf -p no:cacheprovider --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import hwr_load  # noqa: E402

opc, pd, op_mod = hwr_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"

VARIANTS = {
    "base": dict(HWR_MODE="off"),
    "last": dict(HWR_MODE="last", HWR_TARGET="kblocks"),
    "first": dict(HWR_MODE="first", HWR_TARGET="kblocks"),
    "rounds": dict(HWR_MODE="last", HWR_TARGET="rounds"),
    "kwf": dict(HWR_MODE="off", HWR_KWF=True),
    "lastkwf": dict(HWR_MODE="last", HWR_TARGET="kblocks", HWR_KWF=True),
}

# Short names of the 22 [perf] LOOSE cases (order of feature_spec), as the coordinator's brief uses them.
SHORT = [
    "stem1",
    "stem2",
    "stem3",
    "res2",
    "res3_0",
    "res3_x",
    "res4_0",
    "res4_x",
    "res5_0",
    "res5_1",
    "res5_2",
    "aspp1",
    "aspp2",
    "aspp3",
    "semdec_r3_fc0",
    "instdec_r3_fc0",
    "semdec_r3_fc1",
    "semdec_r2_fc0",
    "instdec_r2_fc0",
    "sem_r2_fc1",
    "inst_r2_fc1",
    "center1",
]

# Interleaved (DRAM) shapes of the perf harness (PERF_SHAPES): the in-grid base sender's K-blocked walk. bf16 weights,
# bias, no ReLU (harness config).
INTERLEAVED = [
    ((32, 64, 128), 128, 1),
    ((64, 128, 128), 128, 1),
    ((8, 16, 512), 256, 1),
    ((128, 256, 64), 64, 1),
]
DRAM_LAYOUTS = [
    ("rm_bf16", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ("tile_bf8b", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    focus = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    assert len(focus) == len(SHORT), len(focus)
    for short, c in zip(SHORT, focus):
        (H, W, c_in), c_out, padding, stride, dilation = c["inputs"]
        out[short] = dict(
            focus=c,
            shape=((H, W, c_in), c_out, padding, stride, dilation),
            dtype=c["dtype"],
            layout=c["layout"],
            weight_dtype=c["weight_dtype"],
            bias=c["bias_mode"] == "with_bias",
            relu=c["activation"] == "relu",
            fidelity=c["extras"].get("math_fidelity", ttnn.MathFidelity.HiFi4),
            fp32_dest=c["fp32_dest_acc_en"],
            pcc=c["extras"].get("tolerance", (0.99, None))[0],
            golden=True,
        )
    for (H, W, c_in), c_out, padding in INTERLEAVED:
        for tag, dtype, layout in DRAM_LAYOUTS:
            out[f"dram_H{H}_W{W}_Ci{c_in}_Co{c_out}_{tag}"] = dict(
                shape=((H, W, c_in), c_out, padding, 1, 1),
                dtype=dtype,
                layout=layout,
                weight_dtype=ttnn.bfloat16,
                bias=True,
                relu=False,
                fidelity=None,
                fp32_dest=None,
                pcc=0.995 if dtype == ttnn.bfloat16 else 0.99,
                golden=False,
            )
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


LAST = {}


def apply_variant(variant, monkeypatch):
    for k, v in VARIANTS[variant].items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    real = pd.create_program_descriptor

    def capture(input_tensor, weight_tensor, *args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nb=blocking.max_m_blocks_per_core,
            nkb=blocking.num_k_blocks,
            wd=blocking.weight_depth,
            im2col=blocking.im2col_rm_depth_rows,
            ub=blocking.untilize_batch_rows,
            nslice=bool(blocking.nslice),
            ring=bool(blocking.ring),
            center=int(blocking.center_mode),
            foot_kb=round(blocking.footprint_bytes / 1024, 1),
        )
        desc = real(input_tensor, weight_tensor, *args, geometry=geometry, blocking=blocking, **kwargs)
        for cb in desc.cbs:
            for fd in cb.format_descriptors:
                if fd.buffer_index == pd.CB_WEIGHTS:
                    LAST["w_pages"] = cb.total_size // geometry.weight_tile_bytes
        LAST["servers"] = any("weight_server" in str(k.kernel_source) for k in desc.kernels)
        return desc

    real_kwf = pd.kw_fold_mode

    def kwf_capture(*args, **kwargs):
        mode = real_kwf(*args, **kwargs)
        LAST["kwf"] = bool(mode)
        return mode

    monkeypatch.setattr(op_mod, "create_program_descriptor", capture)
    monkeypatch.setattr(pd, "kw_fold_mode", kwf_capture)


def run_case(case, device, seed=0):
    import torch

    torch.manual_seed(seed)
    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    if case["golden"]:
        from eval.golden_tests.conv2d_3x3 import helpers

        c = case["focus"]
        memory_config = helpers.build_input_memory_config(
            (1, 1, H * W, c_in),
            c["memory_layout"],
            layout=case["layout"],
            dtype=case["dtype"],
            device=device,
            extras=c["extras"],
        )
        if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(c["extras"]["shard_shape"]):
            pytest.skip("pinned shard geometry does not fit this device's grid")
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=memory_config,
        device=device,
    )
    if case["golden"]:
        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if case["bias"] else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    else:
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias"] else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if case["bias"] else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=memory_config
    )
    config = None
    if case["golden"]:
        config = helpers.make_compute_kernel_config(case["fp32_dest"], case["fidelity"])
    tt_output = opc.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if case["relu"] else None,
        compute_kernel_config=config,
    )
    # fp32 torch reference (the golden helper computes in fp32 on bf16-rounded inputs).
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if case["relu"]:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _mae(a, b):
    return float((a.flatten().double() - b.flatten().double()).abs().mean())


PERF_CASES = _selected("HWR_CASES", ["res3_x"])
PERF_VARIANTS = _selected("HWR_VARIANTS", ["base"])
# HWR_SEQ: an explicit variant order per case (repeats allowed, e.g. base,last,last,base); else HWR_VARIANTS.
PERF_SEQ = _selected("HWR_SEQ", PERF_VARIANTS)


@pytest.mark.parametrize("seq", list(enumerate(PERF_SEQ)), ids=[f"{i}-{v}" for i, v in enumerate(PERF_SEQ)])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hwr_perf(case_id, seq, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against a fp32 torch conv."""
    variant = seq[1]
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
                    mae=_mae(out, ref),
                    tag=os.environ.get("HWR_TAG", ""),
                    **LAST,
                )
            )
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


BASE_VARIANT = os.environ.get("HWR_BASE", "base")


@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != BASE_VARIANT])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hwr_identical(case_id, variant, device, monkeypatch):
    """Same inputs through the base (HWR_BASE) and the candidate: the outputs must be bit-identical (and PCC-gated).
    The candidate runs twice (program-cache miss, then hit)."""
    import torch

    case = CASES[case_id]
    outs = {}
    for v in (BASE_VARIANT, variant, variant):
        apply_variant(v, monkeypatch)
        outs.setdefault(v, []).append((run_case(case, device, seed=1), dict(LAST)))
    (base_out, ref), base_blk = outs[BASE_VARIANT][0]
    results = []
    for (out, _), blk in outs[variant]:
        same = bool(torch.equal(base_out, out))
        results.append((same, _pcc(out, ref), _mae(out, ref), float((base_out - out).abs().max())))
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(
                    kind="identical",
                    case=case_id,
                    variant=variant,
                    base_pcc=_pcc(base_out, ref),
                    base_mae=_mae(base_out, ref),
                    results=results,
                    base_blocking=base_blk,
                    blocking=blk,
                )
            )
            + "\n"
        )
    for same, pcc, mae, diff in results:
        assert same, f"{case_id}/{variant}: output differs from base (max |diff| {diff}) {blk}"
        assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc}"

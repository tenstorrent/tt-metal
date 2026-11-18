# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: start the K-blocked walk earlier and keep the weights ahead of it (perf experiment
height_kblock_startup, round 5).

Runs a private copy of conv2d_3x3 (op_copy/, the op at 2ac42333eb2) whose kernels take variant switches as
preprocessor defines (descriptor knob HK_DEFINES). Variant `base` sets none of them and is the op unchanged.

Precision contract is each case's own (fp32 DEST, its fidelity, its dtypes): every variant runs it unchanged. The
variants only move data-movement work in time, so their outputs must be bit-identical to base (test_hk_identical).

One dispatch per test_hk_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  HK_CASES=semantic HK_VARIANTS=base,cand HK_TAG=t1 scripts/run_safe_pytest.sh --profile --run-all \\
      <this file>::test_hk_perf -p no:cacheprovider --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import hk_load  # noqa: E402

opc, pd, op_mod = hk_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"

# Variant -> preprocessor defines (see the kernels' HK_* switches).
Z = {"HK_ZONES": "1"}
LB = {"HK_LAZY_BIAS": "1"}
BL = {"HK_LAZY_BIAS": "1", "HK_BIAS_ROUND": "1"}
BLS = {"HK_LAZY_BIAS": "1", "HK_BIAS_ROUND": "1", "HK_BIAS_STAGGER": "6"}
EP = {"HK_SENDER_EARLY_PUSH": "1"}
VARIANTS = {
    "base": {},
    "zones": dict(Z),
    "lb_z": dict(Z, **LB),
    "bl_z": dict(Z, **BL),
    "bls_z": dict(Z, **BLS),
    "blsep_z": dict(Z, **BLS, **EP),
    "lb": dict(LB),
    "bl": dict(BL),
    "bls": dict(BLS),
    "blsep": dict(BLS, **EP),
    "pf_z": dict(Z, **BLS, HK_PREFETCH="1"),
    "pf": dict(BLS, HK_PREFETCH="1"),
    "pfonly_z": dict(Z, HK_PREFETCH="1"),
    "frbls_z": dict(Z, **BLS, HK_FASTREAD="1"),
    "fr_z": dict(Z, HK_FASTREAD="1"),
    "frbls": dict(BLS, HK_FASTREAD="1"),
    "cand": dict(BLS, **EP, HK_FASTREAD="1"),
    "cand_z": dict(Z, **BLS, **EP, HK_FASTREAD="1"),
    "candpe_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1"),
    "candub4_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", UB_CAP="4"),
    "candub3_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", UB_CAP="3"),
    "candpe": dict(BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1"),
    "candfp_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", HK_FILL_PIPE="1"),
    "candfp": dict(BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", HK_FILL_PIPE="1"),
    "candzl_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", HK_ZERO_LATE="1"),
    "candzl": dict(BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", HK_ZERO_LATE="1"),
    "candzl2_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_ZERO_LATE="1"),
    "candzl2": dict(BLS, **EP, HK_FASTREAD="1", HK_ZERO_LATE="1"),
    "cand1r_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", ONE_RECT="1"),
    "candzl1r_z": dict(Z, **BLS, **EP, HK_FASTREAD="1", HK_FILL_PLAN_EARLY="1", HK_ZERO_LATE="1", ONE_RECT="1"),
}

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _grid(num_cores, grid_x=11):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _height(shard, num_cores):
    return ttnn.MemoryConfig(
        HS, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(num_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    )


# tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py HEIGHT_SHAPES (the HEIGHT guard harness).
HEIGHT_SHAPES = [
    ("stem_conv1", ((512, 1024, 8), 64, 1, 2), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, [4767, 8], 110),
    ("stem_conv3", ((256, 512, 64), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [1216, 64], 110),
    ("res2_conv2", ((128, 256, 64), 64, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 64], 103),
    ("res3_0_conv2", ((128, 256, 128), 128, 1, 2), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [384, 128], 86),
    ("res3_conv2", ((64, 128, 128), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 128], 86),
    ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 288], 103),
    ("stem_conv2", ((256, 512, 64), 64, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [1216, 64], 110),
    ("sem_res3_fuse_conv0", ((64, 128, 320), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 320], 86),
    ("inst_res3_fuse_conv0", ((64, 128, 320), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 320], 86),
    ("sem_res3_fuse_conv1", ((64, 128, 256), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 256], 86),
    ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 160], 103),
    ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 256], 103),
    ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 128], 103),
    ("center_head1", ((128, 256, 128), 32, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 128], 103),
]

# Interleaved (DRAM) shapes of the perf harness (PERF_SHAPES), K-blocked walks with the weight multicast; bf16
# weights, bias, no ReLU (harness config).
INTERLEAVED = [
    ((32, 64, 128), 128, 1),
    ((8, 16, 2048), 256, 1),
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
    ((128, 256, 512), 256, 1),
]
DRAM_LAYOUTS = [
    ("rm_bf16", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ("tile_bf16", ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ("tile_bf8b", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" not in c["extras"]["layer"]:
            continue
        cid = c["extras"]["layer"].split(" ")[0].replace(".", "_")
        (H, W, c_in), c_out, padding, stride, dilation = c["inputs"]
        out[cid] = dict(
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
    for layer, ((H, W, c_in), c_out, padding, stride), dtype, layout, shard, cores in HEIGHT_SHAPES:
        out["h_" + layer] = dict(
            shape=((H, W, c_in), c_out, padding, stride, 1),
            dtype=dtype,
            layout=layout,
            weight_dtype=ttnn.bfloat8_b,
            mem=(lambda shard=shard, cores=cores: _height(shard, cores)),
            bias=True,
            relu=True,
            fidelity=None,
            fp32_dest=None,
            pcc=0.99,
            golden=False,
        )
    for (H, W, c_in), c_out, padding in INTERLEAVED:
        for tag, dtype, layout in DRAM_LAYOUTS:
            out[f"dram_H{H}_W{W}_Ci{c_in}_Co{c_out}_{tag}"] = dict(
                shape=((H, W, c_in), c_out, padding, 1, 1),
                dtype=dtype,
                layout=layout,
                weight_dtype=ttnn.bfloat16,
                mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
                bias=True,
                relu=False,
                fidelity=None,
                fp32_dest=None,
                pcc=0.995 if dtype == ttnn.bfloat16 else 0.99,
                golden=False,
            )
    # Same shapes without bias: the bias-free path of every candidate.
    out["dram_H128_W256_Ci512_Co256_tile_bf8b_nobias"] = dict(out["dram_H128_W256_Ci512_Co256_tile_bf8b"], bias=False)
    out["h_sem_res2_fuse_conv1_nobias"] = dict(out["h_sem_res2_fuse_conv1"], bias=False)
    out["h_center_head1_nobias_norelu"] = dict(out["h_center_head1"], bias=False, relu=False)
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


LAST = {}


def apply_variant(variant, monkeypatch):
    defines = dict(VARIANTS[variant])
    monkeypatch.setattr(pd, "HK_UB_CAP", int(defines.pop("UB_CAP", 0)))
    monkeypatch.setattr(pd, "HK_ONE_RECT", bool(int(defines.pop("ONE_RECT", 0))))
    monkeypatch.setattr(pd, "HK_DEFINES", defines)
    pd.select_blocking.cache_clear()
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nb=blocking.max_m_blocks_per_core,
            nkb=blocking.num_k_blocks,
            ub=blocking.untilize_batch_rows,
            nslice=bool(blocking.nslice),
            ring=bool(blocking.ring),
            center=int(blocking.center_mode),
            foot_kb=round(blocking.footprint_bytes / 1024, 1),
        )
        return real(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op_mod, "create_program_descriptor", capture)


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
        memory_config = case["mem"]()
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
    if case["golden"]:
        ref = helpers.pytorch_conv2d_3x3(
            torch_input,
            torch_weight,
            input_height=H,
            input_width=W,
            bias=torch_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation="relu" if case["relu"] else None,
        ).float()
        return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref
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


PERF_CASES = _selected("HK_CASES", ["semantic"])
PERF_VARIANTS = _selected("HK_VARIANTS", ["base"])
# HK_SEQ: an explicit variant order per case (repeats allowed, e.g. base,cand,cand,base,base,cand); else HK_VARIANTS.
PERF_SEQ = _selected("HK_SEQ", PERF_VARIANTS)


@pytest.mark.parametrize("seq", list(enumerate(PERF_SEQ)), ids=[f"{i}-{v}" for i, v in enumerate(PERF_SEQ)])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hk_perf(case_id, seq, device, monkeypatch):
    variant = seq[1]
    """One dispatch of one variant, PCC-gated against torch."""
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    out, ref = run_case(case, device)
    pcc = _pcc(out, ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(kind="perf", case=case_id, variant=variant, pcc=pcc, tag=os.environ.get("HK_TAG", ""), **LAST)
            )
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


BASE_VARIANT = os.environ.get("HK_BASE", "base")


@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != BASE_VARIANT])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hk_identical(case_id, variant, device, monkeypatch):
    """Same inputs through the base (HK_BASE) and the candidate: the outputs must be bit-identical (and PCC-gated).
    Each variant runs twice (program-cache miss, then hit)."""
    import torch

    case = CASES[case_id]
    outs = {}
    for v in (BASE_VARIANT, variant, variant):
        apply_variant(v, monkeypatch)
        outs.setdefault(v, []).append(run_case(case, device, seed=1))
    ref = outs[BASE_VARIANT][0][1]
    base_out = outs[BASE_VARIANT][0][0]
    results = []
    for out, _ in outs[variant]:
        same = bool(torch.equal(base_out, out))
        results.append((same, _pcc(out, ref), float((base_out - out).abs().max())))
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(dict(kind="identical", case=case_id, variant=variant, results=results, blocking=dict(LAST)))
            + "\n"
        )
    for same, pcc, diff in results:
        assert same, f"{case_id}/{variant}: output differs from base (max |diff| {diff}) {LAST}"
        assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc}"

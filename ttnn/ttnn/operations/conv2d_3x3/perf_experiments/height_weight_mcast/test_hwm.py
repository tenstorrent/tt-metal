# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the HEIGHT K-blocked walk's weight multicast at a higher rate (perf experiment
height_weight_mcast, round 6).

Runs a private copy of conv2d_3x3 (op_copy/, the op at 5f23b256d1a) whose descriptor takes variant knobs
(HWM_* module globals) and whose writer takes HWM_* preprocessor defines. Variant `base` sets none of them and is the
op unchanged (the helper's one-sender channel).

Precision contract is each case's own (fp32 DEST, LoFi, bf8_b): every variant runs it unchanged. The variants only
move weight bytes differently, so outputs must be bit-identical to base (test_hwm_identical).

One dispatch per test_hwm_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  HWM_CASES=res3_x HWM_SEQ=base,ms4 HWM_TAG=t1 scripts/run_safe_pytest.sh --profile --run-all \\
      <this file>::test_hwm_perf -p no:cacheprovider --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import hwm_load  # noqa: E402

opc, pd, op_mod = hwm_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"

Z = {"HWM_ZONES": "1"}
# Variant -> (descriptor knobs, kernel defines).
VARIANTS = {
    "base": ({}, {}),
}
for s in (1, 2, 3, 4, 6, 8):
    VARIANTS[f"ms{s}"] = (dict(HWM_SENDERS=s), {})
    VARIANTS[f"ms{s}_z"] = (dict(HWM_SENDERS=s), dict(Z))
    VARIANTS[f"msr{s}"] = (dict(HWM_SENDERS=s, HWM_PER_RECT=True), {})
    VARIANTS[f"msr{s}_z"] = (dict(HWM_SENDERS=s, HWM_PER_RECT=True), dict(Z))
    VARIANTS[f"ms{s}row_z"] = (dict(HWM_SENDERS=s, HWM_PLACEMENT="row"), dict(Z))
    VARIANTS[f"ms{s}e"] = (dict(HWM_SENDERS=s), dict(HWM_EARLY="1"))
    VARIANTS[f"ms{s}e_z"] = (dict(HWM_SENDERS=s), dict(Z, HWM_EARLY="1"))
    VARIANTS[f"msr{s}e"] = (dict(HWM_SENDERS=s, HWM_PER_RECT=True), dict(HWM_EARLY="1"))
    VARIANTS[f"msr{s}e_z"] = (dict(HWM_SENDERS=s, HWM_PER_RECT=True), dict(Z, HWM_EARLY="1"))
    VARIANTS[f"ms{s}la_z"] = (dict(HWM_SENDERS=s), dict(Z, HWM_LOOKAHEAD="1"))
    VARIANTS[f"ms{s}d3_z"] = (dict(HWM_SENDERS=s, HWM_WEIGHT_DEPTH=3), dict(Z))
    VARIANTS[f"ms{s}sp_z"] = (dict(HWM_SENDERS=s, HWM_SPLIT_SENDERS=True), dict(Z))
    C = {"HWM_CHAIN": "1"}
    LA = {"HWM_LOOKAHEAD_AFTER": "1"}
    E = {"HWM_EARLY": "1"}
    for pr in (False, True):
        p = "msr" if pr else "ms"
        knobs = dict(HWM_SENDERS=s, HWM_PER_RECT=pr)
        VARIANTS[f"{p}{s}c"] = (knobs, dict(C))
        VARIANTS[f"{p}{s}c_z"] = (knobs, dict(Z, **C))
        VARIANTS[f"{p}{s}cl"] = (knobs, dict(C, **LA))
        VARIANTS[f"{p}{s}cl_z"] = (knobs, dict(Z, **C, **LA))
        VARIANTS[f"{p}{s}cle"] = (knobs, dict(C, **LA, **E))
        VARIANTS[f"{p}{s}cle_z"] = (knobs, dict(Z, **C, **LA, **E))
        VARIANTS[f"{p}{s}ce"] = (knobs, dict(C, **E))
        VARIANTS[f"{p}{s}ce_z"] = (knobs, dict(Z, **C, **E))
        VARIANTS[f"{p}{s}l_z"] = (knobs, dict(Z, **LA))
        VARIANTS[f"{p}{s}cld3_z"] = (dict(knobs, HWM_WEIGHT_DEPTH=3), dict(Z, **C, **LA))
        VARIANTS[f"{p}{s}cld3"] = (dict(knobs, HWM_WEIGHT_DEPTH=3), dict(C, **LA))
        # Read-ahead without the chain: B = before this round's multicast, A = after it; dN = weight CB depth N,
        # aN = rounds read ahead.
        for when, lk in (("B", "HWM_LOOKAHEAD"), ("A", "HWM_LOOKAHEAD_AFTER")):
            for d in (2, 3, 4):
                for a in (1, 2, 3):
                    if a >= d:
                        continue
                    dd = {lk: "1", "HWM_LA_DEPTH": str(a)}
                    kd = dict(knobs, HWM_WEIGHT_DEPTH=d) if d != 2 else knobs
                    VARIANTS[f"{p}{s}{when}d{d}a{a}"] = (kd, dict(dd))
                    VARIANTS[f"{p}{s}{when}d{d}a{a}_z"] = (kd, dict(Z, **dd))
                    VARIANTS[f"{p}{s}{when}d{d}a{a}e"] = (kd, dict(dd, **E))
                    VARIANTS[f"{p}{s}{when}d{d}a{a}e_z"] = (kd, dict(Z, **dd, **E))

# Weight servers (spare cores outside the work grid): srvN = N per rectangle, srvgN = N for the whole channel.
for s in (1, 2, 3):
    for g, pr in (("", True), ("g", False)):
        knobs = dict(HWM_SENDERS=s, HWM_PER_RECT=pr, HWM_SERVER=True)
        VARIANTS[f"srv{g}{s}"] = (knobs, {})
        VARIANTS[f"srv{g}{s}_z"] = (knobs, dict(Z))
        VARIANTS[f"srv{g}{s}e"] = (knobs, {"HWM_EARLY": "1"})
        VARIANTS[f"srv{g}{s}e_z"] = (knobs, dict(Z, HWM_EARLY="1"))
        for d in (3, 4):
            VARIANTS[f"srv{g}{s}d{d}"] = (dict(knobs, HWM_WEIGHT_DEPTH=d), {})
            VARIANTS[f"srv{g}{s}d{d}_z"] = (dict(knobs, HWM_WEIGHT_DEPTH=d), dict(Z))
            VARIANTS[f"srv{g}{s}d{d}e"] = (dict(knobs, HWM_WEIGHT_DEPTH=d), {"HWM_EARLY": "1"})
            VARIANTS[f"srv{g}{s}d{d}e_z"] = (dict(knobs, HWM_WEIGHT_DEPTH=d), dict(Z, HWM_EARLY="1"))

# Protocol v2, suffix "n" (one flag per slot, acks at start + slot reuse only, bias completed after round 0).
for name in list(VARIANTS):
    if name.startswith(("srv", "msr1", "msr2", "ms1", "ms2")):
        knobs, defines = VARIANTS[name]
        stem, z = (name[:-2], "_z") if name.endswith("_z") else (name, "")
        if stem.endswith("e"):
            continue
        VARIANTS[f"{stem}n{z}"] = (dict(knobs, HWM_V2=True), dict(defines))

# Server pipelining: kN = N chunks per round; ch = several servers per group chained.
for base_name in ("srv1n", "srv2n", "srv1", "srv1d3n", "srv2d3n", "srvg1n", "srvg1"):
    knobs, defines = VARIANTS[base_name]
    for kn in (2, 3, 4, 6, 8):
        VARIANTS[f"{base_name}k{kn}"] = (dict(knobs, HWM_CHUNKS=kn), dict(defines))
        VARIANTS[f"{base_name}k{kn}_z"] = (dict(knobs, HWM_CHUNKS=kn), dict(defines, **Z))
        VARIANTS[f"{base_name}k{kn}ch"] = (dict(knobs, HWM_CHUNKS=kn, HWM_SRV_CHAIN=True), dict(defines))
        VARIANTS[f"{base_name}k{kn}ch_z"] = (dict(knobs, HWM_CHUNKS=kn, HWM_SRV_CHAIN=True), dict(defines, **Z))
    VARIANTS[f"{base_name}ch"] = (dict(knobs, HWM_SRV_CHAIN=True), dict(defines))
    VARIANTS[f"{base_name}ch_z"] = (dict(knobs, HWM_SRV_CHAIN=True), dict(defines, **Z))

# Fast server reads (one address per DRAM bank, one-packet reads): suffix f.
for name in list(VARIANTS):
    if name.startswith("srv"):
        knobs, defines = VARIANTS[name]
        stem, z = (name[:-2], "_z") if name.endswith("_z") else (name, "")
        VARIANTS[f"{stem}f{z}"] = (knobs, dict(defines, HWM_FASTREAD="1"))

# Weight depth from leftover L1 (suffix Ln: up to n K-blocks).
for name in list(VARIANTS):
    if name.startswith(("srv1nk", "srv1n", "msr1n", "base")) and "d3" not in name:
        knobs, defines = VARIANTS[name]
        stem, z = (name[:-2], "_z") if name.endswith("_z") else (name, "")
        for n in (3, 4):
            VARIANTS[f"{stem}L{n}{z}"] = (dict(knobs, HWM_DEPTH_LEFTOVER=n), dict(defines))
VARIANTS["base_z"] = ({}, {})
VARIANTS["nowt"] = ({}, {"HWM_NOWEIGHTS": "1"})  # ablation bound: wrong output, run it last
# Dual-RISC server reads (suffix D): the server writer reads the odd chunks on NoC1.
for name in list(VARIANTS):
    if name.startswith("srv1nk") or name.startswith("srvg1nk"):
        knobs, defines = VARIANTS[name]
        stem, z = (name[:-2], "_z") if name.endswith("_z") else (name, "")
        VARIANTS[f"{stem}D{z}"] = (knobs, dict(defines, HWM_SRV_DUAL="1"))

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

# Short names of the perf-flagged LOOSE layers (feature_spec "[perf]" rows).
FOCUS_NAMES = {
    "stem.conv2": "stem2",
    "stem.conv3": "stem3",
    "res2.{0,1,2}.conv2": "res2",
    "res3.0.conv2": "res3_0",
    "res3.{1,2,3}.conv2 + instance res3.fuse_conv.1": "res3_x",
    "semantic decoder.res3.fuse_conv.0": "semantic0",
    "instance decoder.res3.fuse_conv.0": "instance0",
    "semantic decoder.res3.fuse_conv.1": "semantic1",
    "semantic decoder.res2.fuse_conv.0": "semantic2",
    "instance decoder.res2.fuse_conv.0": "instance1",
    "semantic res2.fuse_conv.1 + semantic_head.head.{0,1}": "semantic3",
    "instance res2.fuse_conv.1 + center_head.0 + offset_head.0": "instance2",
    "center_head.1 + offset_head.1": "center1",
}

# Interleaved (DRAM) shapes of the perf harness, K-blocked walks with the weight multicast; bf16 weights, bias.
INTERLEAVED = [
    ((32, 64, 128), 128, 1),
    ((64, 128, 128), 128, 1),
    ((8, 16, 2048), 256, 1),
]
DRAM_LAYOUTS = [
    ("rm_bf16", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ("tile_bf8b", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        layer = c["extras"]["layer"]
        if "[perf]" not in layer:
            continue
        base = layer.replace(" @ LoFi [perf]", "")
        if base not in FOCUS_NAMES:
            continue
        (H, W, c_in), c_out, padding, stride, dilation = c["inputs"]
        out[FOCUS_NAMES[base]] = dict(
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
                mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
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
VARIANTS["gbase"] = (dict(WEIGHT_SERVERS=False), {})  # graduation tree (HWM_COPY=grad), servers off = the op
VARIANTS["grad"] = (dict(WEIGHT_SERVERS=True), {})  # graduation tree, servers on
KNOBS = (
    "WEIGHT_SERVERS",
    "HWM_SENDERS",
    "HWM_PER_RECT",
    "HWM_PLACEMENT",
    "HWM_SPLIT_SENDERS",
    "HWM_WEIGHT_DEPTH",
    "HWM_SERVER",
    "HWM_V2",
    "HWM_CHUNKS",
    "HWM_SRV_CHAIN",
    "HWM_DEPTH_LEFTOVER",
)
KNOBS = tuple(k for k in KNOBS if hasattr(pd, k))
KNOB_DEFAULTS = {k: getattr(pd, k) for k in KNOBS}


def apply_variant(variant, monkeypatch):
    knobs, defines = VARIANTS[variant]
    for k in KNOBS:
        monkeypatch.setattr(pd, k, knobs.get(k, KNOB_DEFAULTS[k]))
    if hasattr(pd, "HWM_DEFINES"):
        monkeypatch.setattr(pd, "HWM_DEFINES", dict(defines))
    pd.select_blocking.cache_clear()
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nb=blocking.max_m_blocks_per_core,
            nkb=blocking.num_k_blocks,
            wd=blocking.weight_depth,
            ub=blocking.untilize_batch_rows,
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
    out_mem = tt_output.memory_config()
    LAST["out_spec"] = str(out_mem.shard_spec) if out_mem.is_sharded() else "interleaved"
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


PERF_CASES = _selected("HWM_CASES", ["res3_x"])
PERF_VARIANTS = _selected("HWM_VARIANTS", ["base"])
# HWM_SEQ: an explicit variant order per case (repeats allowed, e.g. base,cand,cand,base); else HWM_VARIANTS.
PERF_SEQ = _selected("HWM_SEQ", PERF_VARIANTS)


@pytest.mark.parametrize("seq", list(enumerate(PERF_SEQ)), ids=[f"{i}-{v}" for i, v in enumerate(PERF_SEQ)])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hwm_perf(case_id, seq, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch."""
    variant = seq[1]
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    out, ref = run_case(case, device)
    pcc = _pcc(out, ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(kind="perf", case=case_id, variant=variant, pcc=pcc, tag=os.environ.get("HWM_TAG", ""), **LAST)
            )
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


BASE_VARIANT = os.environ.get("HWM_BASE", "base")


@pytest.mark.parametrize("variant", [v for v in PERF_VARIANTS if v != BASE_VARIANT])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hwm_identical(case_id, variant, device, monkeypatch):
    """Same inputs through the base (HWM_BASE) and the candidate: the outputs must be bit-identical (and PCC-gated).
    The candidate runs twice (program-cache miss, then hit)."""
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

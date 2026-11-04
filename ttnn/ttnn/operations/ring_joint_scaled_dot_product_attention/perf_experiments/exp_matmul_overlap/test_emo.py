# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""exp_matmul_overlap — streamed score phases with the softmax exp on the PACK thread (cand*) vs the op's compute
kernel (base). The bench is a full op copy (this dir) on the 1x4 ring; only the compute kernel differs per variant.

  Correctness (golden check per variant + bitwise compare of every variant vs base on the same inputs):
    EMO_RUN=base,cand,cand_mathexp scripts/run_safe_pytest.sh --run-all <this file> -k "correct" -s
  Perf (focus shape; per-call op span per chip + per-key-block zones, analysed after the mesh closes):
    TT_METAL_DEVICE_PROFILER=1 EMO_BLOCK_ZONES=1 EMO_RUN=base,cand scripts/run_safe_pytest.sh <this file> -k perf -s
    python <dir>/analyze.py <dir>/out_<tag>.json
Variants: see EMO_VARIANTS in the copied program descriptor.
"""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # lazy (ttnn tree forbids a global torch import)
import ttnn

import eval.golden_tests.ring_joint_scaled_dot_product_attention.helpers as H
from ttnn.operations.ring_joint_scaled_dot_product_attention.perf_experiments.exp_matmul_overlap import (
    ring_joint_scaled_dot_product_attention as emo_op,
)

RUN = os.environ.get("EMO_RUN", "base,cand").split(",")
HERE = Path(__file__).parent
RING14 = ((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)


def _axes(**kw):
    a = dict(
        dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        fp32_dest_acc_en=False,
        layout=ttnn.TILE_LAYOUT,
        joint_mode="none",
        mask_mode="none",
        is_balanced=False,
        seq_padding="exact",
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
    )
    a.update(kw)
    return a


def _qkv(b, h, n, d, jl=None):
    t = ((b, h, n, d),) * 3
    return t + (((b, h, jl, d),) if jl else ())


FOCUS_EXTRAS = {"num_links": 2, "math_fidelity": ttnn.MathFidelity.HiFi2, "logical_n": 32760, "sample_rows": 256}
FOCUS = (_qkv(1, 40, 8192, 128), _axes(seq_padding="padded"), FOCUS_EXTRAS)

# (id, inputs, axes, extras)
CASES = [
    ("focus_wan_1x40x8192x128", *FOCUS),
    ("min_1x1x64x64", _qkv(1, 1, 64, 64), _axes(), {}),
    ("small_1x2x128x64_padded", _qkv(1, 2, 128, 64), _axes(seq_padding="padded"), {}),
    ("wan_1x10x512x128", _qkv(1, 10, 512, 128), _axes(), {}),
    ("wan_1x10x512x128_padded", _qkv(1, 10, 512, 128), _axes(seq_padding="padded"), {}),
    ("wan_1x10x512x128_linear", _qkv(1, 10, 512, 128), _axes(topology=ttnn.Topology.Linear), {}),
    ("fp32_1x8x1024x128", _qkv(1, 8, 1024, 128), _axes(fp32_dest_acc_en=True), {}),
    ("bf8kv_1x10x512x128", _qkv(1, 10, 512, 128), _axes(kv_dtype=ttnn.bfloat8_b), {}),
    ("bf8q_bf8kv_1x10x512x128", _qkv(1, 10, 512, 128), _axes(dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat8_b), {}),
    ("joint_1x6x1024x128_L128", _qkv(1, 6, 1024, 128, 128), _axes(joint_mode="replicated"), {}),
    (
        "flux_joint_1x6x512x128_L512_padded",
        _qkv(1, 6, 512, 128, 512),
        _axes(joint_mode="replicated", seq_padding="padded"),
        {},
    ),
    ("U_just_above_C_1x27x256x64", _qkv(1, 27, 256, 64), _axes(), {}),
    ("ragged_q_1x24x288x64_padded", _qkv(1, 24, 288, 64), _axes(seq_padding="padded"), {}),
    ("dh96_1x8x512x96", _qkv(1, 8, 512, 96), _axes(), {}),
    ("dh256_1x5x384x256", _qkv(1, 5, 384, 256), _axes(), {}),
    ("dh256_1x5x384x256_padded", _qkv(1, 5, 384, 256), _axes(seq_padding="padded"), {}),
    ("large_logits_1x4x1024x128_padded", _qkv(1, 4, 1024, 128), _axes(seq_padding="padded"), {"qk_scale": 6.0}),
]
# perf-only regimes (larger shapes around the focus; correctness of the same code paths is covered by CASES)
PERF_EXTRA = [
    (
        "bf8kv_wan_1x40x8192x128",
        _qkv(1, 40, 8192, 128),
        _axes(kv_dtype=ttnn.bfloat8_b, seq_padding="padded"),
        dict(FOCUS_EXTRAS),
    ),
    ("dh64_1x24x4096x64_padded", _qkv(1, 24, 4096, 64), _axes(seq_padding="padded"), {"num_links": 2}),
    ("dh256_1x10x2048x256", _qkv(1, 10, 2048, 256), _axes(), {"num_links": 2}),
    ("joint_1x24x4096x128_L512", _qkv(1, 24, 4096, 128, 512), _axes(joint_mode="replicated"), {"num_links": 2}),
    ("fp32_1x10x2048x128", _qkv(1, 10, 2048, 128), _axes(fp32_dest_acc_en=True), {"num_links": 2}),
]
CASES_ALL = list(CASES) + PERF_EXTRA
SEL = os.environ.get("EMO_CASES")
if SEL:
    CASES = [c for c in CASES if c[0] in SEL.split(",")]


def _set_variant(v):
    os.environ["EMO_VARIANT"] = v


def _bits(t):
    return t.to(torch.bfloat16).view(torch.int16)


@pytest.mark.parametrize("mesh_shape,fabric_config", [RING14])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_correct(case, mesh_shape, fabric_config, mesh_device):
    """Every variant: golden check (eval helpers, the op's tolerance) + bitwise compare of the raw output vs base."""
    name, inputs, axes, extras = case
    H.ring_joint_scaled_dot_product_attention = emo_op
    extras = dict(extras)
    qk_scale = extras.pop("qk_scale", None)
    results, failures = {}, []
    # bitwise A/B on identical inputs (large-logit case scales Q so logits reach std ~ qk_scale * sqrt(dh))
    ms = tuple(mesh_device.shape)
    q, k, v, jq, jk, jv, logical_l = H.build_inputs(
        inputs,
        ms,
        cluster_axis=axes["cluster_axis"],
        joint_mode=axes["joint_mode"],
        dtype=axes["dtype"],
        kv_dtype=axes["kv_dtype"],
        seed=7,
    )
    if qk_scale:
        q = H.quantize_like_device((q.float() * qk_scale).to(torch.bfloat16), axes["dtype"])
    G = ms[axes["cluster_axis"]]
    logical_n = extras.get("logical_n") or G * inputs[1][-2] - (H.SEQ_PAD if axes["seq_padding"] == "padded" else 0)
    outs = {}
    for var in ["base"] + [r for r in RUN if r != "base"]:
        _set_variant(var)
        out, jout = H.call_op(
            mesh_device,
            q,
            k,
            v,
            jq,
            jk,
            jv,
            dtype=axes["dtype"],
            kv_dtype=axes["kv_dtype"],
            logical_n=logical_n,
            logical_l=logical_l,
            cluster_axis=axes["cluster_axis"],
            topology=axes["topology"],
            num_links=extras.get("num_links"),
            compute_kernel_config=H.compute_config(axes["fp32_dest_acc_en"], extras.get("math_fidelity")),
        )
        comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=ms)
        o = ttnn.to_torch(out, mesh_composer=comp)
        jo = ttnn.to_torch(jout, mesh_composer=comp) if jout is not None else None
        outs[var] = (o, jo)
    base_o, base_j = outs["base"]
    n_q = inputs[0][-2]
    for var, (o, jo) in outs.items():
        if var == "base":
            continue
        # only rows that are valid on every device matter; compare all (padded rows are deterministic too)
        diff = _bits(o) != _bits(base_o)
        nd = int(diff.sum())
        maxabs = float((o.float() - base_o.float()).abs().max())
        jd = int((_bits(jo) != _bits(base_j)).sum()) if jo is not None else 0
        results[var] = {
            "bit_mismatch": nd,
            "total": int(diff.numel()),
            "max_abs_diff_vs_base": maxabs,
            "joint_mismatch": jd,
        }
        print(f"EMO_BITCMP {name} {var}: mismatching elems {nd}/{diff.numel()} maxabs {maxabs:.4g} joint {jd}")
    # golden check per variant (fresh inputs from the harness)
    if not qk_scale:
        for var in RUN:
            _set_variant(var)
            try:
                H.run_ring_joint_scaled_dot_product_attention(inputs, mesh_device=mesh_device, extras=extras, **axes)
                print(f"EMO_GOLDEN {name} {var}: PASS")
                results.setdefault(var, {})["golden"] = "pass"
            except H.CheckOutputError as e:
                print(f"EMO_GOLDEN {name} {var}: FAIL {e}")
                results.setdefault(var, {})["golden"] = str(e)
                failures.append(f"{var}: {e}")
    else:  # large logits: precision vs the fp32 torch reference
        exp_o, _ = H.pytorch_ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            jq,
            jk,
            jv,
            cluster_axis=axes["cluster_axis"],
            logical_n=logical_n,
            logical_l=logical_l,
            is_causal=False,
            is_balanced=False,
            scale=None,
        )
        b, h = inputs[0][0], inputs[0][1]
        for var, (o, _) in outs.items():
            a_all, e_all = [], []
            for r in range(ms[0]):
                for c in range(ms[1]):
                    p_ = c if axes["cluster_axis"] == 1 else r
                    m = H.valid_query_rows(
                        p_, n_q, inputs[1][-2], G, logical_n=logical_n, is_balanced=False, cross=False
                    )
                    a_all.append(o[r * b : (r + 1) * b, c * h : (c + 1) * h][:, :, m].float().reshape(-1))
                    e_all.append(exp_o[r, c][:, :, m].float().reshape(-1))
            pcc, rms = H._pcc_rel_rms(torch.cat(a_all), torch.cat(e_all))
            print(f"EMO_LARGE {name} {var}: pcc {pcc:.6f} rel_rms {rms:.5f}")
            results.setdefault(var, {})["large_logits"] = {"pcc": pcc, "rel_rms": rms}
    with open(HERE / "correct_results.jsonl", "a") as f:
        f.write(json.dumps({"case": name, "results": results}) + "\n")
    assert not failures, failures


@pytest.mark.parametrize("mesh_shape,fabric_config", [RING14])
def test_perf(mesh_shape, fabric_config, mesh_device):
    """Per case (EMO_PERF_CASES, default the focus), per variant: one compile call then EMO_REPS measured calls. The
    call order is written to perf_order.json so analyze.py can map profiler runs to (case, variant)."""
    reps = int(os.environ.get("EMO_REPS", "1"))
    ms = tuple(mesh_device.shape)
    sel = os.environ.get("EMO_PERF_CASES", "focus_wan_1x40x8192x128").split(",")
    all_cases = {c[0]: c for c in CASES_ALL}
    order = []
    for cname in sel:
        _, inputs, axes, extras = all_cases[cname]
        q, k, v, jq, jk, jv, logical_l = H.build_inputs(
            inputs,
            ms,
            cluster_axis=axes["cluster_axis"],
            joint_mode=axes["joint_mode"],
            dtype=axes["dtype"],
            kv_dtype=axes["kv_dtype"],
            seed=3,
        )
        to_dev = (
            lambda t, dt: None
            if t is None
            else H.create_ttnn_input_tensor(t, mesh_device, dtype=dt, layout=ttnn.TILE_LAYOUT)  # noqa: E731
        )
        dq, dk, dv = to_dev(q, axes["dtype"]), to_dev(k, axes["kv_dtype"]), to_dev(v, axes["kv_dtype"])
        djq, djk, djv = to_dev(jq, axes["dtype"]), to_dev(jk, axes["kv_dtype"]), to_dev(jv, axes["kv_dtype"])
        G = ms[axes["cluster_axis"]]
        logical_n = extras.get("logical_n") or G * inputs[1][-2] - (H.SEQ_PAD if axes["seq_padding"] == "padded" else 0)
        cfg = H.compute_config(axes["fp32_dest_acc_en"], extras.get("math_fidelity"))
        rounds = int(os.environ.get("EMO_ROUNDS", "1"))  # variants alternate round by round (A B A B ...)
        for rnd in range(rounds):
            for var in RUN:
                _set_variant(var)
                for i in range((1 if rnd == 0 else 0) + reps):
                    emo_op(
                        dq,
                        dk,
                        dv,
                        djq,
                        djk,
                        djv,
                        logical_n=logical_n,
                        cluster_axis=axes["cluster_axis"],
                        topology=axes["topology"],
                        num_links=extras.get("num_links"),
                        compute_kernel_config=cfg,
                    )
                    ttnn.synchronize_device(mesh_device)
                    ttnn.ReadDeviceProfiler(mesh_device)
                    order.append(f"{cname}|{var}:{'compile' if (rnd == 0 and i == 0) else 'meas'}")
    with open(HERE / "perf_order.json", "w") as f:
        json.dump(order, f)

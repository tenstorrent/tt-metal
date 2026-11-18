# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: fold the bias into the HEIGHT / interleaved K-blocked walk (perf experiment height_bias_fold).

Runs a private copy of conv2d_3x3 (op_copy/, the op at 5f23b256d1a). Variants (hbf_load.set_variant):
  base      the op unchanged (K-blocked walk packs fp32 partials, then the separate add_bias_bcast_rows pass)
  fold      bias seeded into DEST before K-block 0's matmul, last K-block reloads the partials (SrcA) and packs the
            output with ReLU; no bias pass
  fold_utd  fold, cb_matmul_partials tagged UnpackToDestFp32 (the reload is an exact fp32 copy)
Every variant runs the case's own precision contract (fp32 DEST, its fidelity, its dtypes).

One dispatch per test_hbf_perf, for `run_safe_pytest.sh --profile`:
  HBF_CASES=semantic3 HBF_SEQ=base,fold,fold,base HBF_TAG=t1 scripts/run_safe_pytest.sh --profile --run-all \\
      <this file>::test_hbf_perf -p no:cacheprovider --import-mode=prepend
test_hbf_compare runs base and the candidates on the same inputs and records error statistics vs the fp32 torch
reference (the fold is not bit-identical: the bias is added before the partials round). No-bias cases must be
bit-identical (the fold only exists with a bias).
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import hbf_load  # noqa: E402

opc, pd, op_mod = hbf_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"
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


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    perf = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    names = [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in perf]
    seen = {}
    for c, name in zip(perf, names):
        # pytest's duplicate-id numbering (test_conv2d_3x3_perf_focus ids)
        if names.count(name) > 1:
            cid = f"{name}{seen.get(name, 0)}"
            seen[name] = seen.get(name, 0) + 1
        else:
            cid = name
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
    # No-bias twins of focus shapes (the fold only exists with a bias: these must be bit-identical to base).
    out["semantic3_nobias"] = dict(out["semantic3"], bias=False)
    out["stem_conv1_nobias"] = dict(out["stem_conv1"], bias=False)
    # test_conv2d_3x3_perf PERF_SHAPES (DRAM interleaved, whole grid; bf16 weights, bias, no ReLU, HiFi4 default).
    for (H, W, c_in), c_out, padding in [((64, 128, 128), 128, 1), ((128, 256, 64), 64, 1), ((8, 16, 2048), 256, 1)]:
        for tag, dtype, layout in [
            ("rm_bf16", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            ("tile_bf8b", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        ]:
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
    out["dram_H64_W128_Ci128_Co128_tile_bf8b_nobias"] = dict(out["dram_H64_W128_Ci128_Co128_tile_bf8b"], bias=False)
    return out


class _Cases(dict):
    """`<id>_nobias` derives the no-bias twin of any case."""

    def __missing__(self, key):
        if key.endswith("_nobias"):
            return dict(self[key[: -len("_nobias")]], bias=False)
        raise KeyError(key)


CASES = _Cases(_cases())
LAST = {}


def _selected(env, default):
    v = os.environ.get(env)
    # ";" separates ids that contain commas (res2_{0,1,2}_conv2).
    return [x for x in v.split(";" if ";" in v else ",") if x] if v else default


def apply_variant(variant, monkeypatch):
    hbf_load.set_variant(pd, variant)
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            bn=blocking.block_n_tiles,
            sbh=blocking.out_subblock_h,
            sbw=blocking.out_subblock_w,
            nb=blocking.max_m_blocks_per_core,
            nkb=blocking.num_k_blocks,
            nslice=bool(blocking.nslice),
            center=int(blocking.center_mode),
            in0_stream_rows=blocking.in0_stream_rows,
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
    out_spec = (tt_output.memory_config().memory_layout, tt_output.memory_config().shard_spec)
    # fp32 reference on the same (device-quantized) operands: the weights / bias / input as the device holds them.
    xq = torch_input.float()
    if case["dtype"] == ttnn.bfloat8_b:
        xq = ttnn.to_torch(ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    wq = torch_weight.float()
    bq = None if torch_bias is None else torch_bias.float()
    if case["weight_dtype"] == ttnn.bfloat8_b:
        # bf8_b shares exponents across 16 consecutive datums of a tile row; the prepared weight's rows run along
        # C_out, so quantize the [K, C_out] matrix as the device holds it.
        wk = torch_weight.float().permute(2, 3, 1, 0).reshape(-1, c_out)
        wk = ttnn.to_torch(ttnn.from_torch(wk, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        wq = wk.reshape(3, 3, c_in, c_out).permute(3, 2, 0, 1)
        if bq is not None:
            bq = (
                ttnn.to_torch(
                    ttnn.from_torch(torch_bias.float().reshape(1, -1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
                )
                .float()
                .reshape(-1)
            )
    x = xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2).double()
    ref = torch.nn.functional.conv2d(
        x,
        wq.double(),
        None if bq is None else bq.double(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if case["relu"]:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).double().reshape(ref.shape), ref, out_spec


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _err(out, ref):
    d = (out - ref).abs()
    scale = float(ref.abs().max())
    return dict(
        pcc=_pcc(out, ref),
        mae=float(d.mean()),
        rmse=float(d.pow(2).mean().sqrt()),
        max_abs=float(d.max()),
        rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
        ref_absmax=scale,
    )


PERF_CASES = _selected("HBF_CASES", ["semantic3"])
PERF_SEQ = _selected("HBF_SEQ", ["base", "fold"])


@pytest.mark.parametrize("seq", list(enumerate(PERF_SEQ)), ids=[f"{i}-{v}" for i, v in enumerate(PERF_SEQ)])
@pytest.mark.parametrize("case_id", PERF_CASES)
def test_hbf_perf(case_id, seq, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against the fp32 reference."""
    variant = seq[1]
    case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    out, ref, _ = run_case(case, device)
    e = _err(out, ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(dict(kind="perf", case=case_id, variant=variant, tag=os.environ.get("HBF_TAG", ""), **e, **LAST))
            + "\n"
        )
    assert e["pcc"] >= case["pcc"], f"{case_id}/{variant}: pcc {e['pcc']} < {case['pcc']}"


CMP_CASES = _selected("HBF_CMP_CASES", PERF_CASES)
CMP_VARIANTS = _selected("HBF_CMP_VARIANTS", ["fold", "fold_utd"])


@pytest.mark.parametrize("case_id", CMP_CASES)
def test_hbf_compare(case_id, device, monkeypatch):
    """Same inputs through base and every candidate (each twice: program-cache miss, then hit). Records error stats
    vs the fp32 reference; PCC-gates every run; bit-identical to base when the case has no bias."""
    import torch

    case = CASES[case_id]
    rec = dict(kind="compare", case=case_id)
    apply_variant("base", monkeypatch)
    base_out, ref, base_spec = run_case(case, device, seed=1)
    rec["base"] = _err(base_out, ref)
    rec["blocking"] = dict(LAST)
    failures = []
    for v in CMP_VARIANTS:
        runs = []
        for _ in range(2):
            apply_variant(v, monkeypatch)
            out, _, spec = run_case(case, device, seed=1)
            e = _err(out, ref)
            diff = (out - base_out).abs()
            e["differ_frac"] = float((diff > 0).double().mean())
            e["max_diff_vs_base"] = float(diff.max())
            e["same_spec"] = str(spec) == str(base_spec)
            runs.append(e)
            if e["pcc"] < case["pcc"]:
                failures.append(f"{v}: pcc {e['pcc']}")
            if not e["same_spec"]:
                failures.append(f"{v}: output spec {spec} != {base_spec}")
            if not case["bias"] and e["differ_frac"] != 0.0:
                failures.append(f"{v}: no-bias output differs from base ({e['differ_frac']})")
        if not torch.equal(torch.tensor(runs[0]["mae"]), torch.tensor(runs[1]["mae"])):
            failures.append(f"{v}: cache-miss and cache-hit runs differ")
        rec[v] = runs
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert not failures, f"{case_id}: {failures}"

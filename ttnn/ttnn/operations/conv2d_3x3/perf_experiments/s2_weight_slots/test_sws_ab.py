# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B in one process for the s2_weight_slots idea. Each (case, variant) runs one dispatch of a perf-file test body,
the op swapped per call between this experiment's copies (sws_copy.py). A variant is `<pkg>[:NAME=expr+NAME=expr]`:
descriptor knobs monkeypatched on that copy's program-descriptor module for that call only.

Cases: `<focus id>` (tests/.../test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus, e.g. res3_0_conv2) or
`hs:<layer>` (test_conv2d_3x3_height_sharded_perf, HiFi4 default config) or `bs:<layer>` (BLOCK perf shapes).

Env: SWS_CASES (comma list), SWS_VARIANTS (comma list, run in that order), SWS_ORDER=alt (reverse the variant order on
every other repetition), SWS_REPS (default 1), SWS_UNZ=1 (zone-stripped twins), SWS_INFO=1 (print blocking + CB bytes).
Every call's output is compared element-for-element with the first variant of its case (exact.jsonl). PCC vs torch is
asserted by the test body. DEVICE KERNEL DURATION rows come back in call order (run.sh joins them with the calls log).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sws_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as perf  # noqa: E402

HERE = Path(__file__).resolve().parent
FOCUS = dict(zip(perf._FOCUS_IDS, perf._FOCUS_CASES))
HS = {c[0]: c for c in perf.HEIGHT_SHAPES}
BS = {c[0]: c for c in perf.BLOCK_SHAPES}
SEL = [c for c in os.environ.get("SWS_CASES", "res3_0_conv2").split(",") if c]
VARS = [v for v in os.environ.get("SWS_VARIANTS", "base,copy").split(",") if v]
REPS = int(os.environ.get("SWS_REPS", "1"))
PARAMS = []
for r in range(REPS):
    vs = VARS if (os.environ.get("SWS_ORDER") != "alt" or r % 2 == 0) else list(reversed(VARS))
    for c in SEL:
        PARAMS += [(c, v, r) for v in vs]
CALLS = HERE / "runs" / os.environ.get("SWS_CALLS", "calls.txt")
_OUT = {}


def _parse(variant):
    if os.environ.get("SWS_UNZ") == "1":
        pkg, sep, rest = variant.partition(":")
        variant = (pkg if pkg.endswith("_unz") else pkg + "_unz") + sep + rest
    pkg, _, knobs = variant.partition(":")
    kv = {}
    for item in [k for k in knobs.split("+") if k]:
        name, _, val = item.partition("=")
        kv[name] = eval(val)
    return pkg, kv


def _run_case(case_id, device, op_mod, pd_mod, capture, monkeypatch):
    """Cases: focus id | hs:<layer> | bs:<layer> | s2:<stride-2 unit case id> | hu:<HEIGHT unit SHAPES index>:<dtype>."""
    import types

    import torch
    import ttnn

    if case_id.startswith("hs:"):
        layer, case, dtype, layout, shard_shape, num_cores = HS[case_id[3:]]
        perf.test_conv2d_3x3_height_sharded_perf(layer, case, dtype, layout, shard_shape, num_cores, device)
    elif case_id.startswith("bs:"):
        perf.test_conv2d_3x3_block_sharded_perf(*BS[case_id[3:]], device)
    elif case_id.startswith("s2:"):
        import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_stride2_taps as s2

        _, shape, layout, shard, grid, dtype, m_block_cap = {c[0]: c for c in s2.CASES}[case_id[3:]]
        (H, W, c_in), c_out = shape
        monkeypatch.setattr(
            s2,
            "_op_module",
            types.SimpleNamespace(
                prepare_conv2d_weights=op_mod.prepare_conv2d_weights,
                prepare_conv2d_bias=op_mod.prepare_conv2d_bias,
                conv2d_3x3=capture,
            ),
        )
        if m_block_cap is not None:
            real_footprint = pd_mod._footprint

            def capped(g, core_m_tiles, bm, bk, *args, **kwargs):
                fit = real_footprint(g, core_m_tiles, bm, bk, *args, **kwargs)
                if bm > m_block_cap or bk < g.tensor_kkh_tiles:
                    fit = dict(fit, footprint_bytes=1 << 40)
                return fit

            monkeypatch.setattr(pd_mod, "_footprint", capped)
        torch.manual_seed(0)
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16)
        s2._run(
            shape, dtype, s2._memory_config(device, layout, shard, grid), device, torch_input, torch_weight, torch_bias
        )
    elif case_id.startswith("hu:"):
        import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_height_sharded as hu

        idx, dt = case_id[3:].split(":")
        (H, W, c_in), c_out, p, st, d, rows = hu.SHAPES[int(idx)]
        dtype, layout = {
            "rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            "bf16": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "bf8b": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        }[dt]
        for name in ("prepare_conv2d_weights", "prepare_conv2d_bias"):
            monkeypatch.setattr(hu, name, getattr(op_mod, name))
        monkeypatch.setattr(hu, "conv2d_3x3", capture)
        shard_rows = rows if layout == ttnn.TILE_LAYOUT else hu.RM_SHARD_ROWS[rows]
        torch.manual_seed(0)
        hu._run(
            ((H, W, c_in), c_out, p, st, d),
            dtype,
            layout,
            hu._row_major_config(device, H * W, c_in, layout, shard_rows),
            device,
            with_bias=True,
            relu=True,
        )
    else:
        perf.test_conv2d_3x3_perf_focus(FOCUS[case_id], device)


@pytest.mark.parametrize("case_id,variant,rep", PARAMS, ids=[f"{c}-{v}-{r}" for c, v, r in PARAMS])
def test_sws_ab(case_id, variant, rep, device, monkeypatch):
    import ttnn

    # Variants share kernel sources but differ in args / defines: never let one hit another's cached program.
    device.clear_program_cache()
    pkg, knobs = _parse(variant)
    op_mod, pd_mod = copy_modules(pkg)
    for name, val in knobs.items():
        mod = op_mod if name.startswith("@") else pd_mod
        name = name.lstrip("@")
        assert hasattr(mod, name), name
        monkeypatch.setattr(mod, name, val)
    for name in dir(pd_mod):
        m = getattr(pd_mod, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()
    for name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
        monkeypatch.setattr(perf, name, getattr(op_mod, name))
    if os.environ.get("SWS_INFO") == "1":
        real_cpd = op_mod.create_program_descriptor

        def cpd(*a, **k):
            d = real_cpd(*a, **k)
            b = k["blocking"]
            total = sum(cb.total_size for cb in d.cbs)
            budget = pd_mod.l1_cb_budget_bytes(0)
            print(
                f"\nSWS_INFO {case_id} {variant}: bm={b.block_m_tiles} bk={b.block_k_tiles} bn={b.block_n_tiles} "
                f"kblocks={b.num_k_blocks} mblocks={b.max_m_blocks_per_core} depth={b.weight_depth} "
                f"nslice={b.nslice} core_m={b.core_m_tiles} footprint={b.footprint_bytes} cbs_total={total} "
                f"budget(no shards)={budget} sems={len(d.semaphores)} "
                f"servers={sum('weight_server_writer' in str(k.kernel_source) for k in d.kernels)} "
                f"kwf={any('CONV2D_3X3_KW_FOLD' in str(k.defines) for k in d.kernels)}"
            )
            print("SWS_CBS", sorted((cb.format_descriptors[0].buffer_index, cb.total_size) for cb in d.cbs))
            return d

        monkeypatch.setattr(op_mod, "create_program_descriptor", cpd)
    captured = {}
    real = op_mod.conv2d_3x3

    def capture(*a, **k):
        out = real(*a, **k)
        captured["out"] = ttnn.to_torch(out).float()
        return out

    monkeypatch.setattr(perf, "conv2d_3x3", capture)
    with open(CALLS, "a") as f:
        f.write(f"{case_id} {variant} {rep}\n")
    _run_case(case_id, device, op_mod, pd_mod, capture, monkeypatch)
    if os.environ.get("CONV2D_3X3_PERF_ABLATION") == "1":
        return
    import torch

    out = captured["out"]
    first = _OUT.setdefault(case_id, (variant, out))
    same_shape = first[1].shape == out.shape
    diff = int((first[1] != out).sum().item()) if same_shape else -1
    maxabs = float((first[1] - out).abs().max().item()) if same_shape else -1.0
    with open(HERE / "exact.jsonl", "a") as f:
        f.write(
            json.dumps(
                dict(
                    case=case_id, variant=variant, rep=rep, ref=first[0], differ=diff, maxabs=maxabs, numel=out.numel()
                )
            )
            + "\n"
        )
    assert diff == 0, f"{case_id} {variant}: {diff} elements differ from {first[0]}"

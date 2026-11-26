# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 A/B bench: the pre-R3 op (base_pkg = 8db3a5fa93d, Perf 2 final) vs the current op, same session.

    R3_VARIANTS=base,op     'base' = base_pkg, 'op' = the op in the tree
    R3_CASES=a,b            case names from _round2/bench.py CASES, or a group: rm_sh, tile_sh, dram, r3 (op only)
    R3_CHUNK_ZONES=1        request the per-chunk zones on both
One line per run -> runs.jsonl (variant, case, pcc, rms); parse with ../_round2/parse.py <ops csv> runs.jsonl.
"""
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.metrics import compute_metrics_torch
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, pytorch_groupnorm_sc_N_1_HW_C

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "_round2"))
from bench import CASES  # noqa: E402  (the shared case table)


def _load_base():
    name = "gn_r3_base"
    spec = importlib.util.spec_from_file_location(
        name, HERE / "base_pkg" / "__init__.py", submodule_search_locations=[str(HERE / "base_pkg")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.groupnorm_sc_N_1_HW_C_b")


OPS = {
    "op": importlib.import_module("ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b"),
    "base": _load_base(),
}
RM_SH = [k for k, v in CASES.items() if v[2] is not None and v[3] == ttnn.ROW_MAJOR_LAYOUT and not k.startswith("r3_")]
TILE_SH = [k for k, v in CASES.items() if v[2] is not None and v[3] == ttnn.TILE_LAYOUT and not k.startswith("r3_")]
GROUPS = {
    "rm_sh": RM_SH,
    "tile_sh": TILE_SH,
    "dram": [
        "flux1_dram_16384x256",
        "refiner_dram_16384x1152",
        "vae_dram_65536x512",
        "mid_4096x640_rm",
        "mid_16384x320_tile",
    ],
    "r3": [k for k in CASES if k.startswith("r3_")],
}
VARIANTS = os.environ.get("R3_VARIANTS", "base,op").split(",")
CASE_NAMES = [n for s in os.environ.get("R3_CASES", "rm_sh").split(",") for n in GROUPS.get(s, [s])]
# r3 shapes are ragged: the base op refuses them, so they only run on 'op'
PARAMS = [(v, c) for c in CASE_NAMES for v in VARIANTS if not (v == "base" and c.startswith("r3_") and CASES[c][2])]
RUNS = HERE / os.environ.get("R3_RUNS", "runs.jsonl")


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    RUNS.write_text("")
    yield


@pytest.mark.parametrize("variant, case", PARAMS, ids=[f"{v}-{c}" for v, c in PARAMS])
def test_r3_ab(device, variant, case):
    import torch  # local: no global torch import under ttnn/ (pre-commit)

    opmod = OPS[variant]
    shape, G, grid, layout, eps, in_place = CASES[case]
    saved_tuning = dict(opmod.TUNING_OVERRIDES)
    if os.environ.get("R3_CHUNK_ZONES") == "1":
        opmod.TUNING_OVERRIDES["chunk_zones"] = True
    try:
        torch.manual_seed(0)
        C = shape[-1]
        xt = torch.randn(shape, dtype=torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        bt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if grid is None
            else block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        )
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = opmod.groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
        ttnn.synchronize_device(device)
        y = ttnn.to_torch(out).reshape(shape)
        ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
        m = compute_metrics_torch(y, ref, readback_dtype=torch.bfloat16)
        rec = dict(variant=variant, case=case, pcc=m.pcc, rms=m.rms)
        with RUNS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if in_place:
            ttnn.deallocate(out, force=True)
        else:
            out.deallocate()
            x.deallocate()
        assert rec["pcc"] >= 0.999 and rec["rms"] <= 0.01, rec
    finally:
        opmod.TUNING_OVERRIDES.clear()
        opmod.TUNING_OVERRIDES.update(saved_tuning)

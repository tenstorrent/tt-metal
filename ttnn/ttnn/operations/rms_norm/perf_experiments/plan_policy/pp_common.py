# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED PERF EXPERIMENT — idea `plan_policy`.

The op's plan policy (`_choose_group_size` + `_split_cost` + the W-chunk search in
`_solve`) is an ANALYTIC model with two hand-fitted coefficients and no
DRAM-bandwidth term.  This experiment measures, per guard-set cell, what the plan
knobs are actually worth, so the policy can be re-fit from measurement instead of
from the model.

Nothing here edits the op: every arm is reached through the op's EXISTING
`_levers=` hook (`w_group`, `w_split`, `wt_block`, `coarse_chunk`, ...).  The
candidate policy is a pure-python function in `pp_policy.py` that re-scores the
same candidate set; graduating it is a change to `_split_cost` only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config, rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 2
N_ITERS = 8

ART = Path("generated/rms_norm_plan_policy")
MANIFEST_PATH = ART / "manifest.json"
PLANS_PATH = ART / "plans.json"


def cfg_loose():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


CONFIGS = {"default": default_compute_kernel_config, "loose": cfg_loose}

# (shape, dtype, layout, gamma_layout)  — mirrors _bench_rms_norm.BENCH_SHAPES /
# BENCH_GAMMA_LAYOUT so the numbers are comparable with the op's own bench.
SHAPES = {
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_2304": ((1, 1, 32, 2304), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_5120": ((1, 1, 32, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "grid_starved": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "h_nonalign": ((1, 1, 100, 736), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- domain probes (outside the guard set): narrow-W / tall-H shapes, where
    # the row-block cap `ceil(Rt / cores)` is the binding constraint ------------
    "d_4096x1024": ((1, 1, 4096, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "d_8192x512": ((1, 1, 8192, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "d_16384x1024": ((1, 1, 16384, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "d_8192x2048": ((1, 1, 8192, 2048), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
}

# The guard set, one representative per distinct kernel path x layout x placement,
# each at the config it is used at.  Keys are the CELL ids used everywhere below.
CELLS = {
    "focus": ("focus", "loose", None, True),
    "decode_1024": ("decode_1024", "loose", None, True),
    "decode_2304": ("decode_2304", "loose", None, True),
    "decode_5120": ("decode_5120", "loose", None, True),
    "prefill_1024": ("prefill_1024", "loose", None, True),
    "prefill_7168": ("prefill_7168", "loose", None, True),
    "grid_starved": ("grid_starved", "loose", None, True),
    "row_major": ("row_major", "loose", None, True),
    "smallest": ("smallest", "loose", None, True),
    "w_nonalign": ("w_nonalign", "loose", None, True),
    "h_nonalign": ("h_nonalign", "loose", None, True),
    "focus_nogamma": ("focus", "loose", None, False),
    "prefill_1024_bf8b": ("prefill_1024", "loose", ttnn.bfloat8_b, True),
    "prefill_1024_fp32": ("prefill_1024", "default", ttnn.float32, True),
    # domain probes
    "d_4096x1024_bf8b": ("d_4096x1024", "loose", ttnn.bfloat8_b, True),
    "d_8192x512_bf8b": ("d_8192x512", "loose", ttnn.bfloat8_b, True),
    "d_16384x1024_bf8b": ("d_16384x1024", "loose", ttnn.bfloat8_b, True),
    "d_8192x512": ("d_8192x512", "loose", None, True),
    "d_8192x2048": ("d_8192x2048", "loose", None, True),
    "d_4096x1024": ("d_4096x1024", "loose", None, True),
}


def make(device, cell):
    import torch

    name, config, dtype_ovr, has_gamma = CELLS[cell]
    shape, dtype, layout, glayout = SHAPES[name]
    dtype = dtype_ovr or dtype
    if dtype == ttnn.bfloat8_b:
        glayout = ttnn.TILE_LAYOUT
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=dtype, layout=layout, device=device)
    gt = None
    g = None
    if has_gamma:
        gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
        g = ttnn.from_torch(gt, dtype=dtype, layout=glayout, device=device)
    return x, g, xt, gt, CONFIGS[config]()


def torch_ref(xt, gt, eps=1e-6):
    import torch

    x = xt.reshape(-1, xt.shape[-1]).to(torch.float32)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if gt is not None:
        out = out * gt.reshape(1, -1).to(torch.float32)
    return out.reshape(xt.shape)


def pcc(a, b):
    import torch

    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# --- plan introspection -------------------------------------------------------
def candidate_table(device, cell, levers=None):
    """Every G the policy CONSIDERS for this cell, with its model score + solve.

    Spies on `_split_cost` so the table is the policy's own candidate set, not a
    reconstruction of it.
    """
    x, g, _, _, cfg = make(device, cell)
    rows = []
    orig = pd._split_cost

    def spy(solved, **kw):
        out = orig(solved, **kw)
        rows.append(
            dict(
                g=kw["group_size"],
                cost=out[0],
                groups_used=out[1],
                num_groups=kw["num_groups"],
                Wt_core=kw["Wt_core"],
                regime=solved.regime,
                block_ht=solved.BLOCK_HT,
                wr=solved.WT_REDUCE_BLOCK,
                in_depth=solved.IN_BUF_DEPTH,
                out_depth=solved.OUT_BUF_DEPTH,
                gamma_depth=solved.GAMMA_DEPTH,
                num_row_blocks=solved.num_row_blocks,
            )
        )
        return out

    pd._split_cost = spy
    try:
        plan = pd.blocking_plan(x, g, x, device, cfg, levers)
    finally:
        pd._split_cost = orig
    pick = dict(
        G=plan.group_size,
        regime=plan.regime,
        Wt=plan.Wt,
        Rt=plan.Rt,
        Wt_core=plan.Wt_core,
        block_ht=plan.BLOCK_HT,
        wr=plan.WT_REDUCE_BLOCK,
        ws=plan.WT_SCALE_BLOCK,
        in_depth=plan.IN_BUF_DEPTH,
        out_depth=plan.OUT_BUF_DEPTH,
        gamma_depth=plan.GAMMA_DEPTH,
        num_groups=plan.num_groups,
        groups_used=plan.groups_used,
        num_row_blocks=plan.num_row_blocks,
        l1=plan.working_set_bytes(),
        budget=plan.l1_cb_budget,
    )
    return rows, plan, pick


# --- dispatch / measurement ---------------------------------------------------
def run_arm(device, manifest, label, cell, levers=None, iters=N_ITERS, check=True, tensors=None):
    """One (cell, lever-setting) arm: PCC-gate on the warm-up, then time `iters`.

    The PCC gate is paid on a WARM-UP dispatch, which `report_from_csv` skips, so
    correctness costs nothing in the measured window.
    """
    x, g, xt, gt, cfg = tensors if tensors is not None else make(device, cell)
    out = rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=levers)
    p = None
    if check:
        p = pcc(ttnn.to_torch(out).to(__import__("torch").float32), torch_ref(xt, gt))
    for _ in range(N_WARMUP - 1):
        rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=levers)
    ttnn.synchronize_device(device)
    for _ in range(iters):
        rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=levers)
    ttnn.synchronize_device(device)
    manifest.append(dict(label=label, cell=cell, levers=levers or {}, calls=N_WARMUP + iters, profiled=iters, pcc=p))
    return p


def write_manifest(manifest, path=MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def report_from_csv(csv_path, manifest_path=MANIFEST_PATH):
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = {}, 0
    for arm in manifest:
        i += arm["calls"] - arm["profiled"]
        window = rows[i : i + arm["profiled"]]
        i += arm["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in window if r.get(_DURATION_KEY))
        out[arm["label"]] = dict(
            ns=(vals[len(vals) // 2] if vals else None), cell=arm["cell"], levers=arm["levers"], pcc=arm.get("pcc")
        )
    return out

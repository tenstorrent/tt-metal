# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pipeline_overlap graduation: OLD-vs-NEW allocation policy, in ONE tree.

WHY THIS EXISTS AND NOT A PLAIN BEFORE/AFTER OF THE GATE BENCH: this working tree
is shared with a sibling graduation that edits `kernels/rms_norm_reader.cpp`
concurrently, so an absolute number measured now is NOT comparable to a number
measured on the pre-graduation HEAD.  Both arms here run in the SAME process, on
the SAME kernels, and differ in exactly one thing: which `_solve` the host plan
came from.  The delta is therefore attributable to the allocation policy alone.

  arm OLD  `_solve` restored to the pre-graduation ordering (coarsest W-chunk that
           fits at DEPTH 1, then try to double-buffer, cb_gamma_tiles always 1
           generation deep).
  arm NEW  the shipped `_solve` (coarsest W-chunk at which cb_input_tiles AND
           cb_gamma_tiles both reach depth 2).

Correctness is the only pass/fail; both arms are PCC-gated against torch on every
shape, including the two CB-WRAP hazards (a Wt_core that is not a power of two,
and a shape with a partial last W-tile).

NOTE ON WHERE THE DRIVER LIVES: a pytest file physically inside `ttnn/ttnn/...`
no longer COLLECTS in this tree (pytest's basedir insertion re-imports the ttnn
package under a second name -> "Operation with name bernoulli is already
registered"), so the driver is a 20-line shim in the tests tree and all of the
logic is here.

    scripts/run_safe_pytest.sh --profile --run-all \\
        tests/ttnn/unit_tests/operations/rms_norm/test_po_graduation_ab.py -s
    python3 -c "from ttnn.operations.rms_norm import _bench_rms_norm as b; \\
        [print(k, v[0]) for k, v in b.report_from_csv('<csv>', '<manifest>').items()]"
"""

from __future__ import annotations

import json
from pathlib import Path


import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd
from ttnn.operations.rms_norm._bench_rms_norm import _cfg_default, _cfg_loose

MANIFEST = Path(__file__).with_name("ab_manifest.json")
N_WARMUP = 1
N_ITERS = 3
PCC_GATE = 0.9995


# ---------------------------------------------------------------------------
# arm OLD: the pre-graduation solver, verbatim (HEAD 03c5b76 rms_norm_program_
# descriptor.py::_solve), so the counterfactual is the real thing and not a
# strawman.  The ONLY differences from the shipped one are the chunk search's
# target depth profile and gamma_depth.
# ---------------------------------------------------------------------------
def _solve_old(
    *,
    Wt_core,
    w_split_group,
    row_parallel_units,
    Rt,
    maskless_w,
    dest_limit,
    l1_cb_budget,
    gamma_cap_tiles,
    layout_common,
    levers,
):
    common = dict(layout_common, Wt_core=Wt_core, w_split_group=w_split_group)

    def ws_bytes(regime, block_ht, in_depth, out_depth, rm_depth, wr, wsc):
        return opd._working_set_bytes(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_depth,
            out_depth=out_depth,
            rm_depth=rm_depth,
            gamma_depth=1,  # the old hard-coded `ws` page count
            wr=wr,
            ws=wsc,
            gamma_ingest_block=opd._largest_divisor_at_most(wsc, gamma_cap_tiles),
            **common,
        )

    fits = ws_bytes("A", 1, 1, 1, 1, Wt_core, Wt_core) <= l1_cb_budget
    regime = "A" if (maskless_w and fits) else "B"

    max_block_ht = max(1, opd._div_up(Rt, max(1, row_parallel_units)))
    max_block_ht = min(max_block_ht, dest_limit)

    block_ht = 1
    in_depth = out_depth = rm_depth = 1

    if regime == "A":
        wr = wsc = Wt_core
    else:
        wr = wsc = 1
        if opd._lever(levers, "coarse_chunk"):
            forced_wt = opd._lever(levers, "wt_block")
            chunk_cap = min(Wt_core, forced_wt) if forced_wt else Wt_core
            for cand in range(chunk_cap, 0, -1):
                if Wt_core % cand != 0:
                    continue
                if ws_bytes("B", 1, 1, 1, 1, cand, cand) <= l1_cb_budget:
                    wr = wsc = cand
                    break

    if opd._lever(levers, "double_buffer"):
        if ws_bytes(regime, block_ht, 2, 2, 2, wr, wsc) <= l1_cb_budget:
            in_depth = out_depth = rm_depth = 2
        elif ws_bytes(regime, block_ht, 2, 1, 2, wr, wsc) <= l1_cb_budget:
            in_depth = rm_depth = 2
        elif ws_bytes(regime, block_ht, 1, 1, 2, wr, wsc) <= l1_cb_budget:
            rm_depth = 2

    forced_block_ht = opd._lever(levers, "block_ht")
    if forced_block_ht:
        max_block_ht = min(max_block_ht, forced_block_ht)

    while block_ht < max_block_ht and ws_bytes(regime, block_ht + 1, in_depth, out_depth, rm_depth, wr, wsc) <= (
        l1_cb_budget
    ):
        block_ht += 1

    if opd._lever(levers, "double_buffer"):
        while in_depth < 4 and ws_bytes(regime, block_ht, in_depth + 1, out_depth, rm_depth, wr, wsc) <= l1_cb_budget:
            in_depth += 1

    assert Wt_core % wr == 0 and Wt_core % wsc == 0
    return opd._Solved(
        regime=regime,
        BLOCK_HT=block_ht,
        WT_REDUCE_BLOCK=wr,
        WT_SCALE_BLOCK=wsc,
        IN_BUF_DEPTH=in_depth,
        OUT_BUF_DEPTH=out_depth,
        RM_BUF_DEPTH=rm_depth,
        GAMMA_DEPTH=1,
        GAMMA_INGEST_BLOCK=opd._largest_divisor_at_most(wsc, gamma_cap_tiles),
        num_row_blocks=opd._div_up(Rt, block_ht),
    )


# name -> (shape, dtype, layout, gamma_layout|None, config, has_gamma)
CASES = {
    # the gate arm this idea now owns: masked (Regime B by construction), wide
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # same datapath with many row-blocks over the whole grid
    "w_nonalign_1024": ((1, 1, 1024, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # CB-WRAP hazard: Wt_core = 96 is NOT a power of two, chunk must divide it,
    # and the last W-tile is partial (W = 96*32 - 1)
    "w_nonalign_3071": ((1, 1, 32, 3071), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # narrower masked shape (Wt = 64)
    "w_nonalign_2047": ((1, 1, 32, 2047), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # aligned but UNSPLITTABLE (Wt = 257 is prime, so P1 forces G = 1) and too wide
    # for Regime A -> Regime B at Wt_core = 257, whose only divisors are 257 and 1
    "wide_prime_8224": ((1, 1, 32, 8224), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # masked, no gamma at all: the rule degenerates to "cb_input_tiles at depth 2"
    "w_nonalign_no_gamma": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", False),
    # masked at the two other supported dtypes (different tile bytes -> different
    # chunk arithmetic; float32 also runs the fp32-DEST corner)
    "w_nonalign_fp32": ((1, 1, 32, 4095), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "default", True),
    "w_nonalign_bf8b": ((1, 1, 32, 4095), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # masked with ROW_MAJOR gamma (staging CB + compute-side tilize)
    "w_nonalign_rm_gamma": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose", True),
    # ROW_MAJOR input, masked (the reader zero-fills the pad tail)
    "w_nonalign_rm_in": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose", True),
    # The risk regimes for a FINER chunk (more chunk boundaries per row-block):
    #   - a chunk halving on the ROW_MAJOR-gamma path, where every chunk pays a
    #     full tilize init/uninit
    #   - a chunk halving with many row-blocks over the whole grid
    #   - a wider masked shape at bf16 and at bfloat8_b (different tile bytes ->
    #     different chunk arithmetic)
    "w_3071_rm_gamma": ((1, 1, 32, 3071), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose", True),
    "w_3071_1024rows": ((1, 1, 1024, 3071), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    "w_8191": ((1, 1, 32, 8191), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    "w_8191_bf8b": ((1, 1, 32, 8191), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # Regime A control: must come out byte-identical under both arms
    "focus_regime_a": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
}

CONFIGS = {"default": _cfg_default, "loose": _cfg_loose}


def _make(device, shape, dtype, layout, gamma_layout, has_gamma):
    import torch  # lazy: ttnn/ forbids a module-level torch import

    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=dtype, layout=layout, device=device)
    g = None
    if has_gamma:
        g = ttnn.from_torch(
            gt,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else gamma_layout,
            device=device,
        )
    return x, g, xt, gt


def _reference(xt, gt, has_gamma, eps=1e-6):
    import torch  # lazy: ttnn/ forbids a module-level torch import

    ref = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + eps)
    return ref * gt.reshape(-1) if has_gamma else ref


def _pcc(a, b):
    import torch  # lazy: ttnn/ forbids a module-level torch import

    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _plan_row(device, x, g, cfg):
    p = opd.blocking_plan(x, g, x, device, opd._apply_precision_levers(cfg, None), None)
    return (
        f"G={p.group_size} reg={p.regime} Wt={p.Wt} Wtc={p.Wt_core} wr={p.WT_REDUCE_BLOCK} "
        f"ws={p.WT_SCALE_BLOCK} bht={p.BLOCK_HT} in={p.IN_BUF_DEPTH} out={p.OUT_BUF_DEPTH} "
        f"rm={p.RM_BUF_DEPTH} gd={p.GAMMA_DEPTH} nrb={p.num_row_blocks} L1={p.working_set_bytes()}"
    )


def run(device):
    """Dispatch every (case, arm); PCC-gate both arms.  Raises on a PCC failure.

    Takes the pytest `device` fixture rather than opening its own device: under
    `--profile` a self-opened device loses the per-op device data at teardown
    ("Op N not present in cpp_device_perf_report.csv").
    """
    import torch  # lazy: ttnn/ forbids a module-level torch import

    from ttnn.operations.rms_norm import rms_norm

    manifest, plans, pccs = [], {}, {}
    shipped_solve = opd._solve
    try:
        for name, (shape, dtype, layout, gl, config, has_gamma) in CASES.items():
            x, g, xt, gt = _make(device, shape, dtype, layout, gl, has_gamma)
            ref = _reference(xt, gt, has_gamma)
            for arm, solve in (("OLD", _solve_old), ("NEW", shipped_solve)):
                opd._solve = solve
                cfg = CONFIGS[config]()
                plans[f"{name}/{arm}"] = _plan_row(device, x, g, CONFIGS[config]())
                out = rms_norm(x, gamma=g, compute_kernel_config=cfg)
                pccs[f"{name}/{arm}"] = _pcc(ttnn.to_torch(out).to(torch.float32), ref)
                ttnn.deallocate(out)
                # One measurement window per arm, dispatched exactly the way
                # _bench_rms_norm._dispatch does it (no interleaved deallocate).
                keep = [rms_norm(x, gamma=g, compute_kernel_config=CONFIGS[config]()) for _ in range(N_WARMUP)]
                ttnn.synchronize_device(device)
                keep += [rms_norm(x, gamma=g, compute_kernel_config=CONFIGS[config]()) for _ in range(N_ITERS)]
                ttnn.synchronize_device(device)
                for t in keep:
                    ttnn.deallocate(t)
                # Flush the device-side profiler buffer per arm.  Without it the
                # zone-heavy kernels overrun it and tracy's report generation dies
                # on "Op N not present in cpp_device_perf_report.csv".
                ttnn.ReadDeviceProfiler(device)
                manifest.append(
                    {
                        "label": f"{name}/{arm}",
                        "shape": name,
                        "config": config,
                        "dtype": str(dtype),
                        "gamma": has_gamma,
                        "levers": {},
                        # +1 for the correctness dispatch, which is NOT profiled
                        "calls": 1 + N_WARMUP + N_ITERS,
                        "profiled": N_ITERS,
                    }
                )
            ttnn.deallocate(x)
            if g is not None:
                ttnn.deallocate(g)
    finally:
        opd._solve = shipped_solve

    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nAB: manifest -> {MANIFEST} ({len(manifest)} arms)")
    for k in plans:
        print(f"  {k:34s} pcc={pccs[k]:.6f}  {plans[k]}")

    bad = {k: v for k, v in pccs.items() if v < PCC_GATE}
    assert not bad, f"PCC gate {PCC_GATE} failed: {bad}"

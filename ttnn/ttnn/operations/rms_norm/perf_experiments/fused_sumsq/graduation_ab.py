# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fused_sumsq graduation: OLD-vs-NEW Regime B reduce phase, in ONE tree.

WHY THIS EXISTS AND NOT A PLAIN BEFORE/AFTER OF THE GATE BENCH: this working tree
is shared with a sibling graduation that edits `kernels/rms_norm_reader.cpp`
concurrently, so an absolute number measured now is NOT comparable to a number
measured on the pre-graduation HEAD.  Every arm here runs in the SAME process, on
the SAME reader/writer kernels, and differs only in which compute kernel + which
`_cb_layout` the program came from.

  arm `base`      the pre-graduation op, verbatim: HEAD's compute kernel (a
                  frozen copy in graduation_baseline/, with the CURRENT
                  reader/writer symlinked in beside it) plus HEAD's `_cb_layout`,
                  which carries the full-block `cb_squared` intermediate.
  arm `fused_iso` the shipped fused compute kernel, but with the blocking SOLVED
                  under HEAD's `_cb_layout` — so G / regime / W-chunk / every CB
                  depth is byte-identical to `base` and the delta is
                  COMPUTE-ONLY.  (The CBs actually created are the shipped set,
                  i.e. the freed L1 is simply left unspent.)
  arm `fused`     the shipped tree end to end: the fused kernel AND the smaller
                  CB set, so the W-chunk search in `_solve` gets to spend the
                  freed L1 on a coarser chunk / deeper buffer.
  arms `fused_p21` / `fused_p22`
                  `fused`, with the W-chunk search's affordability profile set to
                  (in=2, out=1) - the pre-graduation one - and to (in=2, out=2).
                  The shipped `fused` arm now asks for the depth ladder's actual
                  top rung, (in=IN_DEPTH_CAP, out=2), which is what these two
                  measure the cost of NOT asking for.

`fused` - `base` is what the op actually ships; `fused_iso` - `base` attributes
the part of it that is the kernel rather than the replan.

Correctness is the only pass/fail: every arm is PCC-gated against torch on every
shape, including the masked (non-tile-aligned) cases that are most of what is
still Regime B after the W split graduated.

NOTE ON WHERE THE DRIVER LIVES: a pytest file physically inside `ttnn/ttnn/...`
no longer COLLECTS in this tree (pytest's basedir insertion re-imports the ttnn
package under a second name -> "Operation with name bernoulli is already
registered"), so the driver is a shim in the tests tree and the logic lives here.

    scripts/run_safe_pytest.sh --profile --run-all \\
        tests/ttnn/unit_tests/operations/rms_norm/test_fs_graduation_ab.py -s
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
BASELINE_KERNELS = Path(__file__).with_name("graduation_baseline")
RAWALL_KERNELS = Path(__file__).with_name("graduation_rawall")
N_WARMUP = 1
N_ITERS = 3
PCC_GATE = 0.999


# ---------------------------------------------------------------------------
# arm `base`: HEAD's `_cb_layout`, verbatim (HEAD 6bed74e rms_norm_program_
# descriptor.py::_cb_layout).  The ONLY difference from the shipped one is the
# Regime B branch: a `cb_squared` of block_ht * wr pages instead of a
# `cb_sumsq_acc` of 2 * block_ht pages.
# ---------------------------------------------------------------------------
def _cb_layout_head(
    *,
    regime,
    block_ht,
    in_depth,
    out_depth,
    rm_depth,
    gamma_depth,
    wr,
    ws,
    Wt_core,
    has_gamma,
    gamma_is_row_major,
    is_row_major,
    tile_out,
    W_partial,
    gamma_ingest_block,
    T_in,
    T_g,
    T_interm,
    T_acc,
    T_bf16,
    w_split_group=0,
):
    wmax = max(wr, ws)
    if regime == "A":
        wr = ws = wmax = Wt_core

    layout = [
        (opd.CB_INPUT_TILES, in_depth * block_ht * wmax, T_in, "in"),
        (opd.CB_SUMSQ, (2 if regime == "B" else 1) * block_ht, T_acc, "acc"),
        (opd.CB_RMS_RECIP, block_ht, T_acc, "acc"),
        (opd.CB_REDUCE_SCALER, 2 if (regime == "B" and W_partial) else 1, T_bf16, "bf16"),
    ]
    if regime == "A":
        layout.append((opd.CB_SUMSQ_ACC, block_ht, T_acc, "acc"))
    else:
        # the retired slot 3: the full-block x^2 intermediate
        layout.append((3, block_ht * wr, T_interm, "interm"))
    if has_gamma:
        layout.append((opd.CB_GAMMA_TILES, gamma_depth * ws, T_g, "gamma"))
        if gamma_is_row_major:
            layout.append((opd.CB_GAMMA_RM, gamma_ingest_block, T_g, "gamma"))
        layout.append((opd.CB_NORMED, block_ht * ws, T_interm, "interm"))
    layout.append((opd.CB_OUTPUT_TILES, (out_depth if tile_out else 1) * block_ht * ws, T_in, "out"))
    if is_row_major:
        layout.append((opd.CB_RM_IN, rm_depth * wmax, T_in, "in"))
        layout.append((opd.CB_RM_OUT, rm_depth * ws, T_in, "out"))
    if w_split_group:
        layout.append((opd.CB_PARTIAL_GATHER, w_split_group * block_ht, T_acc, "acc"))
        layout.append((opd.CB_SUMSQ_BCAST, block_ht, T_acc, "acc"))
    return layout


# ---------------------------------------------------------------------------
# The W-CHUNK SEARCH's affordability profile, as a knob.
# ---------------------------------------------------------------------------
# `_solve` picks the COARSEST divisor of Wt_core at which a given streaming-CB
# depth profile fits, then a depth ladder spends whatever is left.  The profile is
# therefore what decides how much of the L1 budget the chunk gets to eat, and this
# graduation frees 94-258 KB into exactly that decision - so the profile has to be
# MEASURED, not inherited.  Three arms:
#     (in=2, out=1)  the pre-graduation profile.  With the freed L1 it buys a
#                    coarser chunk out of the writer's second generation:
#                    (1,1,32,3071) goes wr=48/in=4/out=2 -> wr=96/in=2/out=1, -13%.
#     (in=2, out=2)  fixes that, but at wr=128-affordable shapes it now keeps the
#                    coarsest chunk by eating the INPUT depth instead:
#                    w_nonalign no-gamma / bfloat8_b go in=4 -> in=2/3, -8%/-14%.
#     (in=4, out=2)  the ladder's actual TOP rung: the coarsest chunk at which the
#                    whole allocation `_solve` wants is affordable, so coarseness
#                    is never bought with a buffer generation.  This one WON and is
#                    what the shipped `_solve` now asks for, so it is the `fused`
#                    arm; the other two are kept here as its counterfactuals.
def _solve_profile(in_target, out_target):
    def solve(
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

        def ws_bytes(regime, block_ht, in_depth, out_depth, rm_depth, wr, wsc, gamma_depth):
            return opd._working_set_bytes(
                regime=regime,
                block_ht=block_ht,
                in_depth=in_depth,
                out_depth=out_depth,
                rm_depth=rm_depth,
                gamma_depth=gamma_depth,
                wr=wr,
                ws=wsc,
                gamma_ingest_block=opd._largest_divisor_at_most(wsc, gamma_cap_tiles),
                **common,
            )

        fits = ws_bytes("A", 1, 1, 1, 1, Wt_core, Wt_core, 1) <= l1_cb_budget
        regime = "A" if (maskless_w and fits) else "B"

        max_block_ht = max(1, opd._div_up(Rt, max(1, row_parallel_units)))
        max_block_ht = min(max_block_ht, dest_limit)

        block_ht = 1
        in_depth = out_depth = rm_depth = 1

        gamma_streamed = layout_common["has_gamma"] and regime == "B"
        gamma_depth = 2 if (gamma_streamed and opd._lever(levers, "double_buffer")) else 1
        stream_depth = 2 if opd._lever(levers, "double_buffer") else 1

        if regime == "A":
            wr = wsc = Wt_core
        else:
            wr = wsc = 1
            if opd._lever(levers, "coarse_chunk"):
                forced_wt = opd._lever(levers, "wt_block")
                chunk_cap = min(Wt_core, forced_wt) if forced_wt else Wt_core
                # THE PROFILE UNDER TEST.  Scaled by stream_depth so the
                # double_buffer=0 counterfactual still degenerates to depth 1.
                p_in = in_target if stream_depth > 1 else 1
                p_out = out_target if stream_depth > 1 else 1
                for cand in range(chunk_cap, 0, -1):
                    if Wt_core % cand != 0:
                        continue
                    if ws_bytes("B", 1, p_in, p_out, stream_depth, cand, cand, gamma_depth) <= l1_cb_budget:
                        wr = wsc = cand
                        break

        if opd._lever(levers, "double_buffer"):
            if ws_bytes(regime, block_ht, 2, 2, 2, wr, wsc, gamma_depth) <= l1_cb_budget:
                in_depth = out_depth = rm_depth = 2
            elif ws_bytes(regime, block_ht, 2, 1, 2, wr, wsc, gamma_depth) <= l1_cb_budget:
                in_depth = rm_depth = 2
            elif ws_bytes(regime, block_ht, 1, 1, 2, wr, wsc, gamma_depth) <= l1_cb_budget:
                rm_depth = 2

        forced_block_ht = opd._lever(levers, "block_ht")
        if forced_block_ht:
            max_block_ht = min(max_block_ht, forced_block_ht)

        while (
            block_ht < max_block_ht
            and ws_bytes(regime, block_ht + 1, in_depth, out_depth, rm_depth, wr, wsc, gamma_depth) <= l1_cb_budget
        ):
            block_ht += 1

        if opd._lever(levers, "double_buffer"):
            while (
                in_depth < 4
                and ws_bytes(regime, block_ht, in_depth + 1, out_depth, rm_depth, wr, wsc, gamma_depth) <= l1_cb_budget
            ):
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
            GAMMA_DEPTH=gamma_depth,
            GAMMA_INGEST_BLOCK=opd._largest_divisor_at_most(wsc, gamma_cap_tiles),
            num_row_blocks=opd._div_up(Rt, block_ht),
        )

    return solve


# ---------------------------------------------------------------------------
# The arms, as (compute-kernel dir, _cb_layout used to CREATE the CBs,
# _cb_layout the blocking is SOLVED against).
# ---------------------------------------------------------------------------
def _arms():
    shipped_layout = opd._cb_layout
    shipped_solve = opd._solve

    def solve_under(layout_fn):
        """`_solve`, but with the L1 budget accounted against `layout_fn`.

        `_solve` (and `_choose_group_size` through it) reaches `_cb_layout` only via
        the module-global `_working_set_bytes`, so swapping the global for the
        duration of the call re-solves the whole plan — G included — under the other
        CB set, while `blocking_plan`'s own `_cb_layout` call still builds the CBs
        the arm actually creates.
        """

        def solve(**kw):
            saved = opd._cb_layout
            opd._cb_layout = layout_fn
            try:
                return shipped_solve(**kw)
            finally:
                opd._cb_layout = saved

        return solve

    def solve_head_layout(solve):
        """`solve`, with the L1 budget accounted against HEAD's CB set."""

        def wrapped(**kw):
            saved = opd._cb_layout
            opd._cb_layout = _cb_layout_head
            try:
                return solve(**kw)
            finally:
                opd._cb_layout = saved

        return wrapped

    return {
        # THE TRUE PRE-GRADUATION OP: HEAD's compute kernel, HEAD's CB set AND
        # HEAD's chunk-search profile.  This is the arm the coordinator's
        # published gate numbers were measured on.
        "base_p21": (
            BASELINE_KERNELS,
            _cb_layout_head,
            solve_head_layout(_solve_profile(2, 1)),
        ),
        # HEAD's kernel + CB set, but the SHIPPED chunk-search profile — so
        # base_p21 -> base is the policy change alone.
        "base": (BASELINE_KERNELS, _cb_layout_head, solve_under(_cb_layout_head)),
        "fused_iso": (None, shipped_layout, solve_under(_cb_layout_head)),
        "fused": (None, shipped_layout, shipped_solve),
        "fused_p21": (None, shipped_layout, _solve_profile(2, 1)),
        "fused_p22": (None, shipped_layout, _solve_profile(2, 2)),
        # HELPER vs RAW: identical to `fused` except the TILE-ALIGNED chunk goes
        # through the hand-written eltwise_chain expansion of `sum_of_squares`
        # instead of the helper.  Same blocking, same CBs, same everything else.
        "fused_rawall": (RAWALL_KERNELS, shipped_layout, shipped_solve),
    }


# name -> (shape, dtype, layout, gamma_layout|None, config, has_gamma)
#
# Regime B after the W split is essentially the MASKED (non-tile-aligned) wide
# shapes plus the smallest shapes, so that is what this sweeps.  The Regime A /
# split shapes are here as controls: they must come out flat, because the fused
# form IS what Regime A already ran.
CASES = {
    # the guard arm this idea owns: masked, wide, Regime B by construction
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # wider and narrower masked shapes (different chunk arithmetic)
    "w_8191": ((1, 1, 32, 8191), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # CB-WRAP hazard: Wt_core = 96 is not a power of two AND the last W-tile is partial
    "w_3071": ((1, 1, 32, 3071), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    "w_2047": ((1, 1, 32, 2047), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # masked with MANY row-blocks over the whole grid
    "w_nonalign_1024": ((1, 1, 1024, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # ALIGNED but unsplittable (Wt = 257 is prime -> G = 1) and too wide for
    # Regime A: Regime B with NO mask at all, at Wt_core = 257
    "wide_prime_8224": ((1, 1, 32, 8224), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # the DEGENERATE cell the carve-out exists for (Wt_core = 1)
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # masked, no gamma at all
    "w_nonalign_no_gamma": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", False),
    # masked at the other two supported dtypes.  float32 runs the `default`
    # config: (float32, fp32_dest_acc_en=False) is an EXCLUDED cell of this op.
    "w_nonalign_fp32": ((1, 1, 32, 4095), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "default", True),
    "w_nonalign_bf8b": ((1, 1, 32, 4095), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    # masked with ROW_MAJOR gamma (staging CB + compute-side tilize)
    "w_nonalign_rm_gamma": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose", True),
    # ROW_MAJOR input, masked (the reader zero-fills the pad tail -> maskless_w)
    "w_nonalign_rm_in": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose", True),
    # controls that must be FLAT: Regime A / W-split shapes never enter this code
    "focus_regime_a": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose", True),
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
        f"G={p.group_size} reg={p.regime} Wtc={p.Wt_core} wr={p.WT_REDUCE_BLOCK} ws={p.WT_SCALE_BLOCK} "
        f"bht={p.BLOCK_HT} in={p.IN_BUF_DEPTH} out={p.OUT_BUF_DEPTH} rm={p.RM_BUF_DEPTH} "
        f"gd={p.GAMMA_DEPTH} rva={p.reduce_via_add} nrb={p.num_row_blocks} L1={p.working_set_bytes()}"
    )


def run(device):
    """Dispatch every (case, arm); PCC-gate every arm.  Raises on a PCC failure."""
    import torch  # lazy: ttnn/ forbids a module-level torch import

    from ttnn.operations.rms_norm import rms_norm

    arms = _arms()
    shipped_kernel_dir = opd.KERNEL_DIR
    shipped_layout = opd._cb_layout
    shipped_solve = opd._solve

    manifest, plans, pccs = [], {}, {}
    try:
        for name, (shape, dtype, layout, gl, config, has_gamma) in CASES.items():
            x, g, xt, gt = _make(device, shape, dtype, layout, gl, has_gamma)
            ref = _reference(xt, gt, has_gamma)
            for arm, (kdir, layout_fn, solve_fn) in arms.items():
                opd.KERNEL_DIR = shipped_kernel_dir if kdir is None else kdir
                opd._cb_layout = layout_fn
                opd._solve = solve_fn
                key = f"{name}/{arm}"
                plans[key] = _plan_row(device, x, g, CONFIGS[config]())
                out = rms_norm(x, gamma=g, compute_kernel_config=CONFIGS[config]())
                pccs[key] = _pcc(ttnn.to_torch(out).to(torch.float32), ref)
                ttnn.deallocate(out)
                # One measurement window per arm, dispatched the way
                # _bench_rms_norm._dispatch does it (no interleaved deallocate).
                keep = [rms_norm(x, gamma=g, compute_kernel_config=CONFIGS[config]()) for _ in range(N_WARMUP)]
                ttnn.synchronize_device(device)
                keep += [rms_norm(x, gamma=g, compute_kernel_config=CONFIGS[config]()) for _ in range(N_ITERS)]
                ttnn.synchronize_device(device)
                for t in keep:
                    ttnn.deallocate(t)
                # Flush the device-side profiler buffer per arm: the zone-heavy
                # kernels overrun it otherwise and tracy's report generation dies.
                ttnn.ReadDeviceProfiler(device)
                manifest.append(
                    {
                        "label": key,
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
        opd.KERNEL_DIR = shipped_kernel_dir
        opd._cb_layout = shipped_layout
        opd._solve = shipped_solve

    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nAB: manifest -> {MANIFEST} ({len(manifest)} arms)")
    for k in plans:
        print(f"  {k:36s} pcc={pccs[k]:.6f}  {plans[k]}")

    bad = {k: v for k, v in pccs.items() if v < PCC_GATE}
    assert not bad, f"PCC gate {PCC_GATE} failed: {bad}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: moe_fused_swiglu gate/up matmul `in0`-unpack sharing.

Self-contained — does not import or modify the real op. Lives entirely under this
perf_experiments/gateup_in0_share/ directory (own module + own test file) so it cannot collide with
sibling part-optimizers working the same op in parallel.

Run (device lock + reset handled by the harness):
    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/gateup_in0_share/test_gateup_in0_share.py
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
import statistics
from pathlib import Path

from loguru import logger

# NOTE: `torch` is imported LAZILY here. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/` so that importing ttnn
# never drags torch in. These perf-experiment benches live under the op directory, so they
# obey the same rule: every use sites gets `import torch` inside the function.
import ttnn

# Loaded by explicit file path (not a package-dotted import) so this experiment stays fully
# self-contained under perf_experiments/gateup_in0_share/ — no __init__.py needed in the shared
# perf_experiments/ parent, which would risk colliding with sibling part-optimizers' own idea dirs.
_MOD_PATH = Path(__file__).resolve().parent / "gateup_in0_share.py"
_spec = importlib.util.spec_from_file_location("gateup_in0_share_lib", _MOD_PATH)
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
create_program_descriptor_baseline = _lib.create_program_descriptor_baseline
create_program_descriptor_merged = _lib.create_program_descriptor_merged
create_program_descriptor_wide = _lib.create_program_descriptor_wide
create_sharded_memory_config = _lib.create_sharded_memory_config

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
KERNEL_ITERS = 32  # in-kernel repeat: one fresh-cache dispatch, bigger window for a reliable ns/iter.


# =============================================================================
# small utilities
# =============================================================================


def _pcc(a, b):
    import torch

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _quantized(t, device, dtype, mem_cfg):
    """Round-trip through a device format so torch sees exactly the bytes the kernel sees."""
    import torch

    tt = ttnn.from_torch(
        t.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg
    )
    return tt, ttnn.to_torch(tt).to(torch.float32)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_once(device, run_fn):
    run_fn()
    ttnn.synchronize_device(device)
    return _read_kernel_ns(device)


def _measure(device, run_fn, trials=1):
    """`/perf-measure` discipline: one fresh-cache run; only repeat (median) when asked."""
    vals = [v for v in (_measure_once(device, run_fn) for _ in range(trials)) if v is not None]
    return (statistics.median(vals) if vals else None), vals


def _dealloc(*tensors):
    """Free single-core L1 shards between predicate points — everything here is sharded on ONE
    core's ~1.46 MB budget, so leftover tensors from a previous point exhaust it within a handful of
    iterations (observed: TT_FATAL bank_manager OOM after ~2 predicate points without this)."""
    for t in tensors:
        if t is not None:
            ttnn.deallocate(t)


# =============================================================================
# tensor builders
# =============================================================================


def _build_gu_tensors(device, m_eff, kr, hn_pad, seed=0):
    import torch

    torch.manual_seed(seed)
    x_f32 = torch.randn(m_eff * TILE, kr * TILE, dtype=torch.float32) * 0.1
    wg_f32 = torch.randn(kr * TILE, hn_pad * TILE, dtype=torch.float32) * 0.1
    wu_f32 = torch.randn(kr * TILE, hn_pad * TILE, dtype=torch.float32) * 0.1

    x_tt, x_q = _quantized(x_f32, device, ttnn.bfloat8_b, create_sharded_memory_config(m_eff, kr))
    wg_tt, wg_q = _quantized(wg_f32, device, ttnn.bfloat4_b, create_sharded_memory_config(kr, hn_pad))
    wu_tt, wu_q = _quantized(wu_f32, device, ttnn.bfloat4_b, create_sharded_memory_config(kr, hn_pad))

    gate_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m_eff * TILE, hn_pad * TILE]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(m_eff, hn_pad),
    )
    up_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m_eff * TILE, hn_pad * TILE]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(m_eff, hn_pad),
    )
    ref_gate = x_q @ wg_q
    ref_up = x_q @ wu_q
    return dict(x=x_tt, wg=wg_tt, wu=wu_tt, gate_out=gate_out, up_out=up_out, ref_gate=ref_gate, ref_up=ref_up)


def _build_merged_tensor(device, tensors, m_eff, kr, hn_pad):
    """Concatenate the ALREADY-quantized [W_gate | W_up] into one physical in1 CB + one output CB.
    Re-quantizes the concatenation (round-tripping the already-quantized values through bfloat4_b is a
    no-op / idempotent) so the in1 CB is genuinely one contiguous physical buffer, matching option 2's
    "treat W_gate | W_up as one [kr x 2*HN_PAD] in1 block"."""
    import torch

    wgu_f32 = torch.cat([tensors["ref_gate_w"], tensors["ref_up_w"]], dim=1)
    wgu_tt, wgu_q = _quantized(wgu_f32, device, ttnn.bfloat4_b, create_sharded_memory_config(kr, 2 * hn_pad))
    gu_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m_eff * TILE, 2 * hn_pad * TILE]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(m_eff, 2 * hn_pad),
    )
    return wgu_tt, gu_out, wgu_q


def _build_wide_tensors(device, m_eff, kr, n_total, seed=1):
    import torch

    torch.manual_seed(seed)
    x_f32 = torch.randn(m_eff * TILE, kr * TILE, dtype=torch.float32) * 0.1
    w_f32 = torch.randn(kr * TILE, n_total * TILE, dtype=torch.float32) * 0.1
    x_tt, x_q = _quantized(x_f32, device, ttnn.bfloat8_b, create_sharded_memory_config(m_eff, kr))
    w_tt, w_q = _quantized(w_f32, device, ttnn.bfloat4_b, create_sharded_memory_config(kr, n_total))
    acc_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m_eff * TILE, n_total * TILE]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(m_eff, n_total),
    )
    ref = x_q @ w_q
    return x_tt, w_tt, acc_out, ref


# =============================================================================
# The bake-off
# =============================================================================

# (label, kr, hn_pad, hn_real, m_eff) — hn_real=0 means "full" (inert narrowing).
PREDICATE_POINTS = [
    ("focus: kr23 hn6 m8", 23, 6, 0, 8),
    ("kr22 hn6 m8", 22, 6, 0, 8),
    ("kr20 hn6 m8", 20, 6, 0, 8),
    ("kr19 hn6 m8", 19, 6, 0, 8),
    ("kr23 hn6 m4", 23, 6, 0, 4),
    ("kr23 hn6 m1", 23, 6, 0, 1),
]
RAGGED_POINT = ("kr23 hn6(real4) m8", 23, 6, 4, 8)  # baseline-only (see module docstring on the mismatch)

WIDE_SUBBLOCK_WIDTHS = (2, 3, 4, 6, 8)
WIDE_N_TOTAL = 24  # divisible by every width above
WIDE_KR = 23
WIDE_M_EFF = 8


def test_gateup_in0_share_bakeoff(device):
    import torch

    results = {"predicate": [], "wide": [], "skip_compute": None, "correctness": []}

    # ---- 1. correctness: baseline vs merged at the focus shape (kernel_iters=1) ----
    label, kr, hn_pad, hn_real, m_eff = PREDICATE_POINTS[0]
    t = _build_gu_tensors(device, m_eff, kr, hn_pad)
    desc_b = create_program_descriptor_baseline(
        t["x"],
        t["wg"],
        t["wu"],
        t["gate_out"],
        t["up_out"],
        kr_pad=kr,
        kr_real=kr,
        hn_pad=hn_pad,
        hn_real=hn_real,
        m_eff=m_eff,
        kernel_iters=1,
    )
    ttnn.generic_op([t["x"], t["wg"], t["wu"], t["gate_out"], t["up_out"]], desc_b)
    gate_got = ttnn.to_torch(t["gate_out"]).to(torch.float32)
    up_got = ttnn.to_torch(t["up_out"]).to(torch.float32)
    pcc_gate_base = _pcc(t["ref_gate"], gate_got)
    pcc_up_base = _pcc(t["ref_up"], up_got)
    results["correctness"].append(("baseline gate", pcc_gate_base))
    results["correctness"].append(("baseline up", pcc_up_base))
    assert (
        pcc_gate_base > 0.99 and pcc_up_base > 0.99
    ), f"baseline correctness floor tripped: {pcc_gate_base}, {pcc_up_base}"
    _dealloc(t["x"], t["wg"], t["wu"], t["gate_out"], t["up_out"])

    t2 = _build_gu_tensors(device, m_eff, kr, hn_pad, seed=0)
    wgu_tt, gu_out, wgu_q = _build_merged_tensor(
        device,
        dict(ref_gate_w=ttnn.to_torch(t2["wg"]).to(torch.float32), ref_up_w=ttnn.to_torch(t2["wu"]).to(torch.float32)),
        m_eff,
        kr,
        hn_pad,
    )
    desc_m = create_program_descriptor_merged(
        t2["x"],
        wgu_tt,
        gu_out,
        kr_pad=kr,
        kr_real=kr,
        hn_pad=hn_pad,
        hn_real=hn_real,
        m_eff=m_eff,
        kernel_iters=1,
    )
    ttnn.generic_op([t2["x"], wgu_tt, gu_out], desc_m)
    gu_got = ttnn.to_torch(gu_out).to(torch.float32)
    ref_gu = torch.cat([t2["ref_gate"], t2["ref_up"]], dim=1)
    pcc_merged = _pcc(ref_gu, gu_got)
    results["correctness"].append(("merged_1call gate+up", pcc_merged))
    assert pcc_merged > 0.99, f"merged_1call correctness floor tripped: {pcc_merged}"
    _dealloc(t2["x"], t2["wg"], t2["wu"], t2["gate_out"], t2["up_out"], wgu_tt, gu_out)

    # ---- 2. predicate sweep: baseline vs merged, device ns, per point ----
    # Single-core L1 is ~1.46 MB — dealloc every point's tensors before the next allocates, or the
    # bank_manager runs out of space after ~2-3 predicate points (observed on the first pass).
    for label, kr, hn_pad, hn_real, m_eff in PREDICATE_POINTS:
        tb = _build_gu_tensors(device, m_eff, kr, hn_pad)
        desc_base = create_program_descriptor_baseline(
            tb["x"],
            tb["wg"],
            tb["wu"],
            tb["gate_out"],
            tb["up_out"],
            kr_pad=kr,
            kr_real=kr,
            hn_pad=hn_pad,
            hn_real=hn_real,
            m_eff=m_eff,
            kernel_iters=KERNEL_ITERS,
        )
        run_base = lambda a=tb, d=desc_base: ttnn.generic_op([a["x"], a["wg"], a["wu"], a["gate_out"], a["up_out"]], d)
        ns_base, _ = _measure(device, run_base, trials=3 if label.startswith("focus") else 1)
        _dealloc(tb["x"], tb["wg"], tb["wu"], tb["gate_out"], tb["up_out"])

        tm = _build_gu_tensors(device, m_eff, kr, hn_pad, seed=0)
        wgu_tt, gu_out, _ = _build_merged_tensor(
            device,
            dict(
                ref_gate_w=ttnn.to_torch(tm["wg"]).to(torch.float32), ref_up_w=ttnn.to_torch(tm["wu"]).to(torch.float32)
            ),
            m_eff,
            kr,
            hn_pad,
        )
        # tm["gate_out"]/tm["up_out"] are unused by the merged kernel — free them right away.
        _dealloc(tm["gate_out"], tm["up_out"])
        desc_merged = create_program_descriptor_merged(
            tm["x"],
            wgu_tt,
            gu_out,
            kr_pad=kr,
            kr_real=kr,
            hn_pad=hn_pad,
            hn_real=hn_real,
            m_eff=m_eff,
            kernel_iters=KERNEL_ITERS,
        )
        run_merged = lambda a=tm, w=wgu_tt, o=gu_out, d=desc_merged: ttnn.generic_op([a["x"], w, o], d)
        ns_merged, _ = _measure(device, run_merged, trials=3 if label.startswith("focus") else 1)
        _dealloc(tm["x"], tm["wg"], tm["wu"], wgu_tt, gu_out)

        per_iter_base = ns_base / KERNEL_ITERS if ns_base else None
        per_iter_merged = ns_merged / KERNEL_ITERS if ns_merged else None
        results["predicate"].append((label, per_iter_base, per_iter_merged))

    # ---- 2b. ragged column (hn_real=4), baseline only (see docstring on the merged-variant mismatch) ----
    label, kr, hn_pad, hn_real, m_eff = RAGGED_POINT
    tr = _build_gu_tensors(device, m_eff, kr, hn_pad)
    desc_ragged = create_program_descriptor_baseline(
        tr["x"],
        tr["wg"],
        tr["wu"],
        tr["gate_out"],
        tr["up_out"],
        kr_pad=kr,
        kr_real=kr,
        hn_pad=hn_pad,
        hn_real=hn_real,
        m_eff=m_eff,
        kernel_iters=KERNEL_ITERS,
    )
    run_ragged = lambda a=tr, d=desc_ragged: ttnn.generic_op([a["x"], a["wg"], a["wu"], a["gate_out"], a["up_out"]], d)
    ns_ragged, _ = _measure(device, run_ragged, trials=1)
    results["predicate"].append((label, ns_ragged / KERNEL_ITERS if ns_ragged else None, None))
    # correctness for the ragged case: only the first hn_real columns are defined.
    ttnn.generic_op(
        [tr["x"], tr["wg"], tr["wu"], tr["gate_out"], tr["up_out"]],
        create_program_descriptor_baseline(
            tr["x"],
            tr["wg"],
            tr["wu"],
            tr["gate_out"],
            tr["up_out"],
            kr_pad=kr,
            kr_real=kr,
            hn_pad=hn_pad,
            hn_real=hn_real,
            m_eff=m_eff,
            kernel_iters=1,
        ),
    )
    gate_ragged_got = ttnn.to_torch(tr["gate_out"]).to(torch.float32)[:, :hn_real]
    pcc_ragged = _pcc(tr["ref_gate"][:, :hn_real], gate_ragged_got)
    results["correctness"].append(("baseline ragged (hn_real=4) gate", pcc_ragged))
    assert pcc_ragged > 0.99, f"ragged correctness floor tripped: {pcc_ragged}"
    _dealloc(tr["x"], tr["wg"], tr["wu"], tr["gate_out"], tr["up_out"])

    # ---- 3. SKIP_COMPUTE ablation at the focus shape (unpack+math vs pack/CB-sync-only) ----
    label, kr, hn_pad, hn_real, m_eff = PREDICATE_POINTS[0]
    ts = _build_gu_tensors(device, m_eff, kr, hn_pad)
    desc_skip = create_program_descriptor_baseline(
        ts["x"],
        ts["wg"],
        ts["wu"],
        ts["gate_out"],
        ts["up_out"],
        kr_pad=kr,
        kr_real=kr,
        hn_pad=hn_pad,
        hn_real=hn_real,
        m_eff=m_eff,
        kernel_iters=KERNEL_ITERS,
        skip_compute=True,
    )
    run_skip = lambda a=ts, d=desc_skip: ttnn.generic_op([a["x"], a["wg"], a["wu"], a["gate_out"], a["up_out"]], d)
    ns_skip, _ = _measure(device, run_skip, trials=1)
    results["skip_compute"] = ns_skip / KERNEL_ITERS if ns_skip else None
    _dealloc(ts["x"], ts["wg"], ts["wu"], ts["gate_out"], ts["up_out"])

    # ---- 4. wide_subblock sweep: single matrix, out_subblock_w in {2,3,4,6,8}, N=24 fixed ----
    for w in WIDE_SUBBLOCK_WIDTHS:
        x_tt, w_tt, acc_out, ref = _build_wide_tensors(device, WIDE_M_EFF, WIDE_KR, WIDE_N_TOTAL)
        desc_w = create_program_descriptor_wide(
            x_tt,
            w_tt,
            acc_out,
            kr_pad=WIDE_KR,
            kr_real=WIDE_KR,
            n_total=WIDE_N_TOTAL,
            out_subblock_w=w,
            m_eff=WIDE_M_EFF,
            kernel_iters=KERNEL_ITERS,
        )
        run_w = lambda a=x_tt, b=w_tt, c=acc_out, d=desc_w: ttnn.generic_op([a, b, c], d)
        ns_w, _ = _measure(device, run_w, trials=1)
        if w == WIDE_SUBBLOCK_WIDTHS[0]:
            got = ttnn.to_torch(acc_out).to(torch.float32)
            pcc_wide = _pcc(ref, got)
            results["correctness"].append((f"wide_subblock w={w}", pcc_wide))
            assert pcc_wide > 0.99, f"wide_subblock correctness floor tripped: {pcc_wide}"
        results["wide"].append((w, ns_w / KERNEL_ITERS if ns_w else None))
        _dealloc(x_tt, w_tt, acc_out)

    # ---- report ----
    lines = ["", "=== gateup_in0_share bake-off (device kernel ns, per gate+up-block iteration) ==="]
    lines.append(f"KERNEL_ITERS={KERNEL_ITERS} per dispatch (one fresh-cache run; focus point median of 3)")
    lines.append("")
    lines.append("-- correctness (PCC vs fp32 torch ref of the bfp8/bfp4-quantized inputs) --")
    for name, pcc in results["correctness"]:
        lines.append(f"    {name:<32} pcc={pcc:.6f}")
    lines.append("")
    lines.append(f"{'predicate point':<24} {'baseline ns':>12} {'merged_1call ns':>16} {'speedup':>9}")
    for label, b, m in results["predicate"]:
        if b is None:
            lines.append(f"{label:<24} {'N/A':>12}")
            continue
        if m is None:
            lines.append(f"{label:<24} {b:>12.1f} {'(baseline only)':>16}")
        else:
            sp = b / m if m else float("nan")
            lines.append(f"{label:<24} {b:>12.1f} {m:>16.1f} {sp:>8.3f}x")
    lines.append("")
    lines.append(
        f"SKIP_COMPUTE ablation (focus point, baseline kernel): {results['skip_compute']:.1f} ns/iter"
        if results["skip_compute"]
        else "SKIP_COMPUTE ablation: N/A"
    )
    b_focus = results["predicate"][0][1]
    if results["skip_compute"] and b_focus:
        lines.append(f"    unpack+math share of the stage: {(1 - results['skip_compute']/b_focus)*100:.1f}%")
    lines.append("")
    lines.append(f"-- wide_subblock sweep (single matrix, N={WIDE_N_TOTAL}t, kr={WIDE_KR}, m_eff={WIDE_M_EFF}) --")
    lines.append(f"{'out_subblock_w':>14} {'ns/iter':>12} {'vs w=2':>8}")
    base_w = dict(results["wide"]).get(WIDE_SUBBLOCK_WIDTHS[0])
    for w, ns in results["wide"]:
        tag = f"{base_w/ns:.3f}x" if (ns and base_w) else "N/A"
        lines.append(f"{w:>14} {ns if ns else float('nan'):>12.1f} {tag:>8}")
    logger.info("\n".join(lines))

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness (PCC) + single-core device profiling for the rms_norm pass-1 fused+batched-reduce
COMPOSITION bake-off. Reconstructs ONLY the per-core pass-1 of kernels/rms_norm_xcore_compute.cpp
(the PASS1_FUSED path) and A/B's the op's current per-row fused reduce (baseline_fused) vs the same
fused square with a C-row BATCHED reduce (batch_fused / batch_fused_noreconfig). Correctness is the
only pass/fail; perf (DEVICE KERNEL DURATION [ns]) and accuracy (PCC vs fp32 torch) are measured.

Focus contract (never tuned for speed): bf16 input, fp32 output, HiFi2, fp32_dest_acc_en=False,
math_approx_mode=False. Soft PCC gate (focus) = 0.9995; the fused DEST-accumulate is capped at
vwt<=4 for exactly this reason (bf16 DEST accumulation error grows with vwt).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics

import ttnn
from loguru import logger

# Robust import of the sibling descriptor module by absolute path (decoupled from package init).
_MOD_PATH = Path(__file__).with_name("program_descriptor_with_inline_kernels.py")
_spec = importlib.util.spec_from_file_location("pass1_batch_fused_desc", _MOD_PATH)
_desc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_desc)

BASELINE = _desc.BASELINE
VARIANTS = _desc.VARIANTS
create_sharded_memory_config = _desc.create_sharded_memory_config
input_shape = _desc.input_shape
run_op = _desc.run_op
allocate_output = _desc.allocate_output
cb_xsq_tiles = _desc.cb_xsq_tiles

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Focus: BLOCK_SHARDED (1,1,8192,1024), 8x8 grid -> per core HT_LOCAL=32, PER_W_T=4, vwt=4, C_ROWS=8.
_FOCUS = dict(ht_local=32, per_w_t=4, vwt=4, c_rows=8)

# Predicate sweep: C_ROWS x vwt (per_w_t=vwt, tile-aligned). HT kept modest+divisible by every C.
_SWEEP_HT = 16
_SWEEP_C = (1, 4, 8)
_SWEEP_VWT = (1, 2, 4, 8)

_SOFT_PCC = 0.9995  # focus soft gate (reported; the vwt<=4 cap keeps the fused path above it)
_PCC_GATE = 0.99  # hard correctness gate (catches wiring/scale bugs); real PCC reported per variant


# =============================================================================
# Inputs + torch golden
# =============================================================================
def _make_input(device, ht_local, per_w_t, seed=13):
    import torch

    torch.manual_seed(seed)
    h, w = input_shape(ht_local, per_w_t)
    # Signal-like distribution (nonzero mean, O(1)) — exercises bf16 accumulation, like rms_norm x.
    data = torch.randn(h, w)
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, w)),
    )
    return x_dev, data


def _golden(data, vwt, factor):
    import torch

    # per-row partial = (Sum over the vwt*32 reduced columns of x^2) * (1/factor)
    cols = vwt * TILE
    x = data[:, :cols].to(torch.float64)
    return (x * x).sum(dim=1) / factor  # [ht_local*32]


def _readout(output):
    import torch

    # per-row partial lives in column 0 of each output tile; output[:, 0] is all rows' partials.
    return ttnn.to_torch(output).to(torch.float64)[:, 0]


def _pcc(a, b):
    import torch

    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0 if (a.abs().max().item() == 0 and b.abs().max().item() == 0) else 0.0
    return torch.dot(a, b).item() / denom


def _accuracy(output, golden):
    got = _readout(output)
    pcc = _pcc(got, golden)
    max_abs = (got - golden).abs().max().item()
    return pcc, max_abs


# =============================================================================
# In-process device-kernel timing (validated reduce_accumulate pattern)
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {key: [] for key in runners}
    for trial in range(trials + 1):
        for key, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {key}"
            if trial:
                samples[key].append(duration / kernel_iters)
    return samples


def _int(name, default):
    return int(os.environ.get(name, default))


def _arch_label(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _med_std(vals):
    m = statistics.median(vals)
    s = (statistics.pstdev(vals) / m * 100) if (len(vals) > 1 and m) else 0.0
    return m, s


# =============================================================================
# Correctness (all variants; focus + sweep corners, including the vwt=8 over-cap point)
# =============================================================================
def test_pass1_correctness(device):
    configs = [
        dict(**_FOCUS),
        dict(ht_local=_SWEEP_HT, per_w_t=8, vwt=8, c_rows=8),  # over the vwt<=4 cap (accuracy check)
        dict(ht_local=_SWEEP_HT, per_w_t=2, vwt=1, c_rows=4),
        dict(ht_local=12, per_w_t=4, vwt=4, c_rows=8),  # short last round (12 % 8 != 0)
    ]
    for cfg in configs:
        factor = cfg["vwt"] * TILE
        x_dev, data = _make_input(device, cfg["ht_local"], cfg["per_w_t"])
        out_dev = allocate_output(device, cfg["ht_local"])
        golden = _golden(data, cfg["vwt"], factor)
        for variant in VARIANTS:
            out = run_op(x_dev, out_dev, variant=variant, factor=factor, kernel_iters=2, **cfg)
            pcc, max_abs = _accuracy(out, golden)
            logger.info(
                f"{variant:22s} ht={cfg['ht_local']:2d} pwt={cfg['per_w_t']} vwt={cfg['vwt']:2d} C={cfg['c_rows']} "
                f"pcc={pcc:.6f} max_abs={max_abs:.4e}"
            )
            assert pcc >= _PCC_GATE, f"{variant} cfg={cfg}: PCC {pcc:.6f} < {_PCC_GATE}"
        ttnn.deallocate(x_dev)
        ttnn.deallocate(out_dev)


# =============================================================================
# Device perf — focus shape + predicate sweep over (C_ROWS, vwt)
# =============================================================================
def test_pass1_device_perf(device):
    trials = _int("P1_TRIALS", "9")
    kernel_iters = _int("P1_KERNEL_ITERS", "80")

    lines = []
    box, arch = socket.gethostname(), _arch_label(device)
    lines.append(
        "# rms_norm pass-1 fused + C-row batched reduce COMPOSITION — single-core, compute-bound (sharded L1, no DRAM)"
    )
    lines.append(
        f"box={box} arch={arch} cores=1  N={trials} (median)  kernel-iters={kernel_iters}  "
        "contract: bf16 in / fp32 out / HiFi2 / fp32_dest_acc_en=False / math_approx=False (FIXED)"
    )
    lines.append(
        "metric = DEVICE KERNEL DURATION [ns] per pass-1 (all HT_LOCAL tile-rows). "
        "baseline_fused = op PASS1_FUSED (per-row fused square-acc + per-row 1-tile reduce) | "
        "batch_fused = C fused square-accs + ONE batched reduce of(C,1) | "
        "batch_fused_noreconfig = batch_fused w/ reduce INPUT reconfig dropped"
    )
    lines.append("")

    # ---- Focus shape: full accuracy + perf ----
    factor = _FOCUS["vwt"] * TILE
    x_dev, data = _make_input(device, _FOCUS["ht_local"], _FOCUS["per_w_t"])
    out_dev = allocate_output(device, _FOCUS["ht_local"])
    golden = _golden(data, _FOCUS["vwt"], factor)
    acc = {}
    for variant in VARIANTS:
        out = run_op(x_dev, out_dev, variant=variant, factor=factor, kernel_iters=1, **_FOCUS)
        acc[variant] = _accuracy(out, golden)
    runners = {
        v: (lambda vv=v: run_op(x_dev, out_dev, variant=vv, factor=factor, kernel_iters=kernel_iters, **_FOCUS))
        for v in VARIANTS
    }
    samples = _measure(device, runners, trials, kernel_iters)

    lines.append(
        f"## FOCUS  ht_local={_FOCUS['ht_local']} per_w_t={_FOCUS['per_w_t']} vwt={_FOCUS['vwt']} C_ROWS={_FOCUS['c_rows']}"
    )
    lines.append("| variant | median ns/pass1 | std% | speedup vs baseline | pcc | max_abs | cb_xsq tiles |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    base_med, _ = _med_std(samples[BASELINE])
    for v in VARIANTS:
        m, s = _med_std(samples[v])
        pcc, ma = acc[v]
        spd = "-" if v == BASELINE else f"{base_med / m:.3f}x"
        cap = "" if pcc >= _SOFT_PCC else "  <-- below soft gate"
        lines.append(
            f"| {v} | {m:.0f} | {s:.1f} | {spd} | {pcc:.6f}{cap} | {ma:.3e} | {cb_xsq_tiles(v, _FOCUS['c_rows'])} |"
        )
    lines.append("")
    ttnn.deallocate(x_dev)
    ttnn.deallocate(out_dev)

    # ---- Predicate sweep over (C_ROWS, vwt) ----
    lines.append(
        f"## PREDICATE SWEEP  ht_local={_SWEEP_HT}, C_ROWS in {_SWEEP_C}, vwt in {_SWEEP_VWT} (per_w_t=vwt, tile-aligned)"
    )
    lines.append("speedup = baseline_fused / batch_fused (same vwt). PCC listed for both (identical => cap unshifted).")
    lines.append(
        "| vwt | C_ROWS | baseline_fused ns | batch_fused ns | speedup | batch_noreconfig ns | speedup | pcc(base) | pcc(batch) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for vwt in _SWEEP_VWT:
        f = vwt * TILE
        xs, d = _make_input(device, _SWEEP_HT, vwt)
        os_dev = allocate_output(device, _SWEEP_HT)
        g = _golden(d, vwt, f)
        for c in _SWEEP_C:
            cfg = dict(ht_local=_SWEEP_HT, per_w_t=vwt, vwt=vwt, c_rows=c)
            pcc_base = _accuracy(run_op(xs, os_dev, variant="baseline_fused", factor=f, kernel_iters=1, **cfg), g)[0]
            pcc_batch = _accuracy(run_op(xs, os_dev, variant="batch_fused", factor=f, kernel_iters=1, **cfg), g)[0]
            rr = {
                v: (
                    lambda vv=v, xx=xs, oo=os_dev, ff=f, cc=cfg: run_op(
                        xx, oo, variant=vv, factor=ff, kernel_iters=kernel_iters, **cc
                    )
                )
                for v in VARIANTS
            }
            sm = _measure(device, rr, trials, kernel_iters)
            meds = {v: _med_std(sm[v])[0] for v in VARIANTS}
            spd_b = meds["baseline_fused"] / meds["batch_fused"]
            spd_nr = meds["baseline_fused"] / meds["batch_fused_noreconfig"]
            lines.append(
                f"| {vwt} | {c} | {meds['baseline_fused']:.0f} | {meds['batch_fused']:.0f} | {spd_b:.3f}x | "
                f"{meds['batch_fused_noreconfig']:.0f} | {spd_nr:.3f}x | {pcc_base:.5f} | {pcc_batch:.5f} |"
            )
        ttnn.deallocate(xs)
        ttnn.deallocate(os_dev)

    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    out_path = Path(__file__).with_name("report.md")
    out_path.write_text(report)
    logger.info(f"wrote {out_path}")

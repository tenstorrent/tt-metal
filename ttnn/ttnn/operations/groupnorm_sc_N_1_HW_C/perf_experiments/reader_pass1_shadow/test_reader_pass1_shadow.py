# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + on-device measurement for perf_experiments/reader_pass1_shadow.

Correctness is the only pass/fail: in `check` mode every variant must dump (1) the x tiles bit-identical to the
input, (2) the E^T lanes equal to the Python reference of the op's write_membership_lanes(transposed=True), and
(3) the scaler + E^T pages bit-identical to the baseline variant's dump. Perf is measured (DEVICE KERNEL DURATION
[ns] of the perf-mode program = slowest core's pass-1 reader done, in-process device profiler), never asserted.

Run:  scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/perf_experiments/reader_pass1_shadow/test_reader_pass1_shadow.py
Env:  RPS_CASES="focus_1024x640_110c,floor_32x32_1c,..."  RPS_REPEATS=3 (focus)  RPS_REPORT=<path>
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import csv
import importlib
import socket
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

torch = importlib.import_module("torch")  # perf-experiment test under ttnn/: no module-level `import torch` node
import ttnn
from loguru import logger

from ttnn.operations.groupnorm_sc_N_1_HW_C.perf_experiments.reader_pass1_shadow.reader_pass1_shadow_bench import (
    CASES,
    FOCUS,
    TILE,
    VARIANTS,
    Case,
    core_work,
    run_pass1,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_HERE = Path(__file__).resolve().parent
_DEVICE_LOG = Path("generated/profiler/.logs/profile_log_device.csv")
CLK_MHZ = 1350.0


# =============================================================================
# Inputs / references
# =============================================================================
def make_inputs(device, case: Case, seed=7):
    torch.manual_seed(seed)
    x_t = torch.randn(1, 1, case.HW, case.C).to(torch.bfloat16)
    x = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return x_t, x


def alloc_outputs(device, case: Case):
    rows = case.p_used * case.const_pages_per_core * TILE
    out_x = ttnn.from_torch(
        torch.zeros(1, 1, case.HW, case.C, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_const = ttnn.from_torch(
        torch.zeros(rows, TILE, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return out_x, out_const


def membership_reference(case: Case):
    """E^T pages per core as the op's write_membership_lanes(transposed=True) lays them out:
    E_T[c, g'] = 1 iff channel 32T + c belongs to group 32kg + g', tile index tl*Kg + kg. Returns
    {(core_index, cg): tensor(membership_tiles*32, 32)}."""
    Cg = case.C // case.G
    refs = {}
    for w in core_work(case):
        num_col_groups = -(-w.Ct_core // case.cols)
        for cg in range(num_col_groups):
            valid_cols = w.Ct_core - cg * case.cols if cg + 1 == num_col_groups else case.cols
            E = torch.zeros(case.membership_tiles, TILE, TILE)
            for tl in range(valid_cols):
                T = w.col_begin + cg * case.cols + tl
                for c in range(TILE):
                    ch = T * TILE + c
                    if ch >= case.C:
                        break
                    g = ch // Cg
                    E[tl * case.Kg + (g >> 5), c, g & 31] = 1.0
            refs[(w.index, cg)] = E.reshape(case.membership_tiles * TILE, TILE)
    return refs


def _const_page(out_const_t, page):
    return out_const_t[page * TILE : (page + 1) * TILE]


def check_dump(case: Case, variant, x_t, out_x, out_const, base_const=None):
    ox = ttnn.to_torch(out_x)
    assert torch.equal(ox, x_t), f"{case.name} {variant}: x tiles in the CB differ from the input"
    oc = ttnn.to_torch(out_const)
    refs = membership_reference(case)
    P = case.const_pages_per_core
    mt = case.membership_tiles
    for (ci, cg), E in refs.items():
        got = oc[(ci * P + 1 + cg * mt) * TILE : (ci * P + 1 + (cg + 1) * mt) * TILE]
        assert torch.equal(got, E), f"{case.name} {variant}: E^T lanes differ from the reference (core {ci} cg {cg})"
    # scaler page: bf16 1.0 pairs (0x3F803F80) in row 0 of each face = 32 words, everything else zero
    for w in core_work(case):
        words = _const_page(oc, w.index * P).contiguous().view(torch.int32).flatten()
        nz = words[words != 0]
        assert nz.numel() == 32 and bool((nz == 0x3F803F80).all()), f"{case.name} {variant}: scaler page pattern"
    if base_const is not None:
        assert torch.equal(oc, base_const), f"{case.name} {variant}: constant pages not bit-identical to baseline"
    return oc


# =============================================================================
# Measurement
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


def _measure_once(device, x, out_x, out_const, case, variant, zones=False):
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    run_pass1(device, x, out_x, out_const, case, variant=variant, check=False, zones=zones)
    ns = _read_kernel_ns(device)
    assert ns is not None, f"no profiler data for {case.name} {variant}"
    return ns


def _device_log_lines():
    if not _DEVICE_LOG.exists():
        return 0
    with open(_DEVICE_LOG) as f:
        return sum(1 for _ in f)


def _zone_stats(skip_lines):
    """Per-core end time of the reader's r_pass1 zone relative to the earliest kernel start of the run, from the
    lines appended to profile_log_device.csv after `skip_lines`. Returns dict zone -> (p50_us, max_us, cores) for
    the END offsets, plus the p50/max of the zone's own duration."""
    with open(_DEVICE_LOG) as f:
        lines = f.read().splitlines()
    hdr = [h.strip() for h in lines[1].split(",")]
    ix = {h: i for i, h in enumerate(hdr)}
    rows = []
    for line in lines[max(skip_lines, 2) :]:
        f = [t.strip() for t in line.split(",")]
        if len(f) < len(hdr) - 1:
            continue
        rows.append(f)
    if not rows:
        return {}
    run_id = max(int(f[ix["run host ID"]]) for f in rows)
    rows = [f for f in rows if int(f[ix["run host ID"]]) == run_id]
    starts, spans = {}, defaultdict(list)  # zone -> [(core, start, end)]
    for f in rows:
        core = (int(f[ix["core_x"]]), int(f[ix["core_y"]]))
        key = (core, f[ix["RISC processor type"]], f[ix["zone name"]])
        t = int(f[ix["time[cycles since reset]"]])
        if f[ix["type"]] == "ZONE_START":
            starts[key] = t
        elif f[ix["type"]] == "ZONE_END" and key in starts:
            spans[key[2]].append((core, starts.pop(key), t))
    t0 = min(s for z, v in spans.items() if z.endswith("-KERNEL") for (_, s, _e) in v)
    out = {}
    for z, v in spans.items():
        ends = [(e - t0) / CLK_MHZ for (_, _s, e) in v]
        durs = [(e - s) / CLK_MHZ for (_, s, e) in v]
        out[z] = dict(
            end_p50=statistics.median(ends),
            end_max=max(ends),
            dur_p50=statistics.median(durs),
            dur_max=max(durs),
            n=len(v),
        )
    return out


def _cases():
    names = os.environ.get("RPS_CASES")
    return [CASES[n] for n in names.split(",")] if names else list(CASES.values())


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("case", list(CASES.values()), ids=lambda c: c.name)
def test_reader_pass1_shadow_correctness(device, case):
    x_t, x = make_inputs(device, case)
    base_const = None
    for variant in VARIANTS:
        out_x, out_const = alloc_outputs(device, case)
        run_pass1(device, x, out_x, out_const, case, variant=variant, check=True)
        oc = check_dump(case, variant, x_t, out_x, out_const, base_const)
        if variant == "baseline":
            base_const = oc
        logger.info(f"{case.name} {variant}: dump bit-exact (x tiles, scaler, E^T lanes)")


def test_reader_pass1_shadow_device_perf(device):
    repeats = int(os.environ.get("RPS_REPEATS", "3"))
    rows = []  # (case, variant, [ns...], zone dict or None)
    for case in _cases():
        x_t, x = make_inputs(device, case)
        base_const = None
        for variant in VARIANTS:
            out_x, out_const = alloc_outputs(device, case)
            run_pass1(device, x, out_x, out_const, case, variant=variant, check=True)  # gate before timing
            oc = check_dump(case, variant, x_t, out_x, out_const, base_const)
            if variant == "baseline":
                base_const = oc
            samples = [_measure_once(device, x, out_x, out_const, case, variant) for _ in range(repeats)]
            zones = None
            if case.name == FOCUS.name:
                skip = _device_log_lines()
                _measure_once(device, x, out_x, out_const, case, variant, zones=True)
                zones = _zone_stats(skip)
            rows.append((case, variant, samples, zones))
            logger.info(f"{case.name} {variant}: {[f'{s:.0f}' for s in samples]} ns")
    report = _format_report(rows, box=socket.gethostname(), arch=str(device.arch()))
    logger.info("\n" + report)
    Path(os.environ.get("RPS_REPORT", _HERE / "report_perf_sweep.md")).write_text(report)


def _format_report(rows, *, box, arch):
    lines = [
        "# groupnorm_sc_N_1_HW_C / perf_experiments/reader_pass1_shadow — pass-1 reader schedule",
        "",
        f"box={box}  arch={arch}  metric=DEVICE KERNEL DURATION [ns] of the pass-1 reader + consumer-stub program "
        "(perf mode, zones compiled out); median of RPS_REPEATS fresh-launch runs per cell (each sample is one "
        "launch; device kernel time has no warm-up transient).",
        "",
        "Pure dataflow reorder: every variant's x tiles / scaler / E^T pages are bit-identical to the baseline's "
        "(gated in check mode before every timing).",
        "",
        "| case | cores | per-core block | variant | ns (median) | samples | speedup | r_pass1 end p50 / max us | "
        "r_x_barrier dur p50 / max us | r_memb_fill dur p50 / max us | r_zero_fill dur p50 / max us |",
        "|---|---:|---|---|---:|---|---:|---|---|---|---|",
    ]
    base = {}
    for case, variant, samples, zones in rows:
        med = statistics.median(samples)
        if variant == "baseline":
            base[case.name] = med
        speed = base[case.name] / med if med > 0 else float("nan")
        blk = f"Ht {case.Ht_core_max} x Ct {case.Ct_core_max}, cols {case.cols}, chunk_rows {case.chunk_rows}, {'resident' if case.resident else 'streaming'}"

        def z(name, key):
            if not zones or name not in zones:
                return "-"
            return f"{zones[name][key + '_p50']:.2f} / {zones[name][key + '_max']:.2f}"

        lines.append(
            f"| {case.name} | {case.p_used} | {blk} | {variant} | {med:.0f} | {', '.join(f'{s:.0f}' for s in samples)} | "
            f"{speed:.3f}x | {z('r_pass1', 'end')} | {z('r_x_barrier', 'dur')} | {z('r_memb_fill', 'dur')} | "
            f"{z('r_zero_fill', 'dur')} |"
        )
    return "\n".join(lines) + "\n"

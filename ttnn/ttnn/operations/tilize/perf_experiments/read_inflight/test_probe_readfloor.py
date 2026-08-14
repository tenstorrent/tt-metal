# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight DIAGNOSTIC — the read floor of the DRAM -> L1 crossover.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/read_inflight/test_probe_readfloor.py

Reader kernel ONLY (no CB handshake, no compute, no writer). Two axes, both at
the focus plan's per-core byte count (128 KB per core over 8 cores):

  A. IN-FLIGHT WINDOW — 512 B pages, barrier every {32, 64, 128, 256} pages.
     32 is exactly what `read_sticks_for_tilize` does (one barrier per tile-row).
  B. TRANSACTION COUNT — same 128 KB per core, but as {512, 1024, 2048, 4096} B
     pages, always with the whole run under ONE barrier (max in flight).
  C. CORE COUNT — the 512 B / 32-page shape on 8 vs 64 cores, i.e. is the
     crossover's read per-core-issue-bound or fabric-bound?

Nothing is asserted about speed; the numbers are printed.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pathlib

import pytest


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py
# — the shipped package must not drag torch in). These perf-experiment benches DO need it
# for their bit-exact oracle, so the import is done inside a function scope and published
# under the module-global name, which keeps every `torch.` use below unchanged.
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

KERNEL_DIR = pathlib.Path(__file__).parent / "experiment_kernels"
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

PER_CORE_BYTES = 128 * 1024  # the focus plan's per-core read
WINDOW_BYTES = 128 * 1024


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _src(device, page_bytes, total_pages):
    """A DRAM interleaved ROW_MAJOR tensor whose PAGE is exactly `page_bytes`."""
    w = page_bytes // 2  # bf16
    torch_in = torch.randn([1, 1, total_pages, w]).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _measure(device, label, *, page_bytes, pages_per_barrier, num_cores):
    pages_per_core = PER_CORE_BYTES // page_bytes
    total_pages = pages_per_core * num_cores
    tt_in = _src(device, page_bytes, total_pages)

    grid_x = min(num_cores, 8)
    cores = [ttnn.CoreCoord(c % grid_x, c // grid_x) for c in range(num_cores)]
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})

    cb = ttnn.CBDescriptor(
        total_size=WINDOW_BYTES,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=page_bytes)],
    )
    ct = [page_bytes, pages_per_barrier, WINDOW_BYTES]
    ct.extend(ttnn.TensorAccessorArgs(tt_in).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        rt[core.x][core.y] = [tt_in.buffer_address(), i * pages_per_core, pages_per_core]

    desc = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "probe_read.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[cb],
    )
    # generic_op wants >= 2 io tensors; this probe has one real input, so a tiny
    # unused sink stands in for the output. No kernel ever touches it.
    sink = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM),
        device,
    )
    ttnn.generic_op([tt_in, sink], desc)
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    ttnn.generic_op([tt_in, sink], desc)
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)

    per_core_gbps = PER_CORE_BYTES / ns
    logger.info(
        f"READFLOOR {label}: ns={ns} page={page_bytes}B ppb={pages_per_barrier} cores={num_cores} "
        f"xacts/core={pages_per_core} ns/xact={ns / pages_per_core:.1f} "
        f"per_core={per_core_gbps:.2f}GB/s aggregate={per_core_gbps * num_cores:.1f}GB/s"
    )
    assert ns is not None
    return ns


# A. in-flight window at the focus plan's 512 B transfer
@pytest.mark.parametrize("ppb", [32, 64, 128, 256], ids=lambda v: f"ppb{v}")
def test_inflight_window(device, ppb):
    _measure(device, f"A/inflight/ppb{ppb}", page_bytes=512, pages_per_barrier=ppb, num_cores=8)


# B. transaction count at max in flight (same 128 KB per core)
@pytest.mark.parametrize("page_bytes", [512, 1024, 2048, 4096], ids=lambda v: f"page{v}")
def test_transaction_size(device, page_bytes):
    _measure(
        device,
        f"B/xact/{page_bytes}B",
        page_bytes=page_bytes,
        pages_per_barrier=PER_CORE_BYTES // page_bytes,
        num_cores=8,
    )


# C. core count on the crossover's own read shape
@pytest.mark.parametrize("num_cores", [8, 32, 64], ids=lambda v: f"cores{v}")
def test_core_count(device, num_cores):
    _measure(device, f"C/cores/{num_cores}", page_bytes=512, pages_per_barrier=32, num_cores=num_cores)

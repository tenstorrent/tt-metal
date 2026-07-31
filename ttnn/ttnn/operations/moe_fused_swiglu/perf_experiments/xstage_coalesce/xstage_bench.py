# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED BAKE-OFF (blocking-perf-part-optimizer) — moe_fused_swiglu's `x` activation staging
path: the row-major bf16 stick read (`reader_xstage`) + its fused-tilize twin (`compute_tilize`).

Assigned idea: speed up the reader/compute pair that lands one injected tile-row of `x` — 32
row-major bf16 stick SLICES (1472 B each at KR_PAD=23) read from DRAM, fused-tilized to bfp8_b,
then self-copied into the resident in0 slot. This module is a SELF-CONTAINED reconstruction of
just that stage (single core, no mcast/reduce/h-gather/weight streams — those are held out per
/perf-lab's concept-isolation table) with a VARIANT switch selecting the candidate read strategy.

VARIANT 0 baseline             — verbatim: 32 sub-page reads, 1 barrier, tilize, self-copy.
VARIANT 1 wide_read_individual — 32 WHOLE-page (14336 B) reads + 32 L1->L1 extraction copies.
VARIANT 2 bank_run_grouped     — NUM_BANKS bank-contiguous WHOLE-page group reads (the op's own
                                  WRUN trick, applied to whole pages) + 32 extraction copies.
VARIANT 3 dual_noc_split       — reader(NoC0)/writer(NoC1) split the 32 reads 16/16.
VARIANT 4 bfp8_tile_direct     — the op's INPUT_FORMAT==1 twin: kr whole bfp8 tiles, no tilize.
VARIANT 5 self_copy_ablation   — baseline minus the final self-copy (diagnostic, not correctness-
                                  gated).

Real-op kernels reconstructed from (READ, never touched):
  ttnn/ttnn/operations/moe_fused_swiglu/kernels/moe_fused_swiglu_reader.cpp  (reader_xstage, ~L311)
  ttnn/ttnn/operations/moe_fused_swiglu/kernels/moe_fused_swiglu_compute.cpp (compute_tilize, ~L176)
  ttnn/ttnn/operations/moe_fused_swiglu/kernels/moe_fused_swiglu_bank_runs.hpp (WRUN coalescing)

This module is a plain library (NOT collected by pytest — pytest's --import-mode=importlib cannot
safely collect a test_*.py file living inside the `ttnn/ttnn/operations/...` tree; it derives a
dotted module path starting with "ttnn" and re-executes ttnn/ttnn/__init__.py under a second
qualified name, crashing on duplicate C++ op registration). The pytest entry point that imports
this module lives at
tests/ttnn/unit_tests/operations/moe_fused_swiglu/perf_probes/test_xstage_coalesce_bench.py
(a thin wrapper, uniquely named for this idea so it cannot collide with a sibling part-optimizer's
probe). Everything else — kernels, this module, all analysis — lives under this experiment dir.
"""

import os

# Enable the on-device profiler IN-PROCESS (must be set before the device opens). The pytest entry
# point imports this module before opening the device, so setting it here (module import time) is
# early enough; scoped via setdefault so it never clobbers an outer profiler run.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from pathlib import Path


# NOTE: `torch` is imported LAZILY here. `scripts/validate_no_global_torch_imports.py`
# forbids a module-level torch import anywhere under `ttnn/ttnn/` so that importing ttnn
# never drags torch in. These perf-experiment benches live under the op directory, so they
# obey the same rule: every use sites gets `import torch` inside the function.
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

# CB indices — must match the hardcoded constants in kernels/xstage_*.cpp.
CB_X_IN = 0
CB_X_STAGE = 1
CB_X_RESIDENT = 2
CB_X_WIDE = 3
CB_X_BANKGRP = 4

VARIANTS = {
    0: "baseline",
    1: "wide_read_individual",
    2: "bank_run_grouped",
    3: "dual_noc_split",
    4: "bfp8_tile_direct",
    5: "self_copy_ablation",
}
CORRECTNESS_GATED = (0, 1, 2, 3, 4)  # 5 (self_copy_ablation) intentionally skips the copy -> garbage output

SEM_SPLIT = 0  # the one semaphore id this experiment uses (VARIANT 3 only)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


# ---------------------------------------------------------------------------
# Precision contract — mirrors moe_fused_swiglu.default_compute_kernel_config() VERBATIM. Fixed
# input, never a lever: math_fidelity/math_approx_mode/fp32_dest_acc_en/dst_full_sync_en are set in
# stone by the op's own default and used identically by every variant here.
# ---------------------------------------------------------------------------
def compute_kernel_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


def _core():
    return ttnn.CoreCoord(0, 0)


def _core_range_set():
    c = _core()
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])


def num_banks(device):
    return int(ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.DRAM).num_banks)


def make_x_bf16_rm(device, emb, seed=0):
    import torch

    torch.manual_seed(seed)
    torch_x = torch.randn(TILE, emb, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return torch_x, tt_x


def make_x_bfp8_tile(device, emb, seed=0):
    import torch

    torch.manual_seed(seed)
    torch_x = torch.randn(TILE, emb, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return torch_x, tt_x


def reference_tiles(torch_x, kstart_tiles, kr_pad):
    """ttnn's own bf16->bfloat8_b tilize of the same slice, as the correctness ground truth."""
    import torch

    lo, hi = kstart_tiles * TILE, (kstart_tiles + kr_pad) * TILE
    sl = torch_x[:, lo:hi].contiguous()
    tt_ref = ttnn.from_torch(sl, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt_ref).to(torch.float32)


def build_program(device, variant, x_tensor, out_tensor, *, kr_pad, kstart_tiles, emb, banks):
    core_ranges = _core_range_set()
    bfp8_tile = ttnn.tile_size(ttnn.bfloat8_b)
    x_page = emb * 2  # bf16 stick bytes (unused/ignored for VARIANT 4, which reads the accessor at bfp8_tile page size)
    x_slice = kr_pad * TILE * 2  # cb_x_in page: kr*32*2 bytes

    cbs = [
        ttnn.CBDescriptor(
            total_size=TILE * x_slice,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_X_IN, data_format=ttnn.bfloat16, page_size=x_slice)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=kr_pad * bfp8_tile,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_X_STAGE, data_format=ttnn.bfloat8_b, page_size=bfp8_tile)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=kr_pad * bfp8_tile,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_X_RESIDENT, data_format=ttnn.bfloat8_b, page_size=bfp8_tile)
            ],
        ),
    ]
    if variant == 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=TILE * x_page,
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_X_WIDE, data_format=ttnn.bfloat16, page_size=x_page)
                ],
            )
        )
    if variant == 2:
        pages_per_bank = TILE // banks
        assert pages_per_bank * banks == TILE, f"NUM_BANKS {banks} must divide {TILE}"
        bank_group_bytes = pages_per_bank * x_page
        cbs.append(
            ttnn.CBDescriptor(
                total_size=banks * bank_group_bytes,
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_X_BANKGRP, data_format=ttnn.bfloat16, page_size=bank_group_bytes
                    )
                ],
            )
        )

    semaphores = []
    if variant == 3:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_SPLIT, core_ranges=core_ranges, initial_value=0))

    reader_ct = [variant, kr_pad, x_slice, x_page, bfp8_tile, banks, SEM_SPLIT]
    reader_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())
    writer_ct = [variant, kr_pad, x_slice, x_page, bfp8_tile, SEM_SPLIT]
    writer_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())
    compute_ct = [kr_pad]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    core = _core()
    x_addr = x_tensor.buffer_address()
    out_addr = out_tensor.buffer_address()
    kr = kr_pad  # no ragged tail in this isolated bench
    reader_rt[core.x][core.y] = [x_addr, kstart_tiles, kr]
    writer_rt[core.x][core.y] = [x_addr, out_addr, kstart_tiles, kr]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "xstage_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "xstage_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    if variant != 4:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "xstage_compute.cpp"),
                core_ranges=core_ranges,
                compile_time_args=compute_ct,
                runtime_args=ttnn.RuntimeArgs(),
                config=compute_kernel_config(),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def run_variant(device, variant, *, emb, kr_pad, kstart_tiles=0, seed=0):
    banks = num_banks(device)
    if variant == 4:
        torch_x, tt_x = make_x_bfp8_tile(device, emb, seed=seed)
    else:
        torch_x, tt_x = make_x_bf16_rm(device, emb, seed=seed)

    out_shape = (TILE, kr_pad * TILE)
    out_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_shape)), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = build_program(
        device, variant, tt_x, out_tensor, kr_pad=kr_pad, kstart_tiles=kstart_tiles, emb=emb, banks=banks
    )
    result = ttnn.generic_op([tt_x, out_tensor], program)
    return torch_x, result


def read_kernel_ns(device):
    """DEVICE KERNEL DURATION summed over programs dispatched since the last read.
    ReadDeviceProfiler flushes the queue and consumes the window (see
    examples/double_buffer/test_double_buffer.py's identical helper)."""
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


def measure_once(device, run_fn):
    """/perf-measure discipline: ONE fresh-cache run, no averaging loop — device kernel time has
    no warm-up transient, so a trial loop just re-measures the same number N times."""
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # drain any pending window
    run_fn()
    ns = read_kernel_ns(device)
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    return ns


# ---------------------------------------------------------------------------
# Raw per-transaction rate probe — reader-only program, no compute/writer/tilize. Answers "what IS
# the per-transaction cost of this op's sub-page interleaved stick read, and is it near the
# ~110-125 ns/transaction single-core floor examples/double_buffer/report.md measured for WHOLE
# TILE reads, or far above it (i.e. something else is the limiter)?"
# ---------------------------------------------------------------------------
def build_rateprobe_program(device, x_tensor, *, kr_pad, num_reads):
    core_ranges = _core_range_set()
    x_slice = kr_pad * TILE * 2
    reader_ct = [x_slice]
    reader_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())

    cbs = [
        ttnn.CBDescriptor(
            total_size=TILE * x_slice,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_X_IN, data_format=ttnn.bfloat16, page_size=x_slice)
            ],
        )
    ]
    core = _core()
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[core.x][core.y] = [x_tensor.buffer_address(), 0, kr_pad, num_reads, x_tensor.buffer_page_size()]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rateprobe_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run_rateprobe(device, x_tensor, *, kr_pad, num_reads):
    program = build_rateprobe_program(device, x_tensor, kr_pad=kr_pad, num_reads=num_reads)
    # generic_op requires >= 1 input AND >= 1 output tensor (io_tensors.size() >= 2). This probe is
    # reader-only (no kernel writes anywhere) so the "output" is an unused dummy DRAM tensor.
    dummy_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.generic_op([x_tensor, dummy_out], program)

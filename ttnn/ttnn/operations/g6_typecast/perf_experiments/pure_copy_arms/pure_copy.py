# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pure_copy_arms: the bit-identical typecast arms (uint32->int32, int32->uint32) as a pure page copy.

No compute kernel. Each "mover" owns a contiguous page range (many pages) or a 64B-aligned byte chunk
of one page (fewer pages than movers), copies src -> local L1 -> dst by raw NoC reads/writes.

Variants:
  one  : one mover kernel per core (RISCV_0 unless noc says otherwise)
  dual : two independent movers per core (RISCV_0 on NOC_1, RISCV_1 on NOC_0), each its own slice
  two  : reader (RISCV_1/NOC_0) + writer (RISCV_0/NOC_1) on a shared 2-slot CB
"""
from pathlib import Path
import math

import ttnn

KDIR = Path(__file__).parent / "kernels"
BUF_CAP = 64 * 1024  # L1 scratch cap per mover (one/dual); two: 2 slots of BUF_CAP/2


def plan(num_pages, page_bytes, movers, min_chunk, align):
    """-> list of (page0, count, off, len) per mover (only non-empty ones)."""
    if num_pages >= movers:
        base, rem = divmod(num_pages, movers)
        out, p = [], 0
        for m in range(movers):
            c = base + (m < rem)
            out.append((p, c, 0, page_bytes))
            p += c
        return out
    per_page = max(1, min(movers // num_pages, math.ceil(page_bytes / min_chunk)))
    chunk = math.ceil(math.ceil(page_bytes / per_page) / align) * align
    out = []
    for p in range(num_pages):
        for off in range(0, page_bytes, chunk):
            out.append((p, 1, off, min(chunk, page_bytes - off)))
    return out


def _cores(device, n, first=0):
    g = device.compute_with_storage_grid_size()
    assert first + n <= g.x * g.y
    return [ttnn.CoreCoord(k % g.x, k // g.x) for k in range(first, first + n)]


def _bank_cores(device, units, inp):
    """One core per unit. DRAM, single-page units: the unit of page p (bank p % nb) goes to the logical
    row the device names optimal for that bank on NoC0 (the mover's NoC); spills to the nearest free row.
    Otherwise row-major."""
    g = device.compute_with_storage_grid_size()
    if inp.memory_config().buffer_type != ttnn.BufferType.DRAM or any(c != 1 for (_, c, _, _) in units):
        return _cores(device, len(units))
    nb = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    best = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    used = set()
    out = []
    for p, _, _, _ in units:
        y0 = best[p % nb].y
        for y in sorted(range(g.y), key=lambda y: abs(y - y0)):
            free = [x for x in range(g.x) if (x, y) not in used]
            if free:
                used.add((free[0], y))
                out.append(ttnn.CoreCoord(free[0], y))
                break
    return out


def _dm(proc, noc):
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if proc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1,
    )


def descriptor(inp, out, variant, movers, min_chunk=64, noc=0, first_core=0, buf_cap=BUF_CAP):
    wnoc = 2
    if variant == "onex":  # one RISC, reads on `noc`, writes on the other NoC
        variant, wnoc = "one", 1 - noc
    device = inp.device()
    page = inp.buffer_aligned_page_size()
    assert page == out.buffer_aligned_page_size()
    npages = inp.buffer_num_pages()
    align = 64 if inp.memory_config().buffer_type == ttnn.BufferType.DRAM else 16
    if variant in ("auto", "auto1"):
        units = auto_plan(device, npages, page, align)
        g = device.compute_with_storage_grid_size()
        chunk_mode = npages < g.x * g.y
        # auto: chunk mode (fewer pages than cores) -> ONE NoC0 mover per core, bank-row placed (latency-bound);
        #       page mode -> reader NoC0 + writer NoC1 (bandwidth-bound: keep write traffic off the read NoC).
        if variant == "auto" and not chunk_mode:
            variant, first_core = "two", 0
        else:
            variant, first_core, noc = "one", -1, 0
    else:
        units = plan(npages, page, movers, min_chunk, align)
    per_core = 2 if variant == "dual" else 1
    ncores = math.ceil(len(units) / per_core)
    if first_core < 0:
        assert per_core == 1
        cores = _bank_cores(device, units, inp)
    else:
        cores = _cores(device, ncores, first_core)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    biggest = max(c * l for (_, c, _, l) in units)
    maxlen = max(l for (*_, l) in units)
    if variant == "two":
        slot = min(buf_cap // 2, biggest)
        if maxlen <= slot:
            slot = (slot // maxlen) * maxlen
        nslots = 2
    else:
        slot = min(buf_cap, biggest)
        if maxlen <= slot:
            slot = (slot // maxlen) * maxlen
        nslots = 1
    src_ct = list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
    dst_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    a_in, a_out = inp.buffer_address(), out.buffer_address()

    def cb(idx):
        return ttnn.CBDescriptor(
            total_size=nslots * slot,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=inp.dtype, page_size=slot)],
        )

    kernels, cbs = [], []
    if variant == "one":
        rt = ttnn.RuntimeArgs()
        for c, u in zip(cores, units):
            rt[c.x][c.y] = [a_in, a_out, *u]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "mover.cpp"),
                core_ranges=crs,
                compile_time_args=[page, 0, slot, wnoc] + src_ct + dst_ct,
                runtime_args=rt,
                config=_dm(0, noc),
            )
        )
        cbs.append(cb(0))
    elif variant == "dual":
        for r in (0, 1):
            rt = ttnn.RuntimeArgs()
            for i, c in enumerate(cores):
                k = 2 * i + r
                rt[c.x][c.y] = [a_in, a_out, *(units[k] if k < len(units) else (0, 0, 0, 0))]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KDIR / "mover.cpp"),
                    core_ranges=crs,
                    compile_time_args=[page, r, slot, 2] + src_ct + dst_ct,
                    runtime_args=rt,
                    config=_dm(r, (1 - r) if noc < 2 else noc - 2),
                )
            )
            cbs.append(cb(r))
    elif variant == "two":
        rt = ttnn.RuntimeArgs()
        for c, u in zip(cores, units):
            rt[c.x][c.y] = [a_in, a_out, *u]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "reader.cpp"),
                core_ranges=crs,
                compile_time_args=[page, 0, slot] + src_ct,
                runtime_args=rt,
                config=_dm(1, 0),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "writer.cpp"),
                core_ranges=crs,
                compile_time_args=[page, 0, slot] + dst_ct,
                runtime_args=rt,
                config=_dm(0, 1),
            )
        )
        cbs.append(cb(0))
    else:
        raise ValueError(variant)
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs), len(cores)


CHUNK_TARGET = 1024  # bytes per mover a single page is split into (knee measured on Q2220/Q2189/Q2226)


def auto_plan(device, num_pages, page_bytes, align, grid_cores=None):
    """RECOMMENDED host rule. pages >= cores: contiguous page ranges over all cores (row-major).
    Fewer pages than cores: each page split into min(ceil(P/1024), cores//pages, grid.x) 64B-aligned chunks,
    one chunk per core, cores placed on the NoC0-optimal row of the page's DRAM bank."""
    g = device.compute_with_storage_grid_size()
    cores = grid_cores or g.x * g.y
    if num_pages >= cores:
        return plan(num_pages, page_bytes, cores, page_bytes, align)
    per_page = max(1, min(math.ceil(page_bytes / CHUNK_TARGET), cores // num_pages, g.x))
    chunk = math.ceil(math.ceil(page_bytes / per_page) / align) * align
    return plan(num_pages, page_bytes, num_pages * per_page, chunk, align)


def pure_copy(inp, out_dtype, variant="one", movers=1, min_chunk=64, noc=0, first_core=0, buf_cap=BUF_CAP):
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(inp.shape)), out_dtype, inp.layout, inp.device(), inp.memory_config()
    )
    desc, _ = descriptor(inp, out, variant, movers, min_chunk, noc, first_core, buf_cap)
    return ttnn.generic_op([inp, out], desc)

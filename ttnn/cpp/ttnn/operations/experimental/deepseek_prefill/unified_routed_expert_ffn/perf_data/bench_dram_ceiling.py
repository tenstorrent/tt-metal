# GATE 0: DRAM read-bandwidth ceiling sweep on the local Blackhole via ttnn.generic_op (scratch).
import json
import os
import statistics
import sys
import time

import torch
from loguru import logger

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import RESULTS, RtProfile, TILE_BYTES, DTYPE, append_jsonl, env_info, open_dev  # noqa: E402

KPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "dram_ceiling_reader.cpp")
KSUB = "dram_ceiling_reader"
RING = 256 * 1024
PASSES = 4  # read the whole tensor this many times per program (lengthens the window)


def crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in cores])


def row_major_cores(grid, k):
    cores = [(x, y) for y in range(grid[1]) for x in range(grid[0])]
    return cores[:k]


def bank_cores(info, grid, k, nb):
    base = [tuple(c) for c in info["bank_cores_noc0"]]
    out = []
    per_bank = (k + nb - 1) // nb
    for r in range(per_bank):
        for b in range(nb):
            x, y = base[b]
            c = (x, (y + r) % grid[1])
            if c not in out:
                out.append(c)
    return out[:k], [i % nb for i in range(len(out[:k]))]


def vc_for(cores, banks):
    vcs = []
    for i, (c, b) in enumerate(zip(cores, banks)):
        vc = b & 3
        for j in range(i):
            if cores[j][1] == c[1] and (banks[j] & 3) == (b & 3):
                vc = (vc + 1) & 3
                break
        vcs.append(vc)
    return vcs


def run_config(dev, w, out, info, *, mode, dtype_key, xfer, group, k, placement, nocs, nb):
    grid = (info["grid_x"], info["grid_y"])
    page = TILE_BYTES[dtype_key]
    num_pages = w.buffer_address() and (w.shape[0] // 32) * (w.shape[1] // 32)
    if placement == "bank":
        cores, banks = bank_cores(info, grid, k, nb)
    elif placement == "col":
        # the op's band layout: core (gx, gy), gx < nb, reads bank gx; k/nb rows
        rows = max(1, k // nb)
        cores = [(gx, gy) for gy in range(rows) for gx in range(nb)]
        banks = [gx for (gx, gy) in cores]
        k = len(cores)
    else:
        cores = row_major_cores(grid, k)
        banks = [i % nb for i in range(k)]
    k = len(cores)
    vcs = vc_for(cores, banks)
    ct = [mode, xfer, group, RING] + ttnn.TensorAccessorArgs(w).get_compile_time_args()
    kernels = []
    total_bytes = 0
    risc_noc = [
        (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
    ][:nocs]
    for ri, (risc, noc) in enumerate(risc_noc):
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(cores):
            if mode == 0:
                # each core walks pages i, i+k, i+2k ... over PASSES passes; the RISCs split the passes
                reads_total = (num_pages // k) * PASSES
                reads = reads_total // nocs
                first = i + (ri * reads) % num_pages * 0  # both RISCs start at the same page set (different passes)
                rt[x][y] = [w.buffer_address(), reads, first, k, banks[i], vcs[i]]
                total_bytes += reads * page
            else:
                slab = (num_pages // nb) * page  # bytes of this bank's slab
                readers_this_bank = sum(1 for b in banks if b == banks[i])
                my_rank = sum(1 for j in range(i) if banks[j] == banks[i])
                span = slab // readers_this_bank
                reads_total = (span // xfer) * PASSES
                reads = reads_total // nocs
                first = my_rank * span
                rt[x][y] = [w.buffer_address(), reads, first, xfer, banks[i], vcs[i]]
                total_bytes += reads * xfer
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KPATH,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs(cores),
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc),
            )
        )
    cb = ttnn.CBDescriptor(
        total_size=RING,
        core_ranges=crs(cores),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=RING)],
    )
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])

    def once():
        ttnn.generic_op([w, out], pd)

    once()
    ttnn.synchronize_device(dev)
    prof = RtProfile(dev, lambda: [once() for _ in range(3)])
    ns, ghz = prof.durations(KSUB)
    if len(ns) < 3:
        raise RuntimeError(f"got {len(ns)} matching programs")
    ns, ghz = ns[-3:], ghz[-3:]
    med = statistics.median(ns)
    rec = dict(
        section="dram_ceiling",
        mode=mode,
        dtype=dtype_key,
        xfer=xfer,
        group=group,
        k=k,
        placement=placement,
        nocs=nocs,
        nb=nb,
        total_bytes=total_bytes,
        ns=med,
        ns_all=ns,
        GBps=total_bytes / med,
        ghz=statistics.median(ghz),
        cores=cores[:4],
    )
    append_jsonl("dram_ceiling.jsonl", rec)
    logger.info(
        f"mode={mode} {dtype_key} xfer={xfer} group={group} k={k:3d} {placement:4s} nocs={nocs} -> "
        f"{total_bytes/med:6.1f} GB/s  ({med/1e3:.1f} us, {statistics.median(ghz):.3f} GHz)"
    )
    return rec


def main():
    dev = open_dev()
    try:
        info = env_info(dev)
        logger.info(f"env: {info}")
        with open(os.path.join(RESULTS, "env.json"), "w") as f:
            json.dump(info, f, indent=1)
        nb = info["dram_channels"]
        grid = (info["grid_x"], info["grid_y"])
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        quick = "--quick" in sys.argv
        for dtype_key in ["bf4"] if quick else ["bf4", "bf8"]:
            # one Kimi expert's worth of weights (3 x 7168x2048): 43008 tiles
            w = ttnn.from_torch(
                torch.randn(7168, 2048 * 3),
                dtype=DTYPE[dtype_key],
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            page = TILE_BYTES[dtype_key]
            ks = [8, 11, 22, 44, 88, grid[0] * grid[1]]
            ks = sorted({k for k in ks if k <= grid[0] * grid[1]})
            # (i) interleaved single-page reads, row-major placement
            for group in [16] if quick else [4, 16, 64]:
                for nocs in [1] if quick else [1, 2]:
                    for k in ks:
                        run_config(
                            dev,
                            w,
                            out,
                            info,
                            mode=0,
                            dtype_key=dtype_key,
                            xfer=page,
                            group=group,
                            k=k,
                            placement="row0",
                            nocs=nocs,
                            nb=nb,
                        )
            # (i-b) interleaved reads, bank-adjacent placement (k <= 3*nb)
            if not quick:
                for k in [nb, 2 * nb, 3 * nb]:
                    run_config(
                        dev,
                        w,
                        out,
                        info,
                        mode=0,
                        dtype_key=dtype_key,
                        xfer=page,
                        group=16,
                        k=k,
                        placement="bank",
                        nocs=1,
                        nb=nb,
                    )
            # (ii) bank-direct bursts: tile-sized and 16 KB-class
            big = (16384 // page) * page
            for xfer in [big] if quick else [page, big]:
                for group in [4] if quick else [4, 16]:
                    for k in [nb, 2 * nb, 3 * nb]:
                        for placement in ["bank"] if quick else ["bank", "row0"]:
                            run_config(
                                dev,
                                w,
                                out,
                                info,
                                mode=1,
                                dtype_key=dtype_key,
                                xfer=xfer,
                                group=group,
                                k=k,
                                placement=placement,
                                nocs=1,
                                nb=nb,
                            )
            # (ii-b) the op's band layout: nb columns x rows, every core reads its own column's bank
            if not quick:
                for xfer in [page, 4 * page, big]:
                    for rows in [1, 2, 3, 5, 8, 10]:
                        for group in [4, 16]:
                            run_config(
                                dev,
                                w,
                                out,
                                info,
                                mode=1,
                                dtype_key=dtype_key,
                                xfer=xfer,
                                group=group,
                                k=nb * rows,
                                placement="col",
                                nocs=1,
                                nb=nb,
                            )
                for rows in [5, 10]:
                    run_config(
                        dev,
                        w,
                        out,
                        info,
                        mode=1,
                        dtype_key=dtype_key,
                        xfer=big,
                        group=4,
                        k=nb * rows,
                        placement="col",
                        nocs=2,
                        nb=nb,
                    )
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()

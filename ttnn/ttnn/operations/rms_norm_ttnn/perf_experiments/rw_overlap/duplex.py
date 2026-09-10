# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rw_overlap DUPLEX ROOFLINE PROBE (host half).

The `rw_overlap` idea assumes the op's read stream and write stream *could* run
concurrently and are merely mis-scheduled.  Before spending any effort on CB
depth / granularity / prefetch, this probe measures whether the hardware offers
that concurrency at all.

Three programs over the SAME tensors, the SAME 110-core split and the SAME
per-core page ranges as the shipped rms_norm_ttnn SCHEME_ROWS solve:

    mode="read"    reader kernel only  -- N pages DRAM -> L1
    mode="write"   writer kernel only  -- N pages L1 -> DRAM
    mode="both"    BOTH, with NO circular-buffer handshake between them, so
                   nothing but the hardware can serialize the two streams.

If `both` lands at max(read, write) the machine is full duplex and the op's
46 us "overlap deficit" is a schedulable pipeline bug.  If `both` lands at a
fixed AGGREGATE byte rate, that rate is a DRAM roofline and the deficit is not
recoverable by any amount of pipelining.

NOT a correct program (the writer ships L1 garbage) -- a ceiling probe only.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_IN = 0
CB_OUT = 16

MODES = ("read", "write", "both")


def _core_range_set_full_grid(device):
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def _cores_in(crs):
    return list(ttnn.corerange_to_cores(crs, None, True))


def _core_range_set_n(device, n):
    """`n` cores, row-major (row_wise) prefix of the full grid."""
    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def row_split(device, Rt, num_cores_cap=0):
    """The op's own split: split_work_to_cores(full grid, Rt, row_wise=True)."""
    crs = _core_range_set_full_grid(device) if not num_cores_cap else _core_range_set_n(device, num_cores_cap)
    num_cores, all_cores, g1, g2, rpc1, rpc2 = ttnn.split_work_to_cores(crs, Rt, True)
    out = []
    cursor = 0
    for group, rpc in ((_cores_in(g1), rpc1), (_cores_in(g2), rpc2)):
        for core in group:
            out.append((core, cursor, rpc))
            cursor += rpc
    assert cursor == Rt
    return num_cores, all_cores, out


def create_program_descriptor(x, out, *, mode, block, ring_pages, num_cores_cap=0):
    device = x.device()
    page_bytes = x.buffer_aligned_page_size()
    shape = list(x.shape)
    Wt = shape[-1] // 32
    Rt = x.buffer_num_pages() // Wt

    _, all_cores, assignment = row_split(device, Rt, num_cores_cap)

    read_on = 1 if mode in ("read", "both") else 0
    write_on = 1 if mode in ("write", "both") else 0

    cbs = [
        ttnn.CBDescriptor(
            total_size=ring_pages * page_bytes,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=x.dtype, page_size=page_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=ring_pages * page_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=out.dtype, page_size=page_bytes)
            ],
        ),
    ]

    r_ct = [page_bytes, block, ring_pages, read_on] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    w_ct = [page_bytes, block, ring_pages, write_on] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    r_rt = ttnn.RuntimeArgs()
    w_rt = ttnn.RuntimeArgs()
    in_addr = x.buffer_address()
    out_addr = out.buffer_address()
    for core, row_start, row_count in assignment:
        r_rt[core.x][core.y] = [in_addr, row_start * Wt, row_count * Wt]
        w_rt[core.x][core.y] = [out_addr, row_start * Wt, row_count * Wt]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "duplex_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=r_ct,
            runtime_args=r_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "duplex_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=w_ct,
            runtime_args=w_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def duplex(x, out, *, mode="both", block=72, ring_pages=4, num_cores_cap=0):
    assert mode in MODES
    pd = create_program_descriptor(x, out, mode=mode, block=block, ring_pages=ring_pages, num_cores_cap=num_cores_cap)
    return ttnn.generic_op([x, out], pd)

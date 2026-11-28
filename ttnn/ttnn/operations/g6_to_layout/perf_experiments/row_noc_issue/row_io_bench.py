# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row I/O bench: per-row NoC transactions between an L1 stage and an interleaved ROW_MAJOR tensor.

One kernel (kernels/row_copy.cpp) walks units (tile-row, column block) split contiguously over the
cores. Modes: "write" (writes only, on the writer RISC), "read" (reads only, on the reader RISC),
"check" (copy in -> out on one RISC; the variant under test on one side, the accessor baseline on
the other). See kernels/row_issue.hpp for the issue variants.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

VARIANTS = {
    "accessor": 0,  # baseline: TensorAccessor::get_noc_addr + noc_async_write/read per row
    "cursor": 1,  # stepped bank / slot instead of a divide per row
    "state": 2,  # cursor + stateful cmd buf, coordinate register per row (raw)
    "bank_major": 3,  # public *_one_packet_set_state per bank + *_with_state per row
    "state_bulk": 4,  # state + counters bumped once per unit
    "run_coalesce": 5,  # writes only: run-major stage, one write per bank run (needs full-row units)
    "state_bulk_pace8": 6,  # diagnostics: state_bulk + 8 nops per row
    "state_bulk_pace4": 7,  # diagnostics: state_bulk + 4 nops per row
    "floor48": 8,  # diagnostics: state_bulk, >= 48 cycles between rows
    "floor72": 9,
    "floor96": 10,
}

CB_STAGE = 0
CB_STATS = 1
STATS_WORDS = 16


def split_units(num_units, num_cores):
    cores = min(num_units, num_cores)
    base, rem = divmod(num_units, cores)
    ranges, start = [], 0
    for k in range(cores):
        n = base + (1 if k < rem else 0)
        ranges.append((start, start + n))
        start += n
    return ranges


ROTATE_HOST = 0xFFFF


def num_banks(device, tensor):
    buffer_type = (
        ttnn.BufferType.L1 if tensor.memory_config().buffer_type == ttnn.BufferType.L1 else ttnn.BufferType.DRAM
    )
    return ttnn.get_memory_view(device, buffer_type).num_banks


def spread_start_rows(ranges, row_bytes, block_bytes, nb):
    """Per core, the row of its first unit to start at, so the cores' start banks are spread as
    evenly as the banks allow: cores walk consecutive pages at about the same rate, so a fixed bank
    offset between two cores holds for the whole walk."""
    col_blocks = -(-row_bytes // block_bytes)
    load = [0] * nb
    r0s = []
    for ub, _ in ranges:
        first_page = (ub // col_blocks) * 32
        best = min(range(32), key=lambda r: (load[(first_page + r) % nb], r))
        load[(first_page + best) % nb] += 1
        r0s.append(best)
    return r0s


def unit_count(h, row_bytes, block_bytes):
    return -(-h // 32) * -(-row_bytes // block_bytes)


def build(
    device,
    inp,
    out,
    stats,
    *,
    mode,
    variant,
    block_bytes,
    num_cores,
    rotate=0,
    iters=1,
):
    h, w = inp.shape[-2], inp.shape[-1]
    row_bytes = 2 * w
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    ranges = split_units(unit_count(h, row_bytes, block_bytes), min(num_cores, max_cores))
    coords = [(k % grid.x, k // grid.x) for k in range(len(ranges))]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])

    vid = VARIANTS[variant]
    do_read = mode in ("read", "check")
    do_write = mode in ("write", "check")
    if mode == "read":
        read_variant, write_variant = vid, 0
    elif mode == "write":
        read_variant, write_variant = 0, vid
    else:  # check: the variant on both sides where it exists (run_coalesce is write-only)
        write_variant = vid
        read_variant = 0 if vid == VARIANTS["run_coalesce"] else vid

    r0s = [0] * len(ranges)
    if rotate == "spread":
        rotate = ROTATE_HOST
        r0s = spread_start_rows(ranges, row_bytes, block_bytes, num_banks(device, inp if mode == "read" else out))
    ct = [
        h,
        row_bytes,
        block_bytes,
        int(do_read),
        int(do_write),
        read_variant,
        write_variant,
        rotate,
        iters,
        CB_STAGE,
        CB_STATS,
    ]
    ct += ttnn.TensorAccessorArgs(inp).get_compile_time_args()
    ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
    ct += ttnn.TensorAccessorArgs(stats).get_compile_time_args()

    rt = ttnn.RuntimeArgs()
    for k, ((x, y), (ub, ue)) in enumerate(zip(coords, ranges)):
        rt[x][y] = [ub, ue, k, r0s[k]]

    slot_bytes = 32 * block_bytes
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * slot_bytes,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_STAGE, data_format=ttnn.bfloat16, page_size=slot_bytes)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=64,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_STATS, data_format=ttnn.uint32, page_size=64)],
        ),
    ]
    config = ttnn.ReaderConfigDescriptor() if mode == "read" else ttnn.WriterConfigDescriptor()
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_copy.cpp"),
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[inp.buffer_address(), out.buffer_address(), stats.buffer_address()],
        config=config,
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs), len(ranges)

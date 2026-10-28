# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC: claims-only fetch-and-add stress probe (kernels/faa_probe.cpp)."""
from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
LOG_PAGE = 2048


def build(src, log, total, ret_mode=1, traffic=0, counter_idx=None):
    device = src.device()
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
    N = len(cores)
    ccore = cores[N // 2 if counter_idx is None else counter_idx]
    cphys = device.worker_core_from_logical_core(ccore)
    pb = src.buffer_page_size()
    cbs = [
        ttnn.CBDescriptor(
            total_size=LOG_PAGE,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=LOG_PAGE)],
        ),
        ttnn.CBDescriptor(
            total_size=8 * pb,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=src.dtype, page_size=pb)],
        ),
    ]
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [cphys.x, cphys.y, total, src.buffer_address(), src.buffer_num_pages(), log.buffer_address(), i]
    ct = [0, 1, pb] + list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(log).get_compile_time_args())
    k = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "faa_probe.cpp"),
        core_ranges=all_cores,
        compile_time_args=ct,
        defines=[("RET_MODE", str(ret_mode)), ("TRAFFIC", str(traffic))],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    sems = [
        ttnn.SemaphoreDescriptor(id=0, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=1, core_ranges=all_cores, initial_value=0xFFFFFFFF),
    ]
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=sems, cbs=cbs), N

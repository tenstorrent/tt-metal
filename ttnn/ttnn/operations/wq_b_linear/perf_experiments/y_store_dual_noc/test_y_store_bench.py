# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated y-store bench: 65 active cores, the op's y page mapping, cb_y pre-filled from src (expected y),
then a device-wide sync so all stores start at once.

Variant "s<tenths><order>": the reader (NCRISC/NoC0) stores share/10 of each core's y tiles, the writer
(BRISC/NoC1) the rest; order r=row-major, c=column-major, a=alternate. "base" = writer stores all (op today).
YSB_VARIANTS="base,s5c" selects variants (run order = list order). Output must equal src bit-exactly.
"""

import os
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd

HERE = Path(__file__).parent
ORDERS = {"r": 0, "c": 1, "a": 2}
CB_Y = 16
Y_DONE_SEM = 0
GATHER_SEM = 1
RELEASE_SEM = 2
VARIANTS = os.environ.get("YSB_VARIANTS", "base,s5c").split(",")


def _parse(variant):
    if variant == "base":
        return 0, 0
    return int(variant[1:-1]), ORDERS[variant[-1]]


def _descriptor(device, src, y, share, order):
    tile = y.tile.tile_shape
    tensor_m_tiles = y.shape[-2] // tile[0]
    tensor_n_tiles = y.shape[-1] // tile[1]
    grid = device.compute_with_storage_grid_size()
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    n_groups = pd._split_n(grid, tensor_n_tiles, num_banks)
    assignment = pd._assign_columns(n_groups, tensor_n_tiles, num_banks)
    cores = [c for cs, _ in n_groups for c in cs]
    core_set = pd._core_range_set(cores)
    page = y.buffer_page_size()
    cbs = [
        ttnn.CBDescriptor(
            total_size=tensor_m_tiles * n_groups[0][1] * page,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_Y, data_format=y.dtype, page_size=page)],
        )
    ]
    ct = [CB_Y, tensor_m_tiles, tensor_n_tiles, share, order, Y_DONE_SEM]
    y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    src_ct = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    rect = ttnn.CoreRange(
        ttnn.CoreCoord(min(c.x for c in cores), min(c.y for c in cores)),
        ttnn.CoreCoord(max(c.x for c in cores), max(c.y for c in cores)),
    )
    rect_set = ttnn.CoreRangeSet([rect])
    master = cores[0]
    m_noc = device.worker_core_from_logical_core(master)
    r0 = device.worker_core_from_logical_core(rect.start)
    r1 = device.worker_core_from_logical_core(rect.end)
    w_rt, r_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c in cores:
        a = assignment[(c.x, c.y)]
        w_rt[c.x][c.y] = [y.buffer_address(), *a]
        r_rt[c.x][c.y] = [
            y.buffer_address(),
            *a,
            src.buffer_address(),
            m_noc.x,
            m_noc.y,
            int(c == master),
            r0.x,
            r0.y,
            r1.x,
            r1.y,
        ]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_kernels" / "bench_y_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=ct + y_ct,
        runtime_args=w_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )
    reader = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_kernels" / "bench_y_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=ct + [GATHER_SEM, RELEASE_SEM, len(cores), rect_set.num_cores()] + y_ct + src_ct,
        runtime_args=r_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=rect_set, initial_value=0)
        for i in (Y_DONE_SEM, GATHER_SEM, RELEASE_SEM)
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=sems, cbs=cbs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_y_store(device, variant):
    import torch

    share, order = _parse(variant)
    torch.manual_seed(0)
    ref = torch.randn(1, 1, 160, 4096).bfloat16()
    src = ttnn.from_torch(ref, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    out = ttnn.generic_op([src, y], _descriptor(device, src, y, share, order))
    ttnn.ReadDeviceProfiler(device)
    assert torch.equal(ttnn.to_torch(out), ref)

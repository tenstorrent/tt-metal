# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read interleaved DRAM rows into resident L1 height shards with one final barrier."""

import math
import ttnn

VARIANTS = tuple(f"txn_{n}" for n in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384))

_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef DM_PROFILE_ZONES
#define DM_ZONE(name) DeviceZoneScopedN(name)
#else
#define DM_ZONE(name)
#endif
void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t txn_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    static_assert(txn_bytes <= NOC_MAX_BURST_SIZE);
    static_assert(row_bytes % txn_bytes == 0);
    const auto src = TensorAccessor(TensorAccessorArgs<3>(), get_arg_val<uint32_t>(0));
    const uint32_t dst_base = get_arg_val<uint32_t>(1);
    const uint32_t first_row = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        DM_ZONE("DM_READ_TOTAL");
        {
            DM_ZONE("DM_READ_ISSUE");
            for (uint32_t row = 0; row < rows; ++row) {
                const uint64_t src_row = src.get_noc_addr(first_row + row);
                const uint32_t dst_row = dst_base + row * row_bytes;
                for (uint32_t offset = 0; offset < row_bytes; offset += txn_bytes) {
                    noc_async_read<txn_bytes>(src_row + offset, dst_row + offset, txn_bytes);
                }
            }
        }
        {
            DM_ZONE("DM_READ_BARRIER");
            noc_async_read_barrier();
        }
    }
}
"""


def core_layout(device, cores):
    grid = device.compute_with_storage_grid_size()
    if not 1 <= cores <= grid.x * grid.y:
        raise ValueError("core count exceeds available grid")
    return [(i % grid.x, i // grid.x) for i in range(cores)]


def core_ranges(coords):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])


def assignments(rows, cores):
    q, r = divmod(rows, cores)
    return [(i * q + min(i, r), q + (i < r)) for i in range(cores)]


def output_memory(device, rows, width, cores):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_ranges(core_layout(device, cores)), [math.ceil(rows / cores), width], ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


def create_program_descriptor(input_tensor, output_tensor, *, variant, cores=7, kernel_iters=1, profile_zones=False):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    txn = int(variant.split("_")[1])
    rows, width = list(input_tensor.shape)
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or output_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("row-major tensors required")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.bfloat16:
        raise ValueError("this example uses BF16")
    if rows < cores or kernel_iters < 1 or width * 2 % 64:
        raise ValueError("need >=1 row/core, positive iterations, and 64-byte-aligned rows")
    row_bytes = input_tensor.buffer_aligned_page_size()
    if row_bytes != width * 2 or row_bytes % txn:
        raise ValueError("transaction must divide the stored row size")
    if input_tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("input must be DRAM interleaved")
    if output_tensor.memory_config() != output_memory(input_tensor.device(), rows, width, cores):
        raise ValueError("output must have the expected L1 height-shard layout")
    if list(output_tensor.shape) != [math.ceil(rows / cores) * cores, width]:
        raise ValueError("output must include uniform allocation padding")
    coords = core_layout(input_tensor.device(), cores)
    work = assignments(rows, cores)
    # ReaderConfigDescriptor follows the runtime's architecture-specific read preference.
    runtime = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(coords):
        runtime[x][y] = [input_tensor.buffer_address(), output_tensor.buffer_address(), *work[i]]
    kernel = ttnn.KernelDescriptor(
        defines=[("DM_PROFILE_ZONES", "1")] if profile_zones else [],
        kernel_source=_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges(coords),
        compile_time_args=[row_bytes, txn, kernel_iters]
        + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()),
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel])

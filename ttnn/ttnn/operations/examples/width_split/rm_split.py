# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Width split on Layout::ROW_MAJOR sticks when the op has NO cross-core combine.

The op is tilize: an interleaved DRAM ROW_MAJOR bf16 tensor [H, W] in (one page = one stick of W datums),
an interleaved DRAM TILE tensor out. Nothing is shared between tile-columns, so a width split costs no
combine - only more, smaller stick transfers:

    k = 1 ("row_split", the baseline): whole tile-rows per Tensix core over min(R_t, grid) Tensix cores.
          Each tile-row is 32 reads of 2 * W bytes.
    k > 1 ("width_split"): every tile-row is cut into k column groups of Wt / k tiles, each on its own
          Tensix core: min(R_t, grid // k) row groups x k. Each core reads 32 pieces of 2 * W / k bytes
          per tile-row, so the grid issues k times the read transfers, each k times smaller.

Same reader / compute / writer for every k (only the per-core rectangle changes). Per Tensix core the
column range is tilized in chunks of at most 32 tiles (`chunk_tiles`), double-buffered.
"""

import ttnn

TILE = 32
TILE_BYTES = TILE * TILE * 2
MAX_CHUNK_TILES = 32
CB_RM, CB_OUT = 0, 16


def chunk_tiles(cols):
    """Tiles per tilize block: the largest divisor of a core's column count that is <= MAX_CHUNK_TILES."""
    return max(d for d in range(1, min(cols, MAX_CHUNK_TILES) + 1) if cols % d == 0)


def rm_split_layout(device, rows_t, wt, k):
    """[(core, first tile-row, tile-rows, first tile-column)] for k column groups (k divides Wt)."""
    if k < 1 or wt % k:
        raise ValueError(f"width_split rm: k={k} must divide Wt={wt}")
    grid = device.compute_with_storage_grid_size()
    groups = max(1, min(rows_t, (grid.x * grid.y) // k))
    base, rem = divmod(rows_t, groups)
    cols = wt // k
    out, start, n = [], 0, 0
    for g in range(groups):
        count = base + (1 if g < rem else 0)
        for m in range(k):
            out.append((ttnn.CoreCoord(n % grid.x, n // grid.x), start, count, m * cols))
            n += 1
        start += count
    return out


# Reader CT: 0 stick_bytes (aligned page), 1 kernel_iters, 2 CH, 3 cols (tiles per core), 4.. accessor.
# RT: 0 address, 1 first tile-row, 2 tile-rows, 3 first tile-column.
_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rm = 0;
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t ch = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr auto acc_args = TensorAccessorArgs<4>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);
    const auto acc = TensorAccessor(acc_args, addr, stick_bytes);
    constexpr uint32_t piece = ch * 64;

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = row0; r < row0 + rows; ++r) {
            for (uint32_t c = col0; c < col0 + cols; c += ch) {
                cb_reserve_back(cb_rm, ch);
                uint32_t l1 = get_write_ptr(cb_rm);
                for (uint32_t s = 0; s < 32; ++s) {
                    noc_async_read(acc.get_noc_addr(r * 32 + s, c * 64), l1, piece);
                    l1 += piece;
                }
                noc_async_read_barrier();
                cb_push_back(cb_rm, ch);
            }
        }
    }
}
"""

# Writer CT: 0 kernel_iters, 1 CH, 2 cols, 3 Wt, 4.. accessor. RT: as the reader.
_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
    constexpr uint32_t ch = get_compile_time_arg_val(1);
    constexpr uint32_t cols = get_compile_time_arg_val(2);
    constexpr uint32_t wt = get_compile_time_arg_val(3);
    constexpr auto acc_args = TensorAccessorArgs<4>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);
    const auto acc = TensorAccessor(acc_args, addr, 2048);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = row0; r < row0 + rows; ++r) {
            for (uint32_t c = col0; c < col0 + cols; c += ch) {
                cb_wait_front(cb_out, ch);
                uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < ch; ++i) {
                    noc_async_write(l1, acc.get_noc_addr(r * wt + c + i), 2048);
                    l1 += 2048;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, ch);
            }
        }
    }
}
"""

# Compute CT: 0 CH, 1 kernel_iters, 2 trim (0: the helper's defaults - unpack + pack data-format reconfig before
# the tilize and its uninit after; 1: NoReconfigure + InitOnly - compute_kernel_hw_startup already programmed these
# exact formats and nothing runs after the tilize in this kernel). RT: 0 blocks (tile-rows x chunks).
_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_rm = 0, cb_out = 16;
    constexpr uint32_t ch = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr bool trim = get_compile_time_arg_val(2) != 0;
    const uint32_t blocks = get_arg_val<uint32_t>(0);

    DataflowBuffer rm(cb_rm), out(cb_out);
    compute_kernel_hw_startup(cb_rm, cb_out);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        if constexpr (trim) {
            tilize<ch,
                   cb_rm,
                   cb_out,
                   tilize_config::InitUninitMode::InitOnly,
                   tilize_config::WaitMode::WaitBlock,
                   tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(blocks);
        } else {
            tilize<ch, cb_rm, cb_out>(blocks);
        }
    }
}
"""


def create_rm_split_program_descriptor(input_tensor, output_tensor, *, k, trim=False, kernel_iters=1):
    h, w = list(input_tensor.shape)
    if h % TILE or w % TILE:
        raise ValueError(f"width_split rm: H and W must be multiples of {TILE}, got {[h, w]}")
    device = input_tensor.device()
    rows_t, wt = h // TILE, w // TILE
    layout = rm_split_layout(device, rows_t, wt, k)
    cols = wt // k
    ch = chunk_tiles(cols)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _, _ in layout])

    def cb(index, tiles):
        return ttnn.CBDescriptor(
            total_size=tiles * TILE_BYTES,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            ],
        )

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, row0, rows, col0 in layout:
        reader_rt[core.x][core.y] = [in_addr, row0, rows, col0]
        writer_rt[core.x][core.y] = [out_addr, row0, rows, col0]
        compute_rt[core.x][core.y] = [rows * (cols // ch)]

    reader_ct = [input_tensor.buffer_aligned_page_size(), kernel_iters, ch, cols] + list(
        ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    )
    writer_ct = [kernel_iters, ch, cols, wt] + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    def kernel(src, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core_ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(_READER, reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel(_WRITER, writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
            kernel(_COMPUTE, [ch, kernel_iters, int(trim)], compute_rt, ttnn.ComputeConfigDescriptor()),
        ],
        semaphores=[],
        cbs=[cb(CB_RM, 2 * ch), cb(CB_OUT, 2 * ch)],
    )


def width_split_rm(input_tensor, *, k=1, trim=False, kernel_iters=1):
    """Tilize an interleaved DRAM ROW_MAJOR bf16 [H, W] tensor (H, W multiples of 32) into an interleaved DRAM
    TILE tensor, each tile-row split into k column groups on their own Tensix cores (k = 1: whole tile-rows).
    trim: run the tilize helper with NoReconfigure + InitOnly instead of its defaults (reconfig + uninit)."""
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("width_split rm: input must be ROW_MAJOR bfloat16")
    if input_tensor.memory_config().is_sharded() or input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("width_split rm: input must be DRAM interleaved")
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_rm_split_program_descriptor(
        input_tensor, output_tensor, k=k, trim=trim, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


__all__ = ["chunk_tiles", "create_rm_split_program_descriptor", "rm_split_layout", "width_split_rm"]

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Retile a Layout::TILE tensor from [th_in, 32] tiles to [th_out, 32] tiles: a byte permutation, not math.

Input and output are interleaved DRAM bf16 TILE tensors with the same logical shape [H, W]; only the tile height
differs (th in {1, 2, 4, 8, 16, 32}). A [th, 32] tile is stored as faces of min(th, 16) rows x 16 columns: two
side by side for th <= 16, four (top-left, top-right, bottom-left, bottom-right) for th = 32. One FACE ROW - 16
elements, 32 bytes - is contiguous in every tile shape, and every face row of an output tile is exactly one face
row of some input tile. So a retile is a gather of 32-byte face rows, and consecutive face rows stay consecutive
in both layouts whenever the face geometry allows (16 <-> 32: whole 512-byte faces).

Work unit: U = max(th_in, th_out) rows x CW tile-columns; units split in contiguous ranges over the grid. Every
variant reads the unit's input tiles from DRAM into an L1 staging buffer and writes the unit's output tiles to
DRAM the same way; only the middle differs:

    via_row_major (baseline): un-face the staged tiles into row-major sticks (one 32 B L1-local NoC read per
        face row), then a compute kernel tilizes them into [th_out, 32] tiles - the round trip through
        Layout::ROW_MAJOR that "retile = untilize + tilize" implies.
    face_rows: gather the output tiles straight from the staged input tiles, one 32 B L1-local NoC read per face
        row; no compute kernel.
    runs: the same gather, consecutive face rows merged into one read (whole faces for 16 <-> 32); no compute.
"""

import ttnn

TILE_W = 32
ELEM_BYTES = 2
TILE_HEIGHTS = (1, 2, 4, 8, 16, 32)
VARIANTS = ("via_row_major", "face_rows", "runs")  # baseline first
CB_STAGE, CB_RM, CB_OUT = 0, 1, 16
MAX_CW = 8  # tile-columns per unit


def tile_bytes(th):
    return th * TILE_W * ELEM_BYTES


def unit_shape(th_in, th_out, wt):
    """(U rows, CW tile-columns) of one work unit."""
    cw = max(d for d in range(1, min(wt, MAX_CW) + 1) if wt % d == 0)
    return max(th_in, th_out), cw


def work_split(device, num_units):
    grid = device.compute_with_storage_grid_size()
    n = max(1, min(num_units, grid.x * grid.y))
    base, rem = divmod(num_units, n)
    out, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        out.append((ttnn.CoreCoord(k % grid.x, k // grid.x), start, count))
        start += count
    return out


# Reader. CT: 0 in page bytes, 1 out page bytes, 2 th_in, 3 th_out, 4 CW, 5 Wt, 6 mode (0 via_row_major,
# 1 face_rows, 2 runs), 7 kernel_iters, 8.. input TensorAccessorArgs. RT: 0 address, 1 first unit, 2 units.
_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// byte offset of (tile-local row r, column half h) in a [th, 32] bf16 tile
constexpr uint32_t face_row_offset(uint32_t th, uint32_t r, uint32_t h) {
    const uint32_t face_rows = th < 16 ? th : 16;
    const uint32_t face = th == 32 ? (r / 16) * 2 + h : h;
    const uint32_t fr = th == 32 ? r % 16 : r;
    return face * face_rows * 32 + fr * 32;
}

void kernel_main() {
    constexpr uint32_t cb_stage = 0, cb_rm = 1, cb_out = 16;
    constexpr uint32_t in_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t out_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t th_in = get_compile_time_arg_val(2);
    constexpr uint32_t th_out = get_compile_time_arg_val(3);
    constexpr uint32_t cw = get_compile_time_arg_val(4);
    constexpr uint32_t wt = get_compile_time_arg_val(5);
    constexpr uint32_t mode = get_compile_time_arg_val(6);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(7);
    constexpr auto acc_args = TensorAccessorArgs<8>();
    constexpr uint32_t U = th_in > th_out ? th_in : th_out;
    constexpr uint32_t n_in = (U / th_in) * cw, n_out = (U / th_out) * cw;
    constexpr uint32_t groups = wt / cw;
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    const auto acc = TensorAccessor(acc_args, addr, in_bytes);
    const uint32_t stage = get_write_ptr(cb_stage);  // reader-private staging, never pushed

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t u = u0; u < u0 + units; ++u) {
            const uint32_t row_block = u / groups, c0 = (u % groups) * cw;
            // 1. the unit's input tiles, staged as (a, c): a = input tile-row in the unit, c = tile-column
            for (uint32_t a = 0; a < U / th_in; ++a) {
                for (uint32_t c = 0; c < cw; ++c) {
                    const uint32_t page = (row_block * (U / th_in) + a) * wt + c0 + c;
                    noc_async_read(acc.get_noc_addr(page), stage + (a * cw + c) * in_bytes, in_bytes);
                }
            }
            noc_async_read_barrier();
            if constexpr (mode == 0) {
                // 2a. un-face into row-major sticks: row r of the unit, tile-column c, half h -> r * cw * 64 + c * 64 + h * 32
                cb_reserve_back(cb_rm, n_out);
                const uint32_t rm = get_write_ptr(cb_rm);
                for (uint32_t r = 0; r < U; ++r) {
                    for (uint32_t c = 0; c < cw; ++c) {
                        for (uint32_t h = 0; h < 2; ++h) {
                            const uint32_t src = stage + ((r / th_in) * cw + c) * in_bytes + face_row_offset(th_in, r % th_in, h);
                            noc_async_read(get_noc_addr(src), rm + r * cw * 64 + c * 64 + h * 32, 32);
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_rm, n_out);
            } else {
                // 2b. gather the output tiles in their own byte order, face row by face row
                cb_reserve_back(cb_out, n_out);
                const uint32_t out = get_write_ptr(cb_out);
                constexpr uint32_t faces = th_out == 32 ? 4 : 2, face_rows = th_out < 16 ? th_out : 16;
                // mode 1: one read per face row. mode 2: one read per RUN of `run` face rows - the face rows that
                // are consecutive in both the input and the output face (min of the two faces' heights) - so the
                // loop runs once per run, not once per face row (16 <-> 32: one 512 B read per face).
                constexpr uint32_t in_face_rows = th_in < 16 ? th_in : 16;
                constexpr uint32_t run = mode == 2 ? (in_face_rows < face_rows ? in_face_rows : face_rows) : 1;
                for (uint32_t b = 0; b < U / th_out; ++b) {
                    for (uint32_t c = 0; c < cw; ++c) {
                        const uint32_t dst_tile = out + (b * cw + c) * out_bytes;
                        for (uint32_t f = 0; f < faces; ++f) {
                            for (uint32_t fr = 0; fr < face_rows; fr += run) {
                                const uint32_t ro = th_out == 32 ? (f / 2) * 16 + fr : fr;
                                const uint32_t h = th_out == 32 ? f % 2 : f;
                                const uint32_t r = b * th_out + ro;
                                const uint32_t src = stage + ((r / th_in) * cw + c) * in_bytes + face_row_offset(th_in, r % th_in, h);
                                noc_async_read(get_noc_addr(src), dst_tile + face_row_offset(th_out, ro, h), run * 32);
                            }
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_out, n_out);
            }
        }
    }
}
"""

# Writer. CT: 0 out page bytes, 1 th_in, 2 th_out, 3 CW, 4 Wt, 5 kernel_iters, 6.. output TensorAccessorArgs.
_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t out_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t th_in = get_compile_time_arg_val(1);
    constexpr uint32_t th_out = get_compile_time_arg_val(2);
    constexpr uint32_t cw = get_compile_time_arg_val(3);
    constexpr uint32_t wt = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr auto acc_args = TensorAccessorArgs<6>();
    constexpr uint32_t U = th_in > th_out ? th_in : th_out;
    constexpr uint32_t n_out = (U / th_out) * cw;
    constexpr uint32_t groups = wt / cw;
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    const auto acc = TensorAccessor(acc_args, addr, out_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t u = u0; u < u0 + units; ++u) {
            const uint32_t row_block = u / groups, c0 = (u % groups) * cw;
            cb_wait_front(cb_out, n_out);
            const uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t b = 0; b < U / th_out; ++b) {
                for (uint32_t c = 0; c < cw; ++c) {
                    const uint32_t page = (row_block * (U / th_out) + b) * wt + c0 + c;
                    noc_async_write(l1 + (b * cw + c) * out_bytes, acc.get_noc_addr(page), out_bytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, n_out);
        }
    }
}
"""

# Baseline compute: tilize the un-faced row-major block into [th_out, 32] tiles. CT: 0 CW, 1 kernel_iters.
# RT: 0 blocks (units x U / th_out; one block = th_out rows x CW tiles).
_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_rm = 1, cb_out = 16;
    constexpr uint32_t cw = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    const uint32_t blocks = get_arg_val<uint32_t>(0);
    DataflowBuffer rm(cb_rm), out(cb_out);
    compute_kernel_hw_startup(cb_rm, cb_out);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        tilize<cw, cb_rm, cb_out>(blocks);
    }
}
"""


def _check(input_tensor, th_out):
    if input_tensor.layout != ttnn.TILE_LAYOUT or input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("retile_face_regroup: input must be a bfloat16 TILE tensor")
    if input_tensor.memory_config().is_sharded() or input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("retile_face_regroup: input must be DRAM interleaved")
    th_in, tw = input_tensor.tile.tile_shape
    if tw != TILE_W or th_in not in TILE_HEIGHTS or th_out not in TILE_HEIGHTS:
        raise ValueError(f"retile_face_regroup: tile heights must be in {TILE_HEIGHTS} with width 32")
    h, w = list(input_tensor.shape)
    if h % 32 or w % TILE_W:
        raise ValueError(f"retile_face_regroup: H and W must be multiples of 32, got {[h, w]}")
    return th_in, h, w


def create_program_descriptor(input_tensor, output_tensor, *, variant, kernel_iters=1):
    if variant not in VARIANTS:
        raise ValueError(f"retile_face_regroup: variant must be one of {VARIANTS}, got {variant!r}")
    th_out = int(output_tensor.tile.tile_shape[0])
    th_in, h, w = _check(input_tensor, th_out)
    wt = w // TILE_W
    u_rows, cw = unit_shape(th_in, th_out, wt)
    num_units = (h // u_rows) * (wt // cw)
    split = work_split(input_tensor.device(), num_units)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in split])
    n_in, n_out = (u_rows // th_in) * cw, (u_rows // th_out) * cw
    in_b, out_b = tile_bytes(th_in), tile_bytes(th_out)
    out_tile = ttnn.TileDescriptor(output_tensor.tile)

    def cb(index, pages, page_bytes, tile=None):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page_bytes, tile=tile)
            ],
        )

    cbs = [cb(CB_STAGE, n_in, in_b), cb(CB_OUT, 2 * n_out, out_b, out_tile)]
    mode = VARIANTS.index(variant)
    if mode == 0:
        cbs.append(cb(CB_RM, 2 * n_out, out_b, out_tile))

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, start, count in split:
        reader_rt[core.x][core.y] = [in_addr, start, count]
        writer_rt[core.x][core.y] = [out_addr, start, count]
        compute_rt[core.x][core.y] = [count * (u_rows // th_out)]

    def kernel(src, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core_ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    reader_ct = [in_b, out_b, th_in, th_out, cw, wt, mode, kernel_iters] + list(
        ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    )
    writer_ct = [out_b, th_in, th_out, cw, wt, kernel_iters] + list(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )
    kernels = [
        kernel(_READER, reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
        kernel(_WRITER, writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
    ]
    if mode == 0:
        kernels.append(kernel(_COMPUTE, [cw, kernel_iters], compute_rt, ttnn.ComputeConfigDescriptor()))
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def retile_face_regroup(input_tensor, th_out, *, variant="runs", kernel_iters=1):
    """Return the same [H, W] bf16 data as a TILE tensor with [th_out, 32] tiles (interleaved DRAM).

    variant: "via_row_major" (un-face into sticks + compute tilize; baseline), "face_rows" (gather 32 B face rows
    straight into the output tiles, no compute) or "runs" (the same, contiguous face rows merged into one read)."""
    if kernel_iters < 1:
        raise ValueError(f"retile_face_regroup: kernel_iters must be >= 1, got {kernel_iters}")
    _check(input_tensor, th_out)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape(list(input_tensor.shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Tile([th_out, TILE_W]),
        ),
        input_tensor.device(),
    )
    descriptor = create_program_descriptor(input_tensor, output_tensor, variant=variant, kernel_iters=kernel_iters)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


__all__ = ["TILE_HEIGHTS", "VARIANTS", "create_program_descriptor", "retile_face_regroup", "unit_shape", "work_split"]

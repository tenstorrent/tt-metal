// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_starvation bench — reader (NCRISC / NoC0).
//
// Isolated reconstruction of the op's CURRENT `load_block` plain branch:
// `tilize_kernel::read_sticks_rotated` — one `cb_reserve_back(bw)` /
// `tile_h` rotated stick reads / ONE `noc_async_read_barrier()` /
// `cb_push_back(bw)` cycle per TILE-ROW. Byte-identical to
// `ttnn/ttnn/operations/tilize/kernels/tilize_reader.cpp`'s final `else`.
//
// ONE compile-time knob, `read_barriers`, is added on top of it:
//
//   read_barriers == 1  — the op's current behavior (the honest baseline).
//   read_barriers >  1  — IDEA (b): the tile-row's `tile_h` stick reads are cut
//                         into `read_barriers` runs, each with its own
//                         `noc_async_read_barrier()`, still ONE `cb_push_back`
//                         per tile-row at the end. Same transfers, same sizes,
//                         same destinations, same CB quantum — only the barrier
//                         granularity moves, so it is bit-identical by
//                         construction and needs a perf argument only.
//
// The rotation is preserved in BOTH settings (it is the op's committed Perf-1
// state) and the split runs are still two straight index runs, so no per-stick
// arithmetic is added on either path.
#include "api/dataflow/dataflow_api.h"
#include "bench_zone.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);  // R
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t read_barriers = get_compile_time_arg_val(7);
    constexpr auto in_args = TensorAccessorArgs<8>();

    static_assert(read_barriers >= 1, "read_barriers must be >= 1");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    const auto in_acc = TensorAccessor(in_args, src_addr);

    // A tile-row shorter than the barrier count degenerates to one barrier.
    constexpr uint32_t barriers = (read_barriers > tile_h) ? tile_h : read_barriers;
    constexpr uint32_t rows_per_barrier = tile_h / barriers;
    static_assert(barriers * rows_per_barrier == tile_h, "read_barriers must divide tile_h");

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        const uint32_t byte_offset = w_chunk * block_row_bytes;
        const uint32_t start_page = row_start * tile_h;
        const uint32_t first_row = (tile_h > 1) ? (block_id % tile_h) : 0;

        BenchZone("reader_read_block");
        for (uint32_t blk = 0; blk < block_row_extent; ++blk) {
            const uint32_t page_base = start_page + blk * tile_h;
            cb_reserve_back(cb_input_rows, block_width_tiles);
            const uint32_t l1_base = get_write_ptr(cb_input_rows);

            if constexpr (barriers == 1) {
                // The op's current form: two straight runs behind ONE barrier.
                uint32_t page = page_base + first_row;
                uint32_t l1 = l1_base + first_row * block_row_bytes;
                for (uint32_t row = first_row; row < tile_h; ++row) {
                    noc_async_read(in_acc.get_noc_addr(page, byte_offset), l1, block_row_bytes);
                    ++page;
                    l1 += block_row_bytes;
                }
                page = page_base;
                l1 = l1_base;
                for (uint32_t row = 0; row < first_row; ++row) {
                    noc_async_read(in_acc.get_noc_addr(page, byte_offset), l1, block_row_bytes);
                    ++page;
                    l1 += block_row_bytes;
                }
                noc_async_read_barrier();
            } else {
                // IDEA (b): the SAME rotated sequence, cut into `barriers`
                // equal runs each closed by its own barrier. `row_slot` walks
                // the rotated order; the L1 destination is still derived from
                // the ROW, so the bytes cannot depend on the cut.
                uint32_t row_slot = 0;
                for (uint32_t g = 0; g < barriers; ++g) {
                    for (uint32_t k = 0; k < rows_per_barrier; ++k, ++row_slot) {
                        uint32_t row = first_row + row_slot;
                        if (row >= tile_h) {
                            row -= tile_h;
                        }
                        noc_async_read(
                            in_acc.get_noc_addr(page_base + row, byte_offset),
                            l1_base + row * block_row_bytes,
                            block_row_bytes);
                    }
                    noc_async_read_barrier();
                }
            }
            cb_push_back(cb_input_rows, block_width_tiles);
        }
    }
}

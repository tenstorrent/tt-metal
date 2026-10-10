// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single face unit version of writer_topk_route_finish_tiles.cpp, which documents the split protocol.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "topk_route_finish_gather_common_faces.hpp"

void kernel_main() {
    using namespace topk_route_finish;

    const uint32_t values_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_unit = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t src_addr = get_arg_val<uint32_t>(4);  // TILE bf16 logits
    const uint32_t idx_addr = get_arg_val<uint32_t>(5);  // RM u32 index sticks
    const uint32_t logical_rows = get_arg_val<uint32_t>(6);
    const uint32_t row_tiles_per_batch = get_arg_val<uint32_t>(7);  // R_p / 32
    const uint32_t k_rounded = get_arg_val<uint32_t>(8);

    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(1);  // W_p / 32
    constexpr uint32_t cb_values = get_compile_time_arg_val(2);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stick = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bounce = get_compile_time_arg_val(5);
    constexpr uint32_t value_half_bytes = get_compile_time_arg_val(6);  // 1024
    constexpr uint32_t index_half_bytes = get_compile_time_arg_val(7);  // 1024 (u16) / 2048 (u32)
    constexpr bool index_is_u32 = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t units_per_tile = get_compile_time_arg_val(9);
    constexpr auto values_args = TensorAccessorArgs<10>();
    constexpr auto indices_args = TensorAccessorArgs<decltype(values_args)::next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<decltype(indices_args)::next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<decltype(src_args)::next_compile_time_args_offset()>();

    // Page sizes (2048 or 4096 B index tiles, k_rounded * 4 B sticks) come baked in the host's TensorAccessorArgs.
    const auto values_out = TensorAccessor(values_args, values_addr);
    const auto indices_out = TensorAccessor(indices_args, indices_addr);
    const auto src = TensorAccessor(src_args, src_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    Noc noc;
    DataflowBuffer dfb_values(cb_values);
    DataflowBuffer dfb_indices(cb_indices);
    DataflowBuffer dfb_stick(cb_stick);
    DataflowBuffer dfb_bounce(cb_bounce);

    // Scratch bases stay fixed (nothing pushed); the 64 B bounce slots rely on the allocator's 64 B CB alignment.
    const uint32_t stick_base = dfb_stick.get_write_ptr();
    const uint32_t bounce_base = dfb_bounce.get_write_ptr();
    const CoreLocalMem<uint32_t> stick_dst(stick_base);
    const CoreLocalMem<uint32_t> bounce_dst(bounce_base);
    volatile tt_l1_ptr uint32_t* const stick_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stick_base);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const auto [row_tile, kt, half, col0, ncols] = decode_unit(u, k_tiles, units_per_tile);
        const uint32_t page = row_tile * k_tiles + kt;

        // Same clamps as the reader; this RISC owns rows [8, 16) of the unit.
        const uint32_t batch = row_tile / row_tiles_per_batch;
        const uint32_t row_in_batch0 = (row_tile % row_tiles_per_batch) * 32 + half * half_rows;
        const uint32_t rows_left = row_in_batch0 < logical_rows ? logical_rows - row_in_batch0 : 0;
        const uint32_t valid_rows = rows_left < half_rows ? rows_left : half_rows;
        const uint32_t col_base = kt * tile_width + col0;
        const uint32_t cols_left = col_base < k_rounded ? k_rounded - col_base : 0;
        const uint32_t valid_cols = cols_left < ncols ? cols_left : ncols;
        const uint32_t my_rows = valid_cols == 0 ? 0 : (valid_rows > rows_per_risc ? valid_rows - rows_per_risc : 0);

        // Read before wait_front: this RISC writes only rows [8, 16), which the reader never touches.
        const uint32_t val_base = dfb_values.get_read_ptr();
        const uint32_t idx_out_base = dfb_indices.get_read_ptr();

        // Stick reads for rows [8, 8 + my_rows), indexed from 0 in scratch, overlap the zero fill below.
        for (uint32_t j = 0; j < my_rows; ++j) {
            noc.async_read(
                idx,
                stick_dst,
                valid_cols * 4,
                {.page_id = batch * logical_rows + row_in_batch0 + rows_per_risc + j,
                 .offset_bytes = kt * stick_seg_bytes + col0 * 4},
                {.offset_bytes = j * stick_seg_bytes + col0 * 4});
        }

        // Zero rows [8, 16) of the staging faces this unit covers (the reader zeroes [0, 8)).
        const uint32_t face0 = col0 / 16;
        const uint32_t face1 = face0 + ncols / 16;
        zero_half_rows<2>(val_base, rows_per_risc, half_rows, face0, face1);
        if constexpr (index_is_u32) {
            zero_half_rows<4>(idx_out_base, rows_per_risc, half_rows, face0, face1);
        } else {
            zero_half_rows<2>(idx_out_base, rows_per_risc, half_rows, face0, face1);
        }

        if (my_rows > 0) {
            // Only the stick reads are outstanding: the gather trids drained before the previous unit's writes.
            noc.async_read_barrier();
            gather_unit_rows<index_is_u32>(
                noc,
                src,
                bounce_dst,
                bounce_base,
                stick_l1,
                val_base,
                idx_out_base,
                row_tile,
                width_tiles,
                half,
                rows_per_risc,  // lr_begin: writer owns rows [8, 16)
                my_rows,
                col0,
                col0 + valid_cols);
        }

        // The reader's push means rows [0, 8) and their zero fill are complete.
        dfb_values.wait_front(1);
        dfb_indices.wait_front(1);

        // A face unit writes its 16 columns only; the two faces of a half sit back to back in the tile.
        const uint32_t value_bytes = value_half_bytes * ncols / tile_width;
        const uint32_t index_bytes = index_half_bytes * ncols / tile_width;
        noc.async_write(
            CoreLocalMem<uint32_t>(val_base + face0 * value_bytes),
            values_out,
            value_bytes,
            {.offset_bytes = 0},
            {.page_id = page, .offset_bytes = half * value_half_bytes + face0 * value_bytes});
        noc.async_write(
            CoreLocalMem<uint32_t>(idx_out_base + face0 * index_bytes),
            indices_out,
            index_bytes,
            {.offset_bytes = 0},
            {.page_id = page, .offset_bytes = half * index_half_bytes + face0 * index_bytes});

        // The writes must land before the pops hand the staged pages back to the reader.
        noc.async_write_barrier();
        dfb_values.pop_front(1);
        dfb_indices.pop_front(1);
    }
}

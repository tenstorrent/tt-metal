// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single face unit version of reader_topk_route_finish_gather.cpp, which documents the gather and the face math.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "topk_route_finish_gather_common_faces.hpp"

void kernel_main() {
    using namespace topk_route_finish;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);  // TILE bf16 logits
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);  // RM u32 index sticks
    const uint32_t start_unit = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t logical_rows = get_arg_val<uint32_t>(4);
    const uint32_t row_tiles_per_batch = get_arg_val<uint32_t>(5);  // R_p / 32
    const uint32_t k_rounded = get_arg_val<uint32_t>(6);

    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);      // div_up(k_rounded, 32)
    constexpr uint32_t width_tiles = get_compile_time_arg_val(1);  // W_p / 32
    constexpr uint32_t cb_stick = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bounce = get_compile_time_arg_val(3);
    constexpr uint32_t cb_values = get_compile_time_arg_val(4);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(5);
    constexpr bool index_is_u32 = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t units_per_tile = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto idx_args = TensorAccessorArgs<decltype(src_args)::next_compile_time_args_offset()>();

    // Page sizes (2048 B tiles, k_rounded * 4 B sticks) come baked in the host's TensorAccessorArgs.
    const auto src = TensorAccessor(src_args, src_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    Noc noc;
    DataflowBuffer dfb_stick(cb_stick);
    DataflowBuffer dfb_bounce(cb_bounce);
    DataflowBuffer dfb_values(cb_values);
    DataflowBuffer dfb_indices(cb_indices);

    // Scratch bases stay fixed (nothing pushed); the 64 B bounce slots rely on the allocator's 64 B CB alignment.
    const uint32_t stick_base = dfb_stick.get_write_ptr();
    const uint32_t bounce_base = dfb_bounce.get_write_ptr();
    const CoreLocalMem<uint32_t> stick_dst(stick_base);
    const CoreLocalMem<uint32_t> bounce_dst(bounce_base);
    volatile tt_l1_ptr uint32_t* const stick_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stick_base);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const auto [row_tile, kt, half, col0, ncols] = decode_unit(u, k_tiles, units_per_tile);

        const uint32_t batch = row_tile / row_tiles_per_batch;
        const uint32_t row_in_batch0 = (row_tile % row_tiles_per_batch) * 32 + half * half_rows;

        // Rows past logical_rows are tile padding; all padding units still run to zero their faces.
        const uint32_t rows_left = row_in_batch0 < logical_rows ? logical_rows - row_in_batch0 : 0;
        const uint32_t valid_rows = rows_left < half_rows ? rows_left : half_rows;
        const uint32_t col_base = kt * tile_width + col0;
        const uint32_t cols_left = col_base < k_rounded ? k_rounded - col_base : 0;
        const uint32_t valid_cols = cols_left < ncols ? cols_left : ncols;
        const uint32_t my_rows = valid_cols == 0 ? 0 : (valid_rows < rows_per_risc ? valid_rows : rows_per_risc);

        dfb_values.reserve_back(1);
        dfb_indices.reserve_back(1);
        const uint32_t val_base = dfb_values.get_write_ptr();
        const uint32_t idx_out_base = dfb_indices.get_write_ptr();

        // Stick reads go first so their flight overlaps the zero fill below.
        for (uint32_t lr = 0; lr < my_rows; ++lr) {
            noc.async_read(
                idx,
                stick_dst,
                valid_cols * 4,
                {.page_id = batch * logical_rows + row_in_batch0 + lr, .offset_bytes = kt * stick_seg_bytes + col0 * 4},
                {.offset_bytes = lr * stick_seg_bytes + col0 * 4});
        }

        // The writer zeroes rows [8, 16), so every staging byte written out is zeroed by exactly one RISC.
        const uint32_t face0 = col0 / 16;
        const uint32_t face1 = face0 + ncols / 16;
        zero_half_rows<2>(val_base, 0, rows_per_risc, face0, face1);
        if constexpr (index_is_u32) {
            zero_half_rows<4>(idx_out_base, 0, rows_per_risc, face0, face1);
        } else {
            zero_half_rows<2>(idx_out_base, 0, rows_per_risc, face0, face1);
        }

        if (my_rows > 0) {
            // Only the stick reads are outstanding: the gather trids drained before the previous push.
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
                0,  // lr_begin: reader owns rows [0, 8)
                my_rows,
                col0,
                col0 + valid_cols);
        }

        dfb_values.push_back(1);
        dfb_indices.push_back(1);
    }
}

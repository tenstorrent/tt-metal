// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifndef REDUCE_ROW_SUM_VIA_MM
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#else
#include "ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#endif
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t scaler = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t IN_DF = get_compile_time_arg_val(3);
    constexpr uint32_t LAST_W = get_compile_time_arg_val(4);
    constexpr uint32_t LAST_H = get_compile_time_arg_val(5);
    constexpr uint32_t NEUTRAL_POLICY = get_compile_time_arg_val(6);
    constexpr NeutralPolicy NEUTRAL = static_cast<NeutralPolicy>(NEUTRAL_POLICY);
    constexpr auto tensor_args = TensorAccessorArgs<7>();

    constexpr uint32_t cb_id_in2 = 2;
#ifndef REDUCE_ROW_SUM_VIA_MM
    generate_reduce_scaler(cb_id_in2, scaler);
#else
    generate_mm_scaler(cb_id_in2, scaler);
#endif

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr, tile_bytes);

    // Calculate starting position within the tile grid
    // Tiles are arranged in row-major order: row 0 has tiles 0..Wt-1, row 1 has Wt..2*Wt-1, etc.
    uint32_t start_row = start_id / Wt;
    uint32_t start_col = start_id % Wt;

    uint32_t current_row = start_row;
    uint32_t current_col = start_col;

    // Read tiles and apply padding as needed
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_page(i, tensor_accessor, l1_write_addr);
        noc_async_read_barrier();

        // Apply width padding to the last column of tiles
        if constexpr (LAST_W > 0) {
            if (current_col == Wt - 1) {
                apply_width_padding<IN_DF, LAST_W, NEUTRAL>(l1_write_addr);
            }
        }

        // Apply height padding to the last row of tiles (modulo Ht for batched tensors)
        if constexpr (LAST_H > 0) {
            if ((current_row % Ht) == Ht - 1) {
                apply_height_padding<IN_DF, LAST_H, NEUTRAL>(l1_write_addr);
            }
        }

        cb_push_back(cb_id_in0, onetile);

        // Advance to next tile position
        current_col++;
        if (current_col == Wt) {
            current_col = 0;
            current_row++;
        }
    }
}

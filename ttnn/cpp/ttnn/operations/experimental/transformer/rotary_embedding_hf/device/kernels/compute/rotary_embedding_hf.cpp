// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

ALWI void MUL_TILES(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, uint32_t num_tiles) {
    // Multiply input by cos or sin
    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);

    in0_cb.wait_front(num_tiles);
    in1_cb.wait_front(num_tiles);
    out_cb.reserve_back(num_tiles);

    tile_regs_acquire();
    mul_tiles_init(in0_cb_id, in1_cb_id);
    mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb_id);
    tile_regs_release();
    out_cb.push_back(num_tiles);
    in0_cb.pop_front(num_tiles);
    in1_cb.pop_front(num_tiles);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    CircularBuffer in_cb(in_cb_id);
    CircularBuffer rotated_in_cb(rotated_in_cb_id);
    CircularBuffer cos_cb(cos_cb_id);
    CircularBuffer sin_cb(sin_cb_id);
    CircularBuffer scalar_cb(scalar_cb_id);
    CircularBuffer rotated_in_interm_cb(rotated_in_interm_cb_id);
    CircularBuffer cos_interm_cb(cos_interm_cb_id);
    CircularBuffer sin_interm_cb(sin_interm_cb_id);
    CircularBuffer out_cb(out_cb_id);

    scalar_cb.wait_front(onetile);

    binary_op_init_common(rotated_in_cb_id, scalar_cb_id, rotated_in_interm_cb_id);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                reconfig_data_format(rotated_in_cb_id, scalar_cb_id);
                pack_reconfig_data_format(rotated_in_interm_cb_id);
                rotated_in_cb.wait_front(onetile);
                rotated_in_interm_cb.reserve_back(onetile);
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(rotated_in_cb_id, scalar_cb_id);
                mul_tiles_bcast_scalar(rotated_in_cb_id, scalar_cb_id, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, rotated_in_interm_cb_id);
                tile_regs_release();
                rotated_in_interm_cb.push_back(onetile);
                rotated_in_cb.pop_front(onetile);
                reconfig_data_format_srcb(scalar_cb_id, sin_cb_id);
                pack_reconfig_data_format(rotated_in_interm_cb_id, sin_interm_cb_id);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_interm_cb_id, sin_cb_id, sin_interm_cb_id, onetile);
            } else {
                reconfig_data_format(rotated_in_cb_id, sin_cb_id);
                pack_reconfig_data_format(out_cb_id, sin_interm_cb_id);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_cb_id, sin_cb_id, sin_interm_cb_id, onetile);
            }

            // Multiply input by cos
            MUL_TILES(in_cb_id, cos_cb_id, cos_interm_cb_id, onetile);

            // Add applied sin/cos tensors
            cos_interm_cb.wait_front(onetile);
            sin_interm_cb.wait_front(onetile);
            out_cb.reserve_back(onetile);

            reconfig_data_format_srca(rotated_in_cb_id, cos_interm_cb_id);
            pack_reconfig_data_format(cos_interm_cb_id, out_cb_id);
            tile_regs_acquire();
            add_tiles_init(cos_interm_cb_id, sin_interm_cb_id);
            add_tiles(cos_interm_cb_id, sin_interm_cb_id, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb_id);
            tile_regs_release();

            out_cb.push_back(onetile);
            cos_interm_cb.pop_front(onetile);
            sin_interm_cb.pop_front(onetile);
        }
    }
}

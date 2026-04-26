// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

ALWI void MUL_TILES(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Multiply input by cos or sin
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    ACQ();
    mul_tiles_init(in0_cb, in1_cb);
    mul_tiles(in0_cb, in1_cb, 0, 0, 0);
    pack_tile(0, out_cb);
    REL();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    cb_wait_front(scalar_cb, onetile);

    binary_op_init_common(rotated_in_cb, scalar_cb, rotated_in_interm_cb);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                reconfig_data_format(rotated_in_cb, scalar_cb);
                pack_reconfig_data_format(rotated_in_interm_cb);
                cb_wait_front(rotated_in_cb, onetile);
                cb_reserve_back(rotated_in_interm_cb, onetile);
                ACQ();
                mul_tiles_bcast_scalar_init_short(rotated_in_cb, scalar_cb);
                mul_tiles_bcast_scalar(rotated_in_cb, scalar_cb, 0, 0, 0);
                pack_tile(0, rotated_in_interm_cb);
                REL();
                cb_push_back(rotated_in_interm_cb, onetile);
                cb_pop_front(rotated_in_cb, onetile);
                reconfig_data_format_srcb(scalar_cb, sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_interm_cb, sin_cb, sin_interm_cb, onetile);
            } else {
                reconfig_data_format(rotated_in_cb, sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_cb, sin_cb, sin_interm_cb, onetile);
            }

            // Multiply input by cos
            MUL_TILES(in_cb, cos_cb, cos_interm_cb, onetile);

            // Add applied sin/cos tensors
            cb_wait_front(cos_interm_cb, onetile);
            cb_wait_front(sin_interm_cb, onetile);
            cb_reserve_back(out_cb, onetile);

            reconfig_data_format_srca(rotated_in_cb, cos_interm_cb);
            pack_reconfig_data_format(cos_interm_cb, out_cb);
            ACQ();
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
            pack_tile(0, out_cb);
            REL();

            cb_push_back(out_cb, onetile);
            cb_pop_front(cos_interm_cb, onetile);
            cb_pop_front(sin_interm_cb, onetile);
        }
    }
}

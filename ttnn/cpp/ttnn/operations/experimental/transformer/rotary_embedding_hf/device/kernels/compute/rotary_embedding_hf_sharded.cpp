// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // How many rows (tiles) in n_heads dimension
    constexpr uint32_t half_Wt = Wt / 2;

    binary_op_init_common(in_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    // Fill scalar CB with -1.0
    cb_reserve_back(scalar_cb, onetile);
    cb_push_back(scalar_cb, onetile);
    cb_wait_front(scalar_cb, onetile);

    // Get the sin/cos matrices
    // For decode mode, cos/sin are [1, batch, 1, head_dim] and already in globally allocated CBs
    cb_reserve_back(sin_cb, Wt);
    cb_reserve_back(cos_cb, Wt);

    cb_push_back(sin_cb, Wt);
    cb_push_back(cos_cb, Wt);

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        cb_reserve_back(rotated_in_interm_cb, Wt);
        cb_reserve_back(sin_interm_cb, Wt);
        cb_reserve_back(cos_interm_cb, Wt);
        cb_reserve_back(out_cb, Wt);

        // Get the input
        cb_reserve_back(in_cb, Wt);
        cb_push_back(in_cb, Wt);
        cb_wait_front(in_cb, Wt);

        // Create rotated input: [-x_second_half, x_first_half]
        // First, process first half: just copy (multiplied by 1, which we'll skip)
        // Second half: multiply by -1 scalar

        ACQ();
        for (uint32_t j = 0; j < half_Wt; ++j) {
            // Copy first half tiles to rotated buffer (no multiplication needed yet)
            // These will be used as-is for input*cos later
        }
        REL();

        // Process second half: multiply by -1 and store in rotated buffer
        mul_tiles_bcast_scalar_init_short(in_cb, scalar_cb);
        ACQ();
        for (uint32_t j = 0; j < half_Wt; ++j) {
            // Multiply second half by -1 scalar
            mul_tiles_bcast_scalar(in_cb, scalar_cb, j + half_Wt, 0, j);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        REL();

        // Copy first half to second half of rotated buffer
        ACQ();
        for (uint32_t j = 0; j < half_Wt; ++j) {
            copy_tile_init_with_dt(in_cb);
            copy_tile(in_cb, j, j + half_Wt);
            pack_tile(j + half_Wt, rotated_in_interm_cb, j + half_Wt);
        }
        REL();

        cb_push_back(rotated_in_interm_cb, Wt);
        cb_wait_front(rotated_in_interm_cb, Wt);

        // sin_interim = rotated * sin (broadcast rows)
        mul_bcast_rows_init_short(rotated_in_interm_cb, sin_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles_bcast<BroadcastType::ROW>(rotated_in_interm_cb, sin_cb, j, j, j);
            pack_tile(j, sin_interm_cb, j);
        }
        REL();
        cb_push_back(sin_interm_cb, Wt);
        cb_pop_front(rotated_in_interm_cb, Wt);

        // cos_interim = x * cos (broadcast rows)
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles_bcast<BroadcastType::ROW>(in_cb, cos_cb, j, j, j);
            pack_tile(j, cos_interm_cb, j);
        }
        REL();
        cb_push_back(cos_interm_cb, Wt);
        cb_pop_front(in_cb, Wt);  // Done with input

        // out = cos_interim + sin_interim
        cb_wait_front(sin_interm_cb, Wt);
        cb_wait_front(cos_interm_cb, Wt);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
            pack_tile(j, out_cb, j);
        }
        REL();
        cb_push_back(out_cb, Wt);
        cb_pop_front(sin_interm_cb, Wt);
        cb_pop_front(cos_interm_cb, Wt);
    }

    // Done with the sin/cos matrices, so remove from CB
    cb_pop_front(sin_cb, Wt);
    cb_pop_front(cos_cb, Wt);

    // Done with the scalar, so remove from CB
    cb_pop_front(scalar_cb, onetile);
}

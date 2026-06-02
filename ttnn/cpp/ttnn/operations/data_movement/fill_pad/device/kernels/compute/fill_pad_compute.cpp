// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/where.h"

ALWI void process_masked_tile(uint32_t cb_data_in, uint32_t cb_mask, uint32_t cb_data_out, uint32_t fill_bits) {
    constexpr uint32_t CB_DATA_IN = 0;
    constexpr uint32_t CB_DATA_PADDING = 1;
    constexpr uint32_t CB_MASK = 2;
    constexpr uint32_t CB_OUT = 2;  // reuse CB_MASK tile

    cb_wait_front(cb_data_in, 1);
    cb_reserve_back(cb_data_out, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_data_in);
    copy_tile(cb_data_in, 0, CB_DATA_IN);  // data → DST[0]

    copy_tile_to_dst_init_short(cb_mask);
    copy_tile(cb_mask, 0, CB_MASK);  // mask → DST[2]

    fill_tile_init();
    FILL_PAD_FILL_FN(CB_DATA_PADDING, FILL_PAD_FILL_ARG);  // fill → DST

    where_tile_init();
    where_tile<FILL_PAD_DATA_FMT>(CB_MASK, CB_DATA_PADDING, CB_DATA_IN, CB_OUT);  // if mask then padidng else in -> out

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(CB_OUT, cb_data_out);  // result is at DST[2]
    tile_regs_release();

    cb_pop_front(cb_data_in, 1);
    cb_push_back(cb_data_out, 1);
}

// Corner tile: two sequential where_tile calls give (right_mask OR bot_mask) → fill.
ALWI void process_corner_tile(
    uint32_t cb_data_in, uint32_t cb_right_mask, uint32_t cb_bot_mask, uint32_t cb_data_out, uint32_t fill_bits) {
    constexpr uint32_t CB_DATA_IN = 0;
    constexpr uint32_t CB_DATA_PADDING = 1;
    constexpr uint32_t CB_RIGHT_MASK = 2;
    constexpr uint32_t CB_BOTTOM_MASK = 3;
    constexpr uint32_t CB_OUT = 3;  // reuse CB_MASK tile

    cb_wait_front(cb_data_in, 1);
    cb_reserve_back(cb_data_out, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_data_in);
    copy_tile(cb_data_in, 0, CB_DATA_IN);  // data       → DST[0]

    copy_tile_to_dst_init_short(cb_right_mask);
    copy_tile(cb_right_mask, 0, CB_RIGHT_MASK);  // right_mask → DST[2]

    copy_tile_to_dst_init_short(cb_bot_mask);
    copy_tile(cb_bot_mask, 0, CB_BOTTOM_MASK);  // bot_mask   → DST[3]

    fill_tile_init();
    FILL_PAD_FILL_FN(CB_DATA_PADDING, FILL_PAD_FILL_ARG);  // fill → DST[1]

    where_tile_init();
    // Combine right and bottom mask into corner mask
    where_tile<FILL_PAD_DATA_FMT>(
        CB_RIGHT_MASK, CB_DATA_PADDING, CB_DATA_IN, CB_RIGHT_MASK);  // if mask then padidng else in -> out
    where_tile<FILL_PAD_DATA_FMT>(
        CB_BOTTOM_MASK, CB_DATA_PADDING, CB_RIGHT_MASK, CB_OUT);  // if mask then padidng else in -> out

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(CB_OUT, cb_data_out);  // final result is at DST[3]
    tile_regs_release();

    cb_pop_front(cb_data_in, 1);
    cb_push_back(cb_data_out, 1);
}

void kernel_main() {
    constexpr uint32_t W_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t H_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t has_right_pad = get_compile_time_arg_val(2);
    constexpr uint32_t has_bottom_pad = get_compile_time_arg_val(3);
    constexpr uint32_t elem_size = get_compile_time_arg_val(4);
    constexpr uint32_t fill_bits_ct = get_compile_time_arg_val(5);
    constexpr uint32_t cb_data_in = get_compile_time_arg_val(6);
    constexpr uint32_t cb_right_mask = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bot_mask = get_compile_time_arg_val(8);
    constexpr uint32_t cb_data_out = get_compile_time_arg_val(9);

    // Per-phase tile counts. Phases with num == 0 are skipped. When the
    // corresponding global has_*_pad CT is 0 the host always sets num to 0,
    // so the if constexpr gating below removes the dead code path entirely.
    const uint32_t num_right = get_arg_val<uint32_t>(0);
    const uint32_t num_bottom = get_arg_val<uint32_t>(1);
    const uint32_t num_corner = get_arg_val<uint32_t>(2);

    if (num_right + num_bottom + num_corner == 0) {
        return;
    }

    // Standard init for unary-style SFPU compute with one primary input CB.
    unary_op_init_common(cb_data_in, cb_data_out);

    // Wait for persistent mask tiles pushed once by the writer. They are popped
    // once at cleanup; during the main loop they are reused persistently.
    if constexpr (has_right_pad) {
        cb_wait_front(cb_right_mask, 1);
    }
    if constexpr (has_bottom_pad) {
        cb_wait_front(cb_bot_mask, 1);
    }

    // ---- Main loop: same tile ordering as reader and writer (right/bottom/corner) ----

    if constexpr (has_right_pad) {
        for (uint32_t i = 0; i < num_right; ++i) {
            process_masked_tile(cb_data_in, cb_right_mask, cb_data_out, fill_bits_ct);
        }
    }
    if constexpr (has_bottom_pad) {
        for (uint32_t j = 0; j < num_bottom; ++j) {
            process_masked_tile(cb_data_in, cb_bot_mask, cb_data_out, fill_bits_ct);
        }
    }
    if constexpr (has_right_pad && has_bottom_pad) {
        for (uint32_t k = 0; k < num_corner; ++k) {
            process_corner_tile(cb_data_in, cb_right_mask, cb_bot_mask, cb_data_out, fill_bits_ct);
        }
    }

    // Clean-up
    if constexpr (has_right_pad) {
        cb_pop_front(cb_right_mask, 1);
    }
    if constexpr (has_bottom_pad) {
        cb_pop_front(cb_bot_mask, 1);
    }
}

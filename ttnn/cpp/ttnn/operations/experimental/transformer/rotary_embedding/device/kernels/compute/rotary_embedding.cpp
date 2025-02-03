// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

ALWI void MUL_TILES(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles, uint32_t in1_idx) {
    // Multiply input by cos
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, in1_idx + 1);
    cb_reserve_back(out_cb, num_tiles);

#ifdef DECODE_MODE
    ACQ();
    mul_bcast_rows_init_short(in0_cb, in1_cb);
    mul_tiles_bcast_rows(in0_cb, in1_cb, 0, in1_idx, 0);
    pack_tile(0, out_cb);
    REL();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
// We don't pop in1 in decode which is sin/cos since we don't stream
#else
    ACQ();
    mul_tiles_init();
    mul_tiles(in0_cb, in1_cb, 0, 0, 0);
    pack_tile(0, out_cb);
    REL();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
#endif
}

ALWI void UNTILIZE_TILES(uint32_t in0_cb, uint32_t out_cb, uint32_t num_tiles) {
    untilize_init_short(in0_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    untilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    untilize_uninit(in0_cb);
}

ALWI void TILIZE_ROWS(uint32_t in0_cb, uint32_t sync_cb, uint32_t out_cb, uint32_t num_tiles) {
    tilize_init_short(in0_cb, num_tiles, out_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(sync_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    tilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);

    // Pop shared cbs after tilize
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(sync_cb, num_tiles);
    tilize_uninit(in0_cb, out_cb);
}

namespace NAMESPACE {
void MAIN {
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

    uint32_t updated_cos_cb = cos_cb;
    uint32_t updated_sin_cb = sin_cb;

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_cb = get_compile_time_arg_val(12);
    constexpr uint32_t untilized_cos_sync_cb = get_compile_time_arg_val(13);
    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(15);
    constexpr uint32_t retilized_cos_cb = get_compile_time_arg_val(16);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(17);
    binary_op_init_common(sin_cb, scalar_cb, untilized_sin_cb);
    UNTILIZE_TILES(sin_cb, untilized_sin_cb, Wt);
    UNTILIZE_TILES(cos_cb, untilized_cos_cb, Wt);
    reconfig_data_format_srca(cos_cb, untilized_sin_cb);
    pack_reconfig_data_format(untilized_cos_cb, retilized_sin_cb);
    TILIZE_ROWS(untilized_sin_cb, untilized_sin_sync_cb, retilized_sin_cb, Wt);
    TILIZE_ROWS(untilized_cos_cb, untilized_cos_sync_cb, retilized_cos_cb, Wt);
    updated_cos_cb = retilized_cos_cb;
    updated_sin_cb = retilized_sin_cb;
#else
    binary_op_init_common(rotated_in_cb, scalar_cb, rotated_in_interm_cb);
#endif
    uint32_t in1_idx = 0;
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
#ifdef DECODE_MODE
            in1_idx = j;
#endif
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
                reconfig_data_format_srcb(scalar_cb, updated_sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_interm_cb, updated_sin_cb, sin_interm_cb, onetile, in1_idx);
            } else {
                reconfig_data_format(rotated_in_cb, updated_sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_cb, updated_sin_cb, sin_interm_cb, onetile, in1_idx);
            }

            // Multiply input by cos
            MUL_TILES(in_cb, updated_cos_cb, cos_interm_cb, onetile, in1_idx);

            // Add applied sin/cos tensors
            cb_wait_front(cos_interm_cb, onetile);
            cb_wait_front(sin_interm_cb, onetile);
            cb_reserve_back(out_cb, onetile);

            reconfig_data_format_srca(rotated_in_cb, cos_interm_cb);
            pack_reconfig_data_format(cos_interm_cb, out_cb);
            ACQ();
            add_tiles_init();
            add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
            pack_tile(0, out_cb);
            REL();

            cb_push_back(out_cb, onetile);
            cb_pop_front(cos_interm_cb, onetile);
            cb_pop_front(sin_interm_cb, onetile);
        }
    }
}
}  // namespace NAMESPACE

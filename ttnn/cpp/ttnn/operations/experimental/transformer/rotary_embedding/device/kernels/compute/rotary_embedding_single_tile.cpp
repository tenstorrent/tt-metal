// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for ttnn.experimental.rotary_embedding when head_dim == TILE_WIDTH
// (single tile along the W dimension). Uses an in-L1 transformation matrix to do
// HF-style rotate_half via matmul_tiles, since the inter-tile half-swap used for
// Wt >= 2 cannot express a sub-tile rotation.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

template <uint32_t in0_cb, uint32_t out_cb>
ALWI void UNTILIZE_ONE_TILE() {
    compute_kernel_lib::untilize<
        1,
        in0_cb,
        out_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t in0_cb, uint32_t out_cb>
ALWI void TILIZE_ONE_TILE(uint32_t sync_cb) {
    cb_wait_front(sync_cb, 1);
    compute_kernel_lib::tilize<
        1,
        in0_cb,
        out_cb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    cb_pop_front(sync_cb, 1);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_rows = get_compile_time_arg_val(8);

    uint32_t updated_cos_cb = cos_cb;
    uint32_t updated_sin_cb = sin_cb;

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_cb = get_compile_time_arg_val(9);
    constexpr uint32_t untilized_cos_sync_cb = get_compile_time_arg_val(10);
    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(11);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(12);
    constexpr uint32_t retilized_cos_cb = get_compile_time_arg_val(13);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(14);

    binary_op_init_common(sin_cb, sin_cb, untilized_sin_cb);
    UNTILIZE_ONE_TILE<sin_cb, untilized_sin_cb>();
    UNTILIZE_ONE_TILE<cos_cb, untilized_cos_cb>();
    reconfig_data_format_srca(cos_cb, untilized_sin_cb);
    pack_reconfig_data_format(untilized_cos_cb, retilized_sin_cb);
    TILIZE_ONE_TILE<untilized_sin_cb, retilized_sin_cb>(untilized_sin_sync_cb);
    TILIZE_ONE_TILE<untilized_cos_cb, retilized_cos_cb>(untilized_cos_sync_cb);
    updated_cos_cb = retilized_cos_cb;
    updated_sin_cb = retilized_sin_cb;
#endif

    cb_wait_front(trans_mat_cb, onetile);
    mm_init(in_cb, trans_mat_cb, rotated_in_interm_cb);
    // Binary ops (mul, add) below need their own init path; without this the
    // math-thread register routing stays in matmul mode and mixed-precision
    // binaries (e.g. bf16 x bfp8) produce incorrect results.
    binary_op_init_common(rotated_in_interm_cb, updated_sin_cb, sin_interm_cb);

    for (uint32_t i = 0; i < num_rows; ++i) {
        // rotated = in @ trans_mat  (HF rotate_half on a single 32x32 tile)
        cb_wait_front(in_cb, onetile);
        cb_reserve_back(rotated_in_interm_cb, onetile);
        reconfig_data_format(in_cb, trans_mat_cb);
        pack_reconfig_data_format(rotated_in_interm_cb);
        mm_init_short(in_cb, trans_mat_cb);
        ACQ();
        matmul_tiles(in_cb, trans_mat_cb, 0, 0, 0);
        pack_tile(0, rotated_in_interm_cb);
        REL();
        cb_push_back(rotated_in_interm_cb, onetile);

        // sin_interim = rotated * sin
        cb_wait_front(rotated_in_interm_cb, onetile);
        cb_wait_front(updated_sin_cb, onetile);
        cb_reserve_back(sin_interm_cb, onetile);
        reconfig_data_format(rotated_in_interm_cb, updated_sin_cb);
        pack_reconfig_data_format(sin_interm_cb);
        ACQ();
#ifdef DECODE_MODE
        mul_bcast_rows_init_short(rotated_in_interm_cb, updated_sin_cb);
        mul_tiles_bcast_rows(rotated_in_interm_cb, updated_sin_cb, 0, 0, 0);
#else
        mul_tiles_init(rotated_in_interm_cb, updated_sin_cb);
        mul_tiles(rotated_in_interm_cb, updated_sin_cb, 0, 0, 0);
#endif
        pack_tile(0, sin_interm_cb);
        REL();
        cb_push_back(sin_interm_cb, onetile);
        cb_pop_front(rotated_in_interm_cb, onetile);
#ifndef DECODE_MODE
        cb_pop_front(updated_sin_cb, onetile);
#endif

        // cos_interim = in * cos
        cb_wait_front(updated_cos_cb, onetile);
        cb_reserve_back(cos_interm_cb, onetile);
        reconfig_data_format(in_cb, updated_cos_cb);
        pack_reconfig_data_format(cos_interm_cb);
        ACQ();
#ifdef DECODE_MODE
        mul_bcast_rows_init_short(in_cb, updated_cos_cb);
        mul_tiles_bcast_rows(in_cb, updated_cos_cb, 0, 0, 0);
#else
        mul_tiles_init(in_cb, updated_cos_cb);
        mul_tiles(in_cb, updated_cos_cb, 0, 0, 0);
#endif
        pack_tile(0, cos_interm_cb);
        REL();
        cb_push_back(cos_interm_cb, onetile);
        cb_pop_front(in_cb, onetile);
#ifndef DECODE_MODE
        cb_pop_front(updated_cos_cb, onetile);
#endif

        // out = cos_interim + sin_interim
        cb_wait_front(cos_interm_cb, onetile);
        cb_wait_front(sin_interm_cb, onetile);
        cb_reserve_back(out_cb, onetile);
        reconfig_data_format(cos_interm_cb, sin_interm_cb);
        pack_reconfig_data_format(out_cb);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        ACQ();
        add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
        pack_tile(0, out_cb);
        REL();
        cb_push_back(out_cb, onetile);
        cb_pop_front(cos_interm_cb, onetile);
        cb_pop_front(sin_interm_cb, onetile);
    }
}

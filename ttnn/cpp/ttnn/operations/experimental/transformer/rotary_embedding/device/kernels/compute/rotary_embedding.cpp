// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

ALWI void MUL_TILES(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles, uint32_t in1_idx) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    // Multiply input by cos
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(in1_idx + 1);

    tile_regs_acquire();
#ifdef DECODE_MODE
    mul_bcast_rows_init_short(in0_cb, in1_cb);
    mul_tiles_bcast_rows(in0_cb, in1_cb, 0, in1_idx, 0);
#else
    mul_tiles_init(in0_cb, in1_cb);
    mul_tiles(in0_cb, in1_cb, 0, 0, 0);
#endif
    tile_regs_commit();

    cb_in0.pop_front(num_tiles);
#ifndef DECODE_MODE
    // We don't pop in1 in decode which is sin/cos since we don't stream
    cb_in1.pop_front(num_tiles);
#endif

    cb_out.reserve_back(num_tiles);

    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();

    cb_out.push_back(num_tiles);
}

template <uint32_t num_tiles, uint32_t in0_cb, uint32_t out_cb>
ALWI void UNTILIZE_TILES() {
    compute_kernel_lib::untilize<
        num_tiles,
        in0_cb,
        out_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t num_tiles, uint32_t in0_cb, uint32_t out_cb>
ALWI void TILIZE_ROWS(uint32_t sync_cb) {
    CircularBuffer cb_sync(sync_cb);
    cb_sync.wait_front(num_tiles);
    compute_kernel_lib::tilize<
        num_tiles,
        in0_cb,
        out_cb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    cb_sync.pop_front(num_tiles);
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

    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_rotated_in(rotated_in_cb);
    CircularBuffer cb_scalar(scalar_cb);
    CircularBuffer cb_rotated_in_interm(rotated_in_interm_cb);
    CircularBuffer cb_cos_interm(cos_interm_cb);
    CircularBuffer cb_sin_interm(sin_interm_cb);
    CircularBuffer cb_out(out_cb);

    cb_scalar.wait_front(onetile);

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
    UNTILIZE_TILES<Wt, sin_cb, untilized_sin_cb>();
    UNTILIZE_TILES<Wt, cos_cb, untilized_cos_cb>();
    reconfig_data_format_srca(cos_cb, untilized_sin_cb);
    pack_reconfig_data_format(untilized_cos_cb, retilized_sin_cb);
    TILIZE_ROWS<Wt, untilized_sin_cb, retilized_sin_cb>(untilized_sin_sync_cb);
    TILIZE_ROWS<Wt, untilized_cos_cb, retilized_cos_cb>(untilized_cos_sync_cb);
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
                cb_rotated_in.wait_front(onetile);

                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(rotated_in_cb, scalar_cb);
                mul_tiles_bcast_scalar(rotated_in_cb, scalar_cb, 0, 0, 0);
                tile_regs_commit();

                cb_rotated_in.pop_front(onetile);

                cb_rotated_in_interm.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, rotated_in_interm_cb);
                tile_regs_release();

                cb_rotated_in_interm.push_back(onetile);
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
            cb_cos_interm.wait_front(onetile);
            cb_sin_interm.wait_front(onetile);

            reconfig_data_format_srca(rotated_in_cb, cos_interm_cb);
            pack_reconfig_data_format(cos_interm_cb, out_cb);

            tile_regs_acquire();
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
            tile_regs_commit();

            cb_cos_interm.pop_front(onetile);
            cb_sin_interm.pop_front(onetile);

            cb_out.reserve_back(onetile);

            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();

            cb_out.push_back(onetile);
        }
    }
}

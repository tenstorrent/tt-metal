// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_unary.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t reduce_op = get_compile_time_arg_val(2);
    uint32_t reduce_dim = get_compile_time_arg_val(3);
    uint32_t scaler = get_compile_time_arg_val(4);
    uint32_t src_is_dram = get_compile_time_arg_val(5);
    uint32_t dst_is_dram = get_compile_time_arg_val(6);
    uint32_t num_faces_per_tile = get_compile_time_arg_val(7);
    uint32_t face_h_dim = get_compile_time_arg_val(8);
    uint32_t face_w_dim = get_compile_time_arg_val(9);
    uint32_t output_dtype = get_compile_time_arg_val(10);
    uint32_t use_native_tile_padding = get_compile_time_arg_val(11);
    uint32_t padded_output_tile_height = get_compile_time_arg_val(12);
    uint32_t padded_output_tile_width = get_compile_time_arg_val(13);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_scaler = tt::CB::c_in2;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;

    binary_op_init_common(cb_in0, cb_scaler);
    reduce_init_delta<reduce_op == 0>(cb_in0);
    pack_untilize_init<output_dtype>(cb_out0);

    if constexpr (reduce_op == 0) {
        reduce_init<REDUCE_OP::SUM, REDUCE_DIM::W>(cb_in0);
    } else if constexpr (reduce_op == 1) {
        reduce_init<REDUCE_OP::MAX, REDUCE_DIM::W>(cb_in0);
    } else if constexpr (reduce_op == 2) {
        reduce_init<REDUCE_OP::MIN, REDUCE_DIM::W>(cb_in0);
    }

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        for (uint32_t h = 0; h < per_core_block_tile_cnt; h++) {
            // Acquire input tile
            cb_wait_front(cb_in0, 1);
            
            // Perform reduction operation
            if constexpr (reduce_op == 0) {
                reduce_tile<REDUCE_OP::SUM, REDUCE_DIM::W>(cb_in0, cb_scaler, 0, 0, cb_intermed0, 0);
            } else if constexpr (reduce_op == 1) {
                reduce_tile<REDUCE_OP::MAX, REDUCE_DIM::W>(cb_in0, cb_scaler, 0, 0, cb_intermed0, 0);
            } else if constexpr (reduce_op == 2) {
                reduce_tile<REDUCE_OP::MIN, REDUCE_DIM::W>(cb_in0, cb_scaler, 0, 0, cb_intermed0, 0);
            }
            
            cb_pop_front(cb_in0, 1);
            
            // Apply native tile padding if enabled
            if constexpr (use_native_tile_padding) {
                // Wait for intermediate result
                cb_wait_front(cb_intermed0, 1);
                
                // Reserve output tile with padding
                cb_reserve_back(cb_out0, 1);
                
                // Copy with padding support
                tile_regs_acquire();
                copy_tile_to_dst_init_short_with_dt(cb_intermed0, output_dtype);
                copy_tile(cb_intermed0, 0, 0);
                
                // Apply padding to match target dimensions
                if (padded_output_tile_height > face_h_dim || padded_output_tile_width > face_w_dim) {
                    // Pad the tile to the required dimensions
                    pack_untilize_dst_init_short<output_dtype>(cb_out0, 1, padded_output_tile_height, padded_output_tile_width);
                } else {
                    pack_untilize_dst_init_short<output_dtype>(cb_out0, 1, face_h_dim, face_w_dim);
                }
                
                pack_untilize_dst<output_dtype>(cb_out0, 0);
                tile_regs_release();
                
                cb_push_back(cb_out0, 1);
                cb_pop_front(cb_intermed0, 1);
            } else {
                // Standard path without native padding
                cb_wait_front(cb_intermed0, 1);
                cb_reserve_back(cb_out0, 1);
                
                tile_regs_acquire();
                copy_tile_to_dst_init_short_with_dt(cb_intermed0, output_dtype);
                copy_tile(cb_intermed0, 0, 0);
                pack_untilize_dst_init_short<output_dtype>(cb_out0, 1, face_h_dim, face_w_dim);
                pack_untilize_dst<output_dtype>(cb_out0, 0);
                tile_regs_release();
                
                cb_push_back(cb_out0, 1);
                cb_pop_front(cb_intermed0, 1);
            }
        }
    }
    
    pack_untilize_uninit<output_dtype>();
}
}
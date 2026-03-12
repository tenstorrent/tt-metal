// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/constants.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh_dest.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "api/compute/bcast.h"

#include "rope_sfpu.h"
#include "x2_sum.h"
#include "rms_sum.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_semaphore_id = get_named_compile_time_arg_val("collector_semaphore_id");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t first_physical_x = get_named_compile_time_arg_val("first_physical_x");
    constexpr uint32_t first_physical_y = get_named_compile_time_arg_val("first_physical_y");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto is_collector = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_a_addr = get_arg_val<uint32_t>(argidx++);
    const auto wq_b_addr = get_arg_val<uint32_t>(argidx++);
    const auto q_nope_addr = get_arg_val<uint32_t>(argidx++);
    const auto rope_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto pos = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_r2c_rope = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_c2w_x2 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_x2 = tt::CBIndex::c_6;
    constexpr auto cb_c2c_out_qb = tt::CBIndex::c_7;

    // Constants for MLA WqkvAb
    constexpr uint32_t w_a_k_tiles = 7168 / tt::constants::TILE_WIDTH;
    constexpr uint32_t n_tiles_this_core = 6;
    constexpr uint32_t num_w_a_tiles = w_a_k_tiles * n_tiles_this_core;

    //-------------------------------------------------------------------------
    // W_a reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_a_txns_per_block = 6;
    constexpr uint32_t w_a_tiles_per_txn = 7;
    constexpr uint32_t w_a_tiles_per_block = w_a_tiles_per_txn * w_a_txns_per_block;
    constexpr uint32_t w_a_num_blocks = num_w_a_tiles / w_a_tiles_per_block;

    //-------------------------------------------------------------------------
    // Constants for MLA Wq_b
    constexpr uint32_t wq_b_k_tiles = 49;
    constexpr uint32_t wq_b_n_tiles_this_core = 8;
    constexpr uint32_t num_wq_b_tiles = wq_b_k_tiles * wq_b_n_tiles_this_core;

    //-------------------------------------------------------------------------
    // Wq-b reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t wq_b_txns_per_block = 8;
    constexpr uint32_t wq_b_tiles_per_txn = 7;
    constexpr uint32_t wq_b_tiles_per_block = wq_b_tiles_per_txn * wq_b_txns_per_block;
    constexpr uint32_t wq_b_num_blocks = num_wq_b_tiles / wq_b_tiles_per_block;

    //-------------------------------------------------------------------------
    // Constants for MLA q_nope
    constexpr uint32_t q_nope_heads_per_core = 2;
    constexpr uint32_t q_nope_k_tiles = 4;
    constexpr uint32_t q_nope_n_tiles_per_head = 16;
    constexpr uint32_t q_nope_n_tiles_this_core = q_nope_n_tiles_per_head * q_nope_heads_per_core;
    constexpr uint32_t q_nope_valid_tiles_per_block = q_nope_n_tiles_per_head * q_nope_k_tiles;
    constexpr uint32_t q_nope_num_tiles = q_nope_k_tiles * q_nope_n_tiles_this_core;

    // Each head has 16 tiles in N dimension, so we use ct_dim = 8 and run twice
    constexpr uint32_t q_nope_n_tiles_per_block = 8;
    constexpr uint32_t q_nope_blocks_per_head = q_nope_n_tiles_per_head / q_nope_n_tiles_per_block;

    //-------------------------------------------------------------------------
    // q_nope reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t q_nope_txns_per_block = 5;
    constexpr uint32_t q_nope_tiles_per_txn = 7;
    constexpr uint32_t q_nope_read_tiles_per_block = q_nope_tiles_per_txn * q_nope_txns_per_block;
    constexpr uint32_t q_nope_num_blocks = 1 + (q_nope_num_tiles / q_nope_read_tiles_per_block);  // 1 for ceil

    //-------------------------------------------------------------------------
    // Compute configuration
    //-------------------------------------------------------------------------
    // Pack is configured to Float16_b
    pack_reconfig_data_format(cb_s2c_out);

    // Unpacker B is for input/activation, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for weight, so Bfp8_b
    reconfig_data_format_srca(cb_r2c_w);

    // Initialize matmul: input @ weight -> output
    mm_block_init(
        cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/n_tiles_this_core, /*rt_dim=*/1, /*kt_dim=*/1);

    //-------------------------------------------------------------------------
    // Compute: input @ weight -> output
    //-------------------------------------------------------------------------
    tile_regs_acquire();

    uint32_t in0_index = 0;

    for (uint32_t block_id = 0; block_id < w_a_num_blocks; ++block_id) {
        cb_wait_front(cb_r2c_w, wq_b_tiles_per_block);

        // Process each block in ct_dim chunks, similar to matmul_wo.
        for (uint32_t k = 0; k < w_a_tiles_per_block; k += n_tiles_this_core) {
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/in0_index++,
                /*in1_tile_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/n_tiles_this_core,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        cb_pop_front(cb_r2c_w, wq_b_tiles_per_block);
    }

    //-------------------------------------------------------------------------
    // Apply RMSNorm to q_a (all) and kv_a (only NoPE)
    //-------------------------------------------------------------------------

    uint32_t num_tiles = n_tiles_this_core;

    if (dram_bank_id == 11) {
        // Last core has only 4 tiles
        num_tiles = 4;
    } else if (dram_bank_id < 6) {
        // First 5 cores have only 5 tiles
        num_tiles = 5;
    }

    transpose_wh_dest_init_short</**is_32bit=**/ false>();
    for (uint32_t dst_idx = 0; dst_idx < n_tiles_this_core; dst_idx++) {
        transpose_wh_dest</**is_32bit=**/ false>(dst_idx);
    }

    // Compute sum(X^2) for RMSNorm
    x2_sum_init();
    x2_sum_tile(0, num_tiles);

    // Temporary - to keep output in correct order
    transpose_wh_dest_init_short</**is_32bit=**/ false>();
    for (uint32_t dst_idx = 0; dst_idx < n_tiles_this_core; dst_idx++) {
        transpose_wh_dest</**is_32bit=**/ false>(dst_idx);
    }

    //-------------------------------------------------------------------------
    // Apply RoPE transformation for kv_a RoPE output
    //-------------------------------------------------------------------------
    cb_wait_front(cb_r2c_rope, 1);

    // Configure unpacker A to Float16_b
    reconfig_data_format_srca(cb_r2c_rope);

    if (dram_bank_id == 11) {
        copy_tile_init(cb_r2c_rope);
        copy_tile(cb_r2c_rope, 0, n_tiles_this_core);
        rope_tile_init();

        // This is the first tile with k_pe output
        rope_tile(n_tiles_this_core - 2);
    }

    //-------------------------------------------------------------------------
    tile_regs_commit();

    cb_reserve_back(cb_c2w_x2, 2);
    tile_regs_wait();

    for (uint32_t tile_idx = 0; tile_idx < n_tiles_this_core; tile_idx++) {
        pack_tile</*out_of_order_output=*/true>(tile_idx, cb_s2c_out, tile_idx);
    }
    pack_tile(7, cb_c2w_x2);
    tile_regs_release();
    cb_push_back(cb_c2w_x2, 2);

    //-------------------------------------------------------------------------
    if (is_collector) {
        tile_regs_acquire();
        cb_wait_front(cb_w2c_x2, 2);
        copy_tile_init(cb_w2c_x2);
        copy_tile(cb_w2c_x2, 0, 0);
        copy_tile(cb_w2c_x2, 1, 1);
        cb_pop_front(cb_w2c_x2, 2);

        rms_sum_init();
        rms_sum_tile(0);

        // Divide the sum by N (1536 and 512 respectively)
        binop_with_scalar_tile_init();
        constexpr float q_norm_divisor = 1 / 1536.0f;
        constexpr float kv_norm_divisor = 1 / 512.0f;
        div_unary_tile(0, norm::kernel_util::generic::bit_cast<uint32_t>(q_norm_divisor));
        div_unary_tile(1, norm::kernel_util::generic::bit_cast<uint32_t>(kv_norm_divisor));

        // Take rsqrt of the scale factor
        rsqrt_tile_init</*legacy_compat=*/false>();
        rsqrt_tile</*legacy_compat=*/false>(0);
        rsqrt_tile</*legacy_compat=*/false>(1);

        transpose_wh_dest_init_short</**is_32bit=**/ false>();
        transpose_wh_dest</**is_32bit=**/ false>(0);
        transpose_wh_dest</**is_32bit=**/ false>(1);

        tile_regs_commit();

        cb_reserve_back(cb_c2w_x2, 2);
        tile_regs_wait();
        pack_tile_block(0, cb_c2w_x2, 2);
        tile_regs_release();
        cb_push_back(cb_c2w_x2, 2);
    }

    //-------------------------------------------------------------------------
    // Apply RMSNorm to q_a (all) and kv_a (only NoPE)
    //-------------------------------------------------------------------------
    // Wait for the RMSNorm scale factor to arrive from collector
    cb_wait_front(cb_w2c_x2, 2);

    mul_bcast_cols_init_short(cb_s2c_out, cb_w2c_x2);

    tile_regs_acquire();
    uint32_t scale_idx = (dram_bank_id < 9) ? 0 : 1;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        mul_tiles_bcast_cols(cb_s2c_out, cb_w2c_x2, tile_idx, scale_idx, tile_idx);
    }

    tile_regs_commit();
    cb_pop_front(cb_w2c_x2, 2);

    tile_regs_wait();
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        pack_tile</*out_of_order_output=*/true>(tile_idx, cb_s2c_out, tile_idx);
    }
    tile_regs_release();

    //-------------------------------------------------------------------------
    // Compute: input @ wq_b -> output
    //-------------------------------------------------------------------------
    // Unpacker A is for weight, so Bfp8_b
    reconfig_data_format_srca(cb_r2c_w);

    // Initialize matmul: input @ weight -> output
    mm_block_init(
        cb_s2c_in,
        cb_r2c_w,
        cb_s2c_out,
        /*transpose=*/false,
        /*ct_dim=*/wq_b_n_tiles_this_core,
        /*rt_dim=*/1,
        /*kt_dim=*/1);

    tile_regs_acquire();
    in0_index = 0;

    for (uint32_t block_id = 0; block_id < wq_b_num_blocks; ++block_id) {
        cb_wait_front(cb_r2c_w, wq_b_tiles_per_block);

        for (uint32_t k = 0; k < wq_b_tiles_per_block; k += wq_b_n_tiles_this_core) {
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/in0_index++,
                /*in1_tile_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/wq_b_n_tiles_this_core,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        cb_pop_front(cb_r2c_w, wq_b_tiles_per_block);
    }

    tile_regs_commit();

    cb_reserve_back(cb_c2c_out_qb, wq_b_n_tiles_this_core);
    tile_regs_wait();
    pack_tile_block(0, cb_c2c_out_qb, wq_b_n_tiles_this_core);
    tile_regs_release();
    cb_push_back(cb_c2c_out_qb, wq_b_n_tiles_this_core);

    cb_wait_front(cb_c2c_out_qb, wq_b_n_tiles_this_core);
    //-------------------------------------------------------------------------
    // Apply RoPE transformation for q_pe RoPE output
    //-------------------------------------------------------------------------
    if (dram_bank_id >= 8) {
        // Configure unpacker A to Float16_b
        reconfig_data_format_srca(cb_r2c_rope);

        // Each rope operates on 2 tiles, so we just run it blindly 4 times
        for (uint32_t tile_idx = 0; tile_idx < wq_b_n_tiles_this_core; tile_idx += 2) {
            tile_regs_acquire();
            // Get the rope tensor
            copy_tile_init(cb_r2c_rope);
            copy_tile(cb_r2c_rope, 0, 2);

            // Get the input tensor
            copy_tile_init(cb_c2c_out_qb);
            copy_tile(cb_c2c_out_qb, tile_idx, 0);
            copy_tile(cb_c2c_out_qb, tile_idx + 1, 1);

            // Apply RoPE transformation
            rope_tile_init();
            rope_tile(0);

            tile_regs_commit();

            tile_regs_wait();
            tile_regs_release();
        }
    }
    cb_pop_front(cb_r2c_rope, 1);

    //-------------------------------------------------------------------------
    // Apply q_nope matmul
    //-------------------------------------------------------------------------
    if (dram_bank_id < 8) {
        tile_regs_acquire();

        for (uint32_t head_id = 0; head_id < q_nope_heads_per_core; ++head_id) {
            for (uint32_t block_id = 0; block_id < q_nope_blocks_per_head; ++block_id) {
                in0_index = head_id * q_nope_k_tiles;

                cb_wait_front(cb_r2c_w, wq_b_tiles_per_block);

                for (uint32_t k = 0; k < q_nope_valid_tiles_per_block; k += q_nope_n_tiles_per_block) {
                    matmul_block(
                        cb_c2c_out_qb,
                        cb_r2c_w,
                        /*in0_index=*/in0_index++,
                        /*in1_tile_index=*/k,
                        /*idst=*/0,
                        /*transpose=*/false,
                        /*ct_dim=*/q_nope_n_tiles_per_block,
                        /*rt_dim=*/1,
                        /*kt_dim=*/1);
                }

                cb_pop_front(cb_r2c_w, wq_b_tiles_per_block);
            }
        }

        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }

    cb_pop_front(cb_c2c_out_qb, wq_b_n_tiles_this_core);

    // We have one extra slot reserved, which we won't use, drain it.
    cb_wait_front(cb_r2c_w, wq_b_tiles_per_block);
    cb_pop_front(cb_r2c_w, wq_b_tiles_per_block);
}

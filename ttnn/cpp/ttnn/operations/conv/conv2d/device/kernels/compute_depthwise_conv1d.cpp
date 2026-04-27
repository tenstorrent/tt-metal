// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "internal/mod_div_lib.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

#ifdef FUSE_BIAS
#include "api/compute/bcast.h"
#endif

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define DEBUG_PRINT 0

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

inline void eltwise_mul_and_add_block_v2(
    experimental::CB in0_cb,
    experimental::CB in1_cb,
    experimental::CB mul_partials_cb,
    experimental::CB temp_cb,
    experimental::CB out_cb,
    uint32_t block_num_tiles,
    uint32_t idx,
    uint32_t total_blocks) {
    const uint32_t in0_cb_id = in0_cb.get_cb_id();
    const uint32_t in1_cb_id = in1_cb.get_cb_id();
    const uint32_t mul_partials_cb_id = mul_partials_cb.get_cb_id();
    const uint32_t temp_cb_id = temp_cb.get_cb_id();
    const uint32_t out_cb_id = out_cb.get_cb_id();

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_cb.wait_front(1);
        in0_cb.wait_front(1);
        mul_partials_cb.reserve_back(1);
        mul_tiles_init(in0_cb_id, in1_cb_id);
        ACQ();
        mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
        pack_tile(0, mul_partials_cb_id);
        REL();
        mul_partials_cb.push_back(1);
        in0_cb.pop_front(1);
        in1_cb.pop_front(1);
        if (idx == 0) {
            copy_tile_to_dst_init_short(mul_partials_cb_id);
            ACQ();
            mul_partials_cb.wait_front(1);
            out_cb.reserve_back(1);
            copy_tile(mul_partials_cb_id, 0, 0);
            pack_tile(0, out_cb_id);
            REL();
            out_cb.push_back(1);
            mul_partials_cb.pop_front(1);
        } else {
            add_tiles_init(mul_partials_cb_id, out_cb_id);
            mul_partials_cb.wait_front(1);
            out_cb.wait_front(1);
            ACQ();
            add_tiles(mul_partials_cb_id, out_cb_id, 0, 0, 0);
            pack_tile(0, temp_cb_id);
            REL();
            temp_cb.push_back(1);
            mul_partials_cb.pop_front(1);
            out_cb.pop_front(1);

            copy_tile_to_dst_init_short(temp_cb_id);
            ACQ();
            temp_cb.wait_front(1);
            out_cb.reserve_back(1);
            copy_tile(temp_cb_id, 0, 0);
            pack_tile(0, out_cb_id);
            REL();
            out_cb.push_back(1);
            temp_cb.pop_front(1);
        }
    }
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t reader_num_h_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(5);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(6);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(7);  // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(11);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13);  // out_subblock_h * out_subblock_w;
    constexpr bool tilize_in0 = get_compile_time_arg_val(14);
    constexpr bool untilize_out = get_compile_time_arg_val(15);

    constexpr uint32_t out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t out_block_w = in1_block_w;

    // CB indices
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in0_pretilize_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t in0_cb_second_reader_id = get_compile_time_arg_val(21);
    constexpr uint32_t eltwise_mul_partials_cb = get_compile_time_arg_val(22);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t temp_sum_cb = get_compile_time_arg_val(25);

    constexpr uint32_t in0_num_subblocks_read = in0_num_subblocks;

    constexpr uint32_t num_blocks = in0_num_blocks_h * in0_num_blocks_w;  // num_tokens window

    experimental::CB cb_tilized_in0(tilized_in0_cb_id);
    experimental::CB cb_in1(in1_cb_id);
    experimental::CB cb_mul_partials(eltwise_mul_partials_cb);
    experimental::CB cb_temp_sum(temp_sum_cb);
    experimental::CB cb_out(out_cb_id);

    binary_op_init_common(in0_cb_id, in1_cb_id, out_cb_id);

    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
            uint32_t i = in0_block_h_i * in0_num_blocks_w + in0_block_w_i;
            if constexpr (tilize_in0) {
                compute_kernel_lib::tilize<in0_block_w, in0_cb_id, tilized_in0_cb_id>(in0_num_subblocks_read);
            }
            reconfig_data_format_srca(tilized_in0_cb_id);
            pack_reconfig_data_format(eltwise_mul_partials_cb);
            eltwise_mul_and_add_block_v2(
                cb_tilized_in0, cb_in1, cb_mul_partials, cb_temp_sum, cb_out, in0_block_num_tiles, i, num_blocks);

        }  // for in0_num_blocks_h
    }  // for in0_num_blocks_w
}  // void kernel_main()

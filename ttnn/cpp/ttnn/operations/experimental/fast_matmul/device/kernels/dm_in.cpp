// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;

    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    // DPRINT <<"M_block_tiles: " << M_block_tiles << ", K_block_tiles: " << K_block_tiles << ", N_block_tiles: " <<
    // N_block_tiles << ENDL(); DPRINT <<"M_blocks_per_core: " << M_blocks_per_core << ", N_blocks_per_core: " <<
    // N_blocks_per_core << ENDL(); DPRINT <<"K_num_blocks: " << K_num_blocks << ENDL();

    // DPRINT << "in0_addr " << get_read_ptr(cb_id_in0) << " in1_addr " << get_read_ptr(cb_id_in1) << " out_addr " <<
    // get_read_ptr(cb_id_out) << ENDL();

    // DPRINT <<"in0 tile size "<<get_tile_size(cb_id_in0)<<" in1 tile size "<<get_tile_size(cb_id_in1)<<" out tile size
    // "<<get_tile_size(cb_id_out)<<ENDL();
    cb_reserve_back(cb_id_in0, M_block_tiles * K_block_tiles * M_blocks_per_core * K_num_blocks);
    cb_reserve_back(cb_id_in1, K_block_tiles * N_block_tiles * K_num_blocks * N_blocks_per_core);

    cb_push_back(cb_id_in0, M_block_tiles * K_block_tiles * M_blocks_per_core * K_num_blocks);
    cb_push_back(cb_id_in1, K_block_tiles * N_block_tiles * K_num_blocks * N_blocks_per_core);
}

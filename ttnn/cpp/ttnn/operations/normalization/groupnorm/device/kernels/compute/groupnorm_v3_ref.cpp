// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWSUB
#define BCAST_DIM BroadcastType::SCALAR

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint_tile.h"
#include "debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    // CB indices
    // ----------
    constexpr uint32_t src_cb_idx = tt::CBIndex::c_0;
    constexpr uint32_t dst_cb_idx = tt::CBIndex::c_1;
    constexpr uint32_t sum_cb_idx = tt::CBIndex::c_2;
    constexpr uint32_t mean_cb_idx = tt::CBIndex::c_3;
    constexpr uint32_t varsum_cb_idx = tt::CBIndex::c_4;
    constexpr uint32_t variance_cb_idx = tt::CBIndex::c_5;
    constexpr uint32_t sum_scaler_cb_idx = tt::CBIndex::c_6;
    constexpr uint32_t mean_scaler_cb_idx = tt::CBIndex::c_7;

    // Compile time args
    // -----------------
    constexpr uint32_t src_tiles_per_page = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_batches = get_compile_time_arg_val(2);
    constexpr uint32_t dst_tiles_per_page = get_compile_time_arg_val(3);

    // Destination regs
    // ----------------
    constexpr uint32_t mean_dst_reg = 4;

    //-------------------------------------------------------------------------
    // Sum
    binary_op_init_common(src_cb_idx, sum_scaler_cb_idx, /*not_used*/ sum_cb_idx);
    reduce_init(src_cb_idx, sum_scaler_cb_idx, /*not_used*/ sum_cb_idx);

    cb_wait_front(sum_scaler_cb_idx, 1);
    cb_reserve_back(sum_cb_idx, 1);

    tile_regs_acquire();  // compute thread acquires the tile registers
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {  // num_pages = num groups here
            cb_wait_front(src_cb_idx, src_tiles_per_page);

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {  // iterating through each channel in the group
                reduce_tile(
                    src_cb_idx,
                    sum_scaler_cb_idx,
                    tile,
                    0,
                    mean_dst_reg);  // every element is multiplied by sum_scaler in a reduce operation (in this case
                                    // everything is multiplied by 1.0f)
            }

            cb_pop_front(src_cb_idx, src_tiles_per_page);
        }
    }
    dprint_tensix_dest_reg(mean_dst_reg);
    tile_regs_commit();  // compute thread releases the tile registers

    tile_regs_wait();  // packing thread acquires the tile registers
    pack_tile(mean_dst_reg, sum_cb_idx);
    tile_regs_release();  // packing thread releases the tile registers

    cb_push_back(sum_cb_idx, 1);

    reduce_uninit();

    // Make it mean
    mul_tiles_init(sum_cb_idx, mean_scaler_cb_idx);  // multiply by 1/N to get the mean

    cb_wait_front(sum_cb_idx, 1);
    cb_wait_front(mean_scaler_cb_idx, 1);
    cb_reserve_back(mean_cb_idx, 1);

    tile_regs_acquire();
    mul_tiles(sum_cb_idx, mean_scaler_cb_idx, 0, 0, 0);
    dprint_tensix_dest_reg(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, mean_cb_idx);
    tile_regs_release();

    cb_pop_front(sum_cb_idx, 1);
    cb_pop_front(mean_scaler_cb_idx, 1);
    cb_push_back(mean_cb_idx, 1);

    //-------------------------------------------------------------------------
    // xmm
    sub_tiles_bcast_scalar_init_short(src_cb_idx, mean_cb_idx);
    square_tile_init();
    add_binary_tile_init();
    // reduce_init(src_cb_idx, sum_scaler_cb_idx, /*not_used*/ varsum_cb_idx);

    cb_wait_front(mean_cb_idx, 1);
    cb_reserve_back(varsum_cb_idx, 1);

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            cb_wait_front(src_cb_idx, src_tiles_per_page);

            tile_regs_acquire();
            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                sub_tiles_bcast_scalar(src_cb_idx, mean_cb_idx, tile, 0, /*dst_reg_idx=*/tile);
                // dprint_tensix_dest_reg(tile);
                // Now square the values in the dst registers
                square_tile(tile);
                // dprint_tensix_dest_reg(tile);
                // reduce_tile(src_cb_idx, sum_scaler_cb_idx, tile, 0, mean_dst_reg);
                // dprint_tensix_dest_reg(mean_dst_reg);
            }
            add_binary_tile(0, 1);
            add_binary_tile(2, 3);
            add_binary_tile(0, 2);

            tile_regs_commit();

            tile_regs_wait();
            tile_regs_release();

            cb_pop_front(src_cb_idx, src_tiles_per_page);

            // TODO: Ping pong between sum and varsum CBs to store and pick up the result
            // TODO: Call binary op 4 times to get the sum of the var
            // TODO: Call reduce on the tile at the end to get a single scalar value
            // TODO: Inverse and multiply by gamma and add beta to get final value

            // cb_reserve_back(var_cb_idx, 1);

            // tile_regs_wait();
            // pack_tile(mean_dst_reg, var_cb_idx);
            // tile_regs_release();
            // cb_push_back(var_cb_idx, 1);
        }
    }
    cb_pop_front(mean_cb_idx, 1);
    cb_pop_front(sum_scaler_cb_idx, 1);
    cb_push_back(varsum_cb_idx, 1);

    // reduce_uninit();

    //-------------------------------------------------------------------------
    // Output
    copy_tile_init(src_cb_idx);
    square_tile_init();

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            cb_wait_front(src_cb_idx, src_tiles_per_page);
            cb_reserve_back(dst_cb_idx, dst_tiles_per_page);

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                copy_tile(src_cb_idx, tile, tile);
            }

            cb_pop_front(src_cb_idx, src_tiles_per_page);
            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                pack_tile(/*dst_reg_idx=*/tile, dst_cb_idx, /*output_tile_index=*/tile);
            }
            cb_push_back(dst_cb_idx, dst_tiles_per_page);
        }
    }
}
}  // namespace NAMESPACE

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::AVG
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWSUB
#define BCAST_DIM BroadcastType::SCALAR

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    // -----------------
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t src_tiles_per_page = get_compile_time_arg_val(1);
    constexpr uint32_t num_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_batches = get_compile_time_arg_val(3);
    constexpr uint32_t mean_cb_idx = get_compile_time_arg_val(4);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(5);
    constexpr uint32_t dst_tiles_per_page = get_compile_time_arg_val(6);
    constexpr uint32_t scaler_cb_idx = get_compile_time_arg_val(7);

    constexpr uint32_t mean_dst_reg = 0;

    //-------------------------------------------------------------------------
    // Mean
    binary_op_init_common(src_cb_idx, scaler_cb_idx, /*not_used*/ mean_cb_idx);
    reduce_init(src_cb_idx, scaler_cb_idx, /*not_used*/ mean_cb_idx);

    cb_wait_front(scaler_cb_idx, 1);
    cb_reserve_back(mean_cb_idx, 1);

    tile_regs_acquire();
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            cb_wait_front(src_cb_idx, src_tiles_per_page);

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                reduce_tile(src_cb_idx, scaler_cb_idx, tile, 0, mean_dst_reg);
            }

            cb_pop_front(src_cb_idx, src_tiles_per_page);
        }
    }
    cb_pop_front(scaler_cb_idx, 1);

    dprint_tensix_dest_reg(mean_dst_reg);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(mean_dst_reg, mean_cb_idx);
    tile_regs_release();
    cb_push_back(mean_cb_idx, 1);
    reduce_uninit();

    //-------------------------------------------------------------------------
    // xmm
    cb_wait_front(mean_cb_idx, 1);

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            cb_wait_front(src_cb_idx, src_tiles_per_page);
            sub_tiles_bcast_scalar_init_short(src_cb_idx, mean_cb_idx);

            tile_regs_acquire();

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                sub_tiles_bcast_scalar(src_cb_idx, mean_cb_idx, tile, 0, /*dst_reg_idx=*/tile);
            }

            tile_regs_commit();

            // TODO: Read this out to output CB as this is the partial result (numerator)
            for (uint32_t tile = 0; tile < dst_tiles_per_page; ++tile) {
                pack_tile(/*dst_reg_idx=*/tile, dst_cb_idx, /*output_tile_index=*/tile);
            }

            // (x - E[x])^2
            square_tile_init();
            tile_regs_acquire();
            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                square_tile(tile);
            }
            tile_regs_commit();

            // Sum of (x - E[x])^2
            reduce_init(src_cb_idx, scaler_cb_idx, /*not_used*/ mean_cb_idx);
            tile_regs_acquire();

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                reduce_tile(src_cb_idx, scaler_cb_idx, tile, 0, mean_dst_reg);
            }
            tile_regs_commit();
            reduce_uninit();

            // TODO: Maybe keep only 4 regs and use the 8th reg to store the sum of the var?
            // TODO: Otherwise make a CB to store the var and do an add call in each iteration
            // TODO: Confirm whether the (x - E[x])^2 will be retained in the DST register if commit, wait and release
            // and then commit is called.
            cb_pop_front(src_cb_idx, src_tiles_per_page);
            cb_reserve_back(var_cb_idx, 1);

            tile_regs_wait();
            pack_tile(mean_dst_reg, var_cb_idx);
            tile_regs_release();
            cb_push_back(var_cb_idx, 1);
        }
    }
    cb_pop_front(mean_cb_idx, 1);

    DPRINT << "src_tiles_per_page: " << src_tiles_per_page << ENDL();

    // Output
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            cb_wait_front(src_cb_idx, src_tiles_per_page);
            // cb_reserve_back(dst_cb_idx, dst_tiles_per_page);

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                // TODO: Implement
            }

            cb_pop_front(src_cb_idx, src_tiles_per_page);
            // cb_push_back(dst_cb_idx, dst_tiles_per_page);
        }
    }
}
}  // namespace NAMESPACE

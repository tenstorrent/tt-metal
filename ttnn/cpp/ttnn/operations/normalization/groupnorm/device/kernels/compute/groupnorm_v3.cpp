// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#define REDUCE_OP PoolType::AVG
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWSUB
#define BCAST_DIM BroadcastType::SCALAR

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack.h"

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

    constexpr uint32_t mean_dst_reg = 0;
    constexpr uint32_t var_dst_reg = 1;

    // Read this 3 times for mean, var, and output
    constexpr uint32_t num_reads = 3;

    constexpr bool fp32_transpose = true;

    binary_op_init_common(src_cb_idx, mean_cb_idx, /*not_used*/ mean_cb_idx);
    reconfig_data_format(src_cb_idx, mean_cb_idx);

    // Mean
    reduce_init(src_cb_idx, src_cb_idx, /*not_used*/ src_cb_idx);
    cb_reserve_back(mean_cb_idx, 1);
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            cb_wait_front(src_cb_idx, src_tiles_per_page);
            tile_regs_acquire();

            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                // TODO: Implement
                reduce_tile(src_cb_idx, src_cb_idx, tile, tile, mean_dst_reg);
            }

            cb_pop_front(src_cb_idx, src_tiles_per_page);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(mean_dst_reg, mean_cb_idx);
            tile_regs_release();
        }
    }
    cb_push_back(mean_cb_idx, 1);
    reduce_uninit();

    // xmm
    cb_wait_front(mean_cb_idx, 1);
    sub_tiles_bcast_scalar_init_short(src_cb_idx, mean_cb_idx);
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        for (uint32_t page = 0; page < num_pages; ++page) {
            // tile_regs_acquire();
            cb_wait_front(src_cb_idx, src_tiles_per_page);
            for (uint32_t tile = 0; tile < src_tiles_per_page; ++tile) {
                //     // TODO: Implement
                // sub_tiles_bcast_scalar(src_cb_idx, mean_cb_idx, tile, 0, mean_dst_reg);
            }

            cb_pop_front(src_cb_idx, src_tiles_per_page);
            // tile_regs_commit();

            cb_reserve_back(dst_cb_idx, dst_tiles_per_page);
            // tile_regs_wait();
            for (uint32_t tile = 0; tile < dst_tiles_per_page; ++tile) {
                pack_tile(mean_dst_reg, dst_cb_idx, tile);
            }
            // tile_regs_release();
            cb_push_back(dst_cb_idx, dst_tiles_per_page);
        }
    }
    cb_pop_front(mean_cb_idx, 1);

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

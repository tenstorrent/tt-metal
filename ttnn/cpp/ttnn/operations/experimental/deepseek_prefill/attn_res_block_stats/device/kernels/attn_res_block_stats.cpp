// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Both statistics of a candidate's token row-tile, into one accumulator:
//
//   dst  = sum_k v_k @ Q_k           the dots, in Q's columns 1..R; column 0 of Q is zero
//   dst += sum_k (v_k * v_k) @ E     the sum of squares, E being ones in column 0
//
// The squares have to reach a circular buffer before a matmul can read them, so they are
// produced first, a row at a time, and the two matmul runs then share one dest session.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/attn_res_block_stats_items.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t kSplitArgs = 1;

    constexpr auto cb_v = tt::CBIndex::c_0;
    constexpr auto cb_queries = tt::CBIndex::c_1;
    constexpr auto cb_ones = tt::CBIndex::c_2;
    constexpr auto cb_squares = tt::CBIndex::c_3;
    constexpr auto cb_out = tt::CBIndex::c_16;

    CircularBuffer cb_v_obj(cb_v);
    CircularBuffer cb_queries_obj(cb_queries);
    CircularBuffer cb_ones_obj(cb_ones);
    CircularBuffer cb_squares_obj(cb_squares);
    CircularBuffer cb_out_obj(cb_out);

    compute_kernel_hw_startup(cb_v, cb_v, cb_squares);
    cb_queries_obj.wait_front(Wt);
    cb_ones_obj.wait_front(1);

    for_each_attn_res_block_stats_item<kSplitArgs>([&](uint32_t) {
        cb_v_obj.wait_front(Wt);

        cb_squares_obj.reserve_back(Wt);
        mul_init(cb_v, cb_v);
        pack_reconfig_data_format(cb_squares);
        for (uint32_t k = 0; k < Wt; ++k) {
            tile_regs_acquire();
            mul_tiles(cb_v, cb_v, k, k, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_squares, k);
            tile_regs_release();
        }
        cb_squares_obj.push_back(Wt);
        cb_squares_obj.wait_front(Wt);

        cb_out_obj.reserve_back(1);
        tile_regs_acquire();
        matmul_init(cb_v, cb_queries);
        for (uint32_t k = 0; k < Wt; ++k) {
            matmul_tiles(cb_v, cb_queries, k, k, 0);
        }
        matmul_init(cb_squares, cb_ones);
        for (uint32_t k = 0; k < Wt; ++k) {
            matmul_tiles(cb_squares, cb_ones, k, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_out_obj.push_back(1);

        cb_squares_obj.pop_front(Wt);
        cb_v_obj.pop_front(Wt);
    });
}

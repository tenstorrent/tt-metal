// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t arg_index = 0;
    uint32_t start_n = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_c = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_t = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_th = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_tw = get_arg_val<uint32_t>(arg_index++);
    uint32_t num_tiles = get_arg_val<uint32_t>(arg_index++);
    uint32_t n_stride = get_arg_val<uint32_t>(arg_index++);
    uint32_t c_stride = get_arg_val<uint32_t>(arg_index++);
    uint32_t N = get_arg_val<uint32_t>(arg_index++);
    uint32_t C = get_arg_val<uint32_t>(arg_index++);
    uint32_t Ht = get_arg_val<uint32_t>(arg_index++);
    uint32_t Wt = get_arg_val<uint32_t>(arg_index++);

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto cb_id_dst = get_compile_time_arg_val(1);
    unary_bcast_init<BroadcastType::SCALAR>(cb_id_src, cb_id_dst);

    uint32_t HtWt = Ht * Wt;
    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            cb_wait_front(cb_id_src, 1);
            tile_regs_acquire();
            unary_bcast<BroadcastType::SCALAR>(cb_id_src, 0, 0);
            tile_regs_commit();

            cb_pop_front(cb_id_src, 1);
            cb_reserve_back(cb_id_dst, 1);
            tile_regs_wait();
            pack_tile(0, cb_id_dst);

            cb_push_back(cb_id_dst, 1);
            tile_regs_release();
            num_tiles_read += HtWt - start_t;
        }
    }
}
}  // namespace NAMESPACE

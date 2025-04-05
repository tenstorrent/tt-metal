// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t start_n = get_arg_val<uint32_t>(0);
    uint32_t start_c = get_arg_val<uint32_t>(1);
    uint32_t start_t = get_arg_val<uint32_t>(2);
    uint32_t start_th = get_arg_val<uint32_t>(3);
    uint32_t start_tw = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t n_stride = get_arg_val<uint32_t>(6);
    uint32_t c_stride = get_arg_val<uint32_t>(7);
    uint32_t N = get_arg_val<uint32_t>(8);
    uint32_t C = get_arg_val<uint32_t>(9);
    uint32_t Ht = get_arg_val<uint32_t>(10);
    uint32_t Wt = get_arg_val<uint32_t>(11);

    unary_bcast_init<BroadcastType::ROW>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    uint32_t HtWt = Ht * Wt;
    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;
    uint32_t next_channel_shift = c_stride - HtWt;

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw) {
                    cb_wait_front(tt::CBIndex::c_0, 1);
                    tile_regs_acquire();
                    unary_bcast<BroadcastType::ROW>(tt::CBIndex::c_0, 0, 0);
                    tile_regs_commit();

                    cb_pop_front(tt::CBIndex::c_0, 1);
                    cb_reserve_back(tt::CBIndex::c_16, 1);
                    tile_regs_wait();
                    pack_tile(0, tt::CBIndex::c_16);

                    cb_push_back(tt::CBIndex::c_16, 1);
                    tile_regs_release();
                    ++num_tiles_read;
                }
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
}  // namespace NAMESPACE

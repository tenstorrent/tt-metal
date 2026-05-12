// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "tools/profiler/kernel_profiler.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

// Migrated stages: per-tile UnaryBcast<Row> + PackTile body.
// Skipped stages: irregular outer (n, c, th, tw) iteration with `num_tiles_read < num_tiles` early-exit
// stays raw — the chain helper iterates a single linear count and cannot express the multi-axis early exit.
void kernel_main() {
    using namespace compute_kernel_lib;

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

    uint32_t HtWt = Ht * Wt;
    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw) {
                    eltwise_chain(
                        1u,
                        UnaryBcast<BroadcastDim::Row, cb_id_src, cb_id_dst, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                        PackTile<cb_id_dst, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                    ++num_tiles_read;
                }
            }
        }
    }
}

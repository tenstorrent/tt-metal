// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
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
    // Emit the chain's one-time setup ONCE, out of the loop — the original raw bcast init
    // (BIG hw config + SCALAR bcast MOP + pack init). The per-(n,c) chain calls below pass
    // SetupOwner::Caller, so the chain skips that setup instead of re-emitting it each iteration.
    unary_bcast_init<BroadcastType::SCALAR>(cb_id_src, cb_id_dst);

    uint32_t HtWt = Ht * Wt;
    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            // Per-(n,c) 1-tile scalar broadcast; setup already emitted above (SetupOwner::Caller).
            compute_kernel_lib::eltwise_chain<compute_kernel_lib::SetupOwner::Caller>(
                1u,
                compute_kernel_lib::UnaryBcast<
                    compute_kernel_lib::BroadcastDim::Scalar,
                    cb_id_src,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::UnaryBcastReconfig::None>{},  // Caller owns setup -> no chain reconfig
                compute_kernel_lib::PackTile<
                    cb_id_dst,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
            num_tiles_read += HtWt - start_t;
        }
    }
}

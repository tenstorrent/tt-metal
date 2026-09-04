// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "tools/profiler/kernel_profiler.hpp"

namespace ckl = compute_kernel_lib;

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

    constexpr auto dfb_id_src_id = get_compile_time_arg_val(0);
    constexpr auto dfb_id_dst_id = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(dfb_id_src_id, dfb_id_dst_id);
    unary_bcast_init<BroadcastType::SCALAR>(dfb_id_src_id);

    uint32_t HtWt = Ht * Wt;
    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                ckl::IterationShape::one_tile(),
                // The caller owns setup, so the chain must not reconfigure formats.
                ckl::UnaryBcast<
                    ckl::BroadcastDim::Scalar,
                    ckl::input(
                        dfb_id_src_id,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled)>{},
                ckl::PackTile<ckl::output(
                    dfb_id_dst_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});
            num_tiles_read += HtWt - start_t;
        }
    }
}

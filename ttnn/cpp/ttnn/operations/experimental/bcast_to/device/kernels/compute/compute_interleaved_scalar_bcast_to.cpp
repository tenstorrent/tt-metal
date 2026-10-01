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
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t start_n = get_arg(args::start_n);
    uint32_t start_c = get_arg(args::start_c);
    uint32_t start_t = get_arg(args::start_t);
    uint32_t start_th = get_arg(args::start_th);
    uint32_t start_tw = get_arg(args::start_tw);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t n_stride = get_arg(args::n_stride);
    uint32_t c_stride = get_arg(args::c_stride);
    uint32_t N = get_arg(args::N);
    uint32_t C = get_arg(args::C);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);

    compute_kernel_hw_startup(dfb::src, dfb::dst);
    unary_bcast_init<BroadcastType::SCALAR>(dfb::src);

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
                        dfb::src,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled)>{},
                ckl::PackTile<ckl::output(
                    dfb::dst,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});
            num_tiles_read += HtWt - start_t;
        }
    }
}

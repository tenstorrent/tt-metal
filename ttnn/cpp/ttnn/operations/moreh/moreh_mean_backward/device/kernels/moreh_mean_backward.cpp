// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/kernel_args.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // compile-time args
    constexpr auto num_output_tiles = get_arg(args::num_output_tiles);
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    DataflowBuffer dfb_zero_obj(dfb::zero);  // zero tile
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::in, dfb::zero, dfb::out);
    dfb_zero_obj.wait_front(onetile);

    constexpr bool has_bcast = ht_need_bcast || wt_need_bcast;
    constexpr auto bcast_dim = (ht_need_bcast && wt_need_bcast) ? ckl::BroadcastDim::Scalar
                               : ht_need_bcast                  ? ckl::BroadcastDim::Row
                               : wt_need_bcast                  ? ckl::BroadcastDim::Col
                                                                : ckl::BroadcastDim::None;

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::Optional<
                has_bcast,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(dfb::zero, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::input(dfb::in, bcast_dim)>>{},
            ckl::Optional<!has_bcast, ckl::CopyTile<ckl::input(dfb::in)>>{},
            ckl::PackTile<ckl::output(dfb::intermed)>{});

        // output * (1 / number_of_elements)
        ckl::mul<
            ckl::input(dfb::intermed),
            // 1/num_dim bcast scalar
            ckl::input(dfb::scalar, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(dfb::out)>(ckl::IterationShape::tiles(onetile));
    }
    dfb_zero_obj.pop_front(onetile);
}

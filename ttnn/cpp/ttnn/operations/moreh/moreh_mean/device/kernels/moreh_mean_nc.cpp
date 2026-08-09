// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);

    constexpr auto dfb_in0_id = dfb::input;
    constexpr auto dfb_in1_id = dfb::in1;
    DataflowBuffer dfb_in1_obj(dfb_in1_id);
    constexpr auto dfb_scalar_id = dfb::scalar;
    DataflowBuffer dfb_scalar_obj(dfb_scalar_id);
    constexpr auto dfb_out0_id = dfb::out;
    constexpr auto dfb_intermed0_id = dfb::intermed0;
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::input, dfb::in1, dfb::out);

    dfb_in1_obj.wait_front(onetile);
    dfb_scalar_obj.wait_front(1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            if (enable_reload) {
                ckl::add<ckl::input(dfb_in0_id), ckl::input(dfb_intermed0_id), ckl::output(dfb_intermed0_id)>(
                    ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<
                    ckl::input(dfb_in0_id),
                    ckl::input(dfb_in1_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::output(dfb_intermed0_id),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
            }

            enable_reload = true;
        }

        ckl::mul<
            ckl::input(dfb_intermed0_id),
            ckl::input(dfb_scalar_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(dfb_out0_id),
            ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::tiles(onetile));
    }
}

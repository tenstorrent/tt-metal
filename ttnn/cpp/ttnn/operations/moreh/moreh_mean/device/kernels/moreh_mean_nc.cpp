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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);

    DataflowBuffer dfb_in1_obj(dfb::in1);
    DataflowBuffer dfb_scalar_obj(dfb::scalar);
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::input, dfb::in1, dfb::out);

    dfb_in1_obj.wait_front(onetile);
    dfb_scalar_obj.wait_front(1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            if (enable_reload) {
                ckl::add<ckl::input(dfb::input), ckl::input(dfb::intermed0), ckl::output(dfb::intermed0)>(
                    ckl::IterationShape::tiles(onetile));
            } else {
                ckl::add<
                    ckl::input(dfb::input),
                    ckl::input(dfb::in1, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::output(dfb::intermed0)>(ckl::IterationShape::tiles(onetile));
            }

            enable_reload = true;
        }

        // output * (1 / number_of_elements)
        ckl::mul<
            ckl::input(dfb::intermed0),
            ckl::input(dfb::scalar, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(dfb::out)>(ckl::IterationShape::tiles(onetile));
    }
}

// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    DataflowBuffer dfb_in1_obj(cb_in1);
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    DataflowBuffer dfb_scalar_obj(cb_scalar);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    dfb_in1_obj.wait_front(onetile);
    dfb_scalar_obj.wait_front(1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            if (enable_reload) {
                ckl::add<ckl::input(cb_in0), ckl::input(cb_intermed0), ckl::output(cb_intermed0)>(
                    ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<
                    ckl::input(cb_in0),
                    ckl::input(cb_in1, ckl::InputLifecycle::CallerManaged),
                    ckl::output(cb_intermed0),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
            }

            enable_reload = true;
        }

        ckl::mul<
            ckl::input(cb_intermed0),
            ckl::input(cb_scalar, ckl::InputLifecycle::CallerManaged),
            ckl::output(cb_out0),
            ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::tiles(onetile));
    }
}

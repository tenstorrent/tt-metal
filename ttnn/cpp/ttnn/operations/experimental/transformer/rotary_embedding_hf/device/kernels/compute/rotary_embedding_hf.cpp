// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

// out = input*cos + rotate_half(input)*sin.
template <uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id>
ALWI void mul_tiles_chain(DataflowBuffer& in0_cb, DataflowBuffer& in1_cb, DataflowBuffer& out_cb) {
    // Multiply input by cos or sin
    ckl::mul<
        ckl::input(in0_cb_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::input(in1_cb_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::output(
            out_cb_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::one_tile(), in0_cb, in1_cb, out_cb);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    DataflowBuffer scalar_cb(scalar_cb_id);
    DataflowBuffer in_cb(in_cb_id);
    DataflowBuffer rotated_in_cb(rotated_in_cb_id);
    DataflowBuffer cos_cb(cos_cb_id);
    DataflowBuffer sin_cb(sin_cb_id);
    DataflowBuffer rotated_in_interm_cb(rotated_in_interm_cb_id);
    DataflowBuffer cos_interm_cb(cos_interm_cb_id);
    DataflowBuffer sin_interm_cb(sin_interm_cb_id);
    DataflowBuffer out_cb(out_cb_id);
    scalar_cb.wait_front(onetile);

    compute_kernel_hw_startup(rotated_in_cb_id, scalar_cb_id, rotated_in_interm_cb_id);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                ckl::mul<
                    ckl::input(rotated_in_cb_id),
                    ckl::input(scalar_cb_id, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::output(rotated_in_interm_cb_id)>(
                    ckl::IterationShape::tiles(onetile), rotated_in_cb, scalar_cb, rotated_in_interm_cb);
                reconfig_data_format_srcb(scalar_cb_id, sin_cb_id);
                pack_reconfig_data_format(rotated_in_interm_cb_id, sin_interm_cb_id);
                // Multiply rotated input by sin
                mul_tiles_chain<rotated_in_interm_cb_id, sin_cb_id, sin_interm_cb_id>(
                    rotated_in_interm_cb, sin_cb, sin_interm_cb);
            } else {
                reconfig_data_format(rotated_in_cb_id, sin_cb_id);
                pack_reconfig_data_format(out_cb_id, sin_interm_cb_id);
                // Multiply rotated input by sin
                mul_tiles_chain<rotated_in_cb_id, sin_cb_id, sin_interm_cb_id>(rotated_in_cb, sin_cb, sin_interm_cb);
            }

            // Multiply input by cos
            mul_tiles_chain<in_cb_id, cos_cb_id, cos_interm_cb_id>(in_cb, cos_cb, cos_interm_cb);

            // Add applied sin/cos tensors
            ckl::add<ckl::input(cos_interm_cb_id), ckl::input(sin_interm_cb_id), ckl::output(out_cb_id)>(
                ckl::IterationShape::tiles(onetile), cos_interm_cb, sin_interm_cb, out_cb);
        }
    }
}

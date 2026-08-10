// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

template <uint32_t in0_dfb_id, uint32_t in1_dfb_id, uint32_t out_dfb_id>
ALWI void mul_tiles_chain() {
    ckl::mul<
        ckl::input(in0_dfb_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::input(in1_dfb_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::output(
            out_dfb_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::one_tile());
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_dfb_id = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_dfb_id = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    DataflowBuffer dfb_scalar(scalar_dfb_id);
    dfb_scalar.wait_front(onetile);

    compute_kernel_hw_startup(rotated_in_dfb_id, scalar_dfb_id, rotated_in_interm_dfb_id);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                ckl::mul<
                    ckl::input(rotated_in_dfb_id),
                    ckl::input(scalar_dfb_id, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::output(rotated_in_interm_dfb_id)>(ckl::IterationShape::tiles(onetile));
                reconfig_data_format_srcb(scalar_dfb_id, sin_dfb_id);
                pack_reconfig_data_format(rotated_in_interm_dfb_id, sin_interm_dfb_id);
                mul_tiles_chain<rotated_in_interm_dfb_id, sin_dfb_id, sin_interm_dfb_id>();
            } else {
                reconfig_data_format(rotated_in_dfb_id, sin_dfb_id);
                pack_reconfig_data_format(out_dfb_id, sin_interm_dfb_id);
                mul_tiles_chain<rotated_in_dfb_id, sin_dfb_id, sin_interm_dfb_id>();
            }

            mul_tiles_chain<in_dfb_id, cos_dfb_id, cos_interm_dfb_id>();

            ckl::add<ckl::input(cos_interm_dfb_id), ckl::input(sin_interm_dfb_id), ckl::output(out_dfb_id)>(
                ckl::IterationShape::tiles(onetile));
        }
    }
}

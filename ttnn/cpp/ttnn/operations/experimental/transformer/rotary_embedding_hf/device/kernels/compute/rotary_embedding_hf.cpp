// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb>
ALWI void mul_tiles_chain() {
    ckl::mul<
        ckl::input(in0_cb, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
        ckl::input(in1_cb, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
        ckl::output(out_cb, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
        ckl::BroadcastDim::None>(ckl::EltwiseShape::single());
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    cb_wait_front(scalar_cb, onetile);

    binary_op_init_common(rotated_in_cb, scalar_cb, rotated_in_interm_cb);

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            if (j < half_Wt) {
                ckl::mul<
                    ckl::input(rotated_in_cb),
                    ckl::input(scalar_cb, ckl::InputLifecycle::CallerManaged),
                    ckl::output(rotated_in_interm_cb),
                    ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::tiles(onetile));
                reconfig_data_format_srcb(scalar_cb, sin_cb);
                pack_reconfig_data_format(rotated_in_interm_cb, sin_interm_cb);
                mul_tiles_chain<rotated_in_interm_cb, sin_cb, sin_interm_cb>();
            } else {
                reconfig_data_format(rotated_in_cb, sin_cb);
                pack_reconfig_data_format(out_cb, sin_interm_cb);
                mul_tiles_chain<rotated_in_cb, sin_cb, sin_interm_cb>();
            }

            mul_tiles_chain<in_cb, cos_cb, cos_interm_cb>();

            ckl::add<ckl::input(cos_interm_cb), ckl::input(sin_interm_cb), ckl::output(out_cb)>(
                ckl::EltwiseShape::tiles(onetile));
        }
    }
}

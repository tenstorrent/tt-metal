// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"  // Dropout (owns dropout_kernel_init via init_runtime)

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t int_probability = get_compile_time_arg_val(2);
    constexpr uint32_t int_scale_factor = get_compile_time_arg_val(3);

    uint32_t seed = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(total_tiles),
        ckl::CopyTile<
            cb_input,
            ckl::Dst::D0,
            ckl::input(ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{},
        ckl::Dropout<ckl::Dst::D0>{int_probability, int_scale_factor, seed},
        ckl::PackTile<cb_output, ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}

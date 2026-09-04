// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"  // Dropout (owns dropout_kernel_init via init_runtime)

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t int_probability = get_compile_time_arg_val(2);
    constexpr uint32_t int_scale_factor = get_compile_time_arg_val(3);

    uint32_t seed = get_arg_val<uint32_t>(0);

    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(total_tiles).block_size(per_core_block_dim),
        ckl::CopyTile<
            ckl::input(
                dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::Dropout<ckl::Dst::D0>{int_probability, int_scale_factor, seed},
        ckl::PackTile<ckl::output(
            dfb_output_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}

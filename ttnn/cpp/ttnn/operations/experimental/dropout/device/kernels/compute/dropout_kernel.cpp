// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"  // Dropout
#include "api/compute/eltwise_unary/dropout.h"          // dropout_kernel_init

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t int_probability = get_compile_time_arg_val(2);
    constexpr uint32_t int_scale_factor = get_compile_time_arg_val(3);

    uint32_t seed = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);
    dropout_kernel_init(seed);

    // Original: per-tile copy_tile + dropout_tile + pack_tile. Chain compresses
    // the two nested loops into a single n=block_cnt*block_dim sweep, since
    // there's no inter-block state.
    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    compute_kernel_lib::eltwise_chain(
        total_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::Dropout<compute_kernel_lib::Dst::D0>{int_probability, int_scale_factor},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}

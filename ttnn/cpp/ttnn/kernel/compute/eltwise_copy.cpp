// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);

    compute_kernel_lib::copy<
        compute_kernel_lib::input(
            cb_in, compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            cb_out, compute_kernel_lib::OutputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled)>(
        compute_kernel_lib::EltwiseShape::tiles(per_core_tile_cnt));
}

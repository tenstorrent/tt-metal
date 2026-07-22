// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    compute_kernel_lib::copy<
        compute_kernel_lib::input(
            cb_input, compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            cb_output,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(compute_kernel_lib::EltwiseShape::tiles(num_tiles));
}

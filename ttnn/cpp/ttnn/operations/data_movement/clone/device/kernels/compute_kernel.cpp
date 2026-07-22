// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(src_cb_id, dst_cb_id);

    compute_kernel_lib::copy<
        compute_kernel_lib::input(
            src_cb_id, compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            dst_cb_id,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(compute_kernel_lib::EltwiseShape::tiles(num_tiles));
}

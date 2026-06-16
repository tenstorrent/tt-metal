// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of data_movement/untilize/device/kernels/compute/untilize_wh.cpp.
// The legacy source lives outside this op's directory (untilize) and is shared, so it is forked
// here (not edited in place) and ported to Metal 2.0 named bindings for untilize_with_unpadding's
// multi-core block-interleaved factory.
// Logic is UNCHANGED; only the access mechanism moves to named bindings:
//   block_size_col / block_size_row / third_dim CTAs -> named CTAs (get_arg(args::...))
//   CB ids c_0 / c_16 -> dfb::in / dfb::out

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t block_size_col = get_arg(args::block_size_col);
    constexpr uint32_t block_size_row = get_arg(args::block_size_row);
    constexpr uint32_t third_dim = get_arg(args::third_dim);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    compute_kernel_lib::untilize<
        block_size_row,
        dfb::in,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
        block_size_col * third_dim);
}

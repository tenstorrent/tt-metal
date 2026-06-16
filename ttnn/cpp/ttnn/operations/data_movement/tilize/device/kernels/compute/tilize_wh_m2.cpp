// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp,
// for the multi-core block tilize compute kernel. The legacy source is shared with the sibling
// tilize_with_val_padding block factory, so it is forked rather than edited in place.

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
// #include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t block_size_col = get_arg(args::block_size_col);
    constexpr uint32_t block_size_row = get_arg(args::block_size_row);
    constexpr uint32_t third_dim = get_arg(args::third_dim);

    constexpr uint32_t cb_in0 = dfb::src0;
    constexpr uint32_t cb_out = dfb::output;

    compute_kernel_hw_startup(cb_in0, cb_out);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<cb_in0>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        block_size_row,
        cb_in0,
        cb_out,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(block_size_col * third_dim);
}

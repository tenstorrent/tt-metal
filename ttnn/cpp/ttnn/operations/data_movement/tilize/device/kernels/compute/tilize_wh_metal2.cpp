// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of tilize_wh.cpp.
//
// Bindings:
//   dfb::src                — DFB endpoint (CONSUMER) — row-major input
//   dfb::dst                — DFB endpoint (PRODUCER) — tiled output
//   args::block_size_col    — CTA
//   args::block_size_row    — CTA
//   args::third_dim         — CTA

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto block_size_col = get_arg(args::block_size_col);
    constexpr auto block_size_row = get_arg(args::block_size_row);
    constexpr auto third_dim = get_arg(args::third_dim);

    compute_kernel_hw_startup(dfb::src, dfb::dst);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<dfb::src>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        block_size_row,
        dfb::src,
        dfb::dst,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(block_size_col * third_dim);
}

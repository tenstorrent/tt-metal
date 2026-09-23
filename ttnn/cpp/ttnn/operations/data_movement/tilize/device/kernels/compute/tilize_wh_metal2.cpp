// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of tilize_wh.cpp, which lives beside it. Ops ported to Metal 2.0 bind this file; the
// original serves the consumers still on the legacy API. Until the last of them migrates and the
// original is retired, changes here likely belong there too.
//
// The binding names below (dfb::in, dfb::out) and named args are this fork's interface: a factory
// binding this source must supply exactly these DFB / arg names.

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

    // Each instance binds the buffer set matching its cores' block width: `in` / `out` are that set's
    // input and output buffers. A factory whose work split gives cores different block widths declares
    // one correctly-sized buffer pair per width -- see BlockBufferSet in data_movement/common.
    compute_kernel_hw_startup(dfb::in, dfb::out);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<dfb::in>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        block_size_row,
        dfb::in,
        dfb::out,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(block_size_col * third_dim);
}

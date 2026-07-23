// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of tilize.cpp. Identical compute logic; the input/output CB indices become dfb::in /
// dfb::out named bindings and the remaining compile-time args become named. The legacy tilize.cpp is
// retained for the not-yet-ported consumers that still instantiate it (tilize, tilize_with_val_padding,
// untilize, untilize_with_unpadding, moreh/moreh_getitem, pool/upsample, sliding_window/halo,
// deepseek_prefill/combine, quasar/tilize_with_val_padding). Delete this fork once they are all on
// Metal 2.0.

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    compute_kernel_hw_startup(dfb::in, dfb::out);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<dfb::in>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        dfb::in,
        dfb::out,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);
}

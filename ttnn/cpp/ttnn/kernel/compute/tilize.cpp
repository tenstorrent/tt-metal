// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KEEP IN SYNC WITH: tilize_metal2.cpp (this directory)
//
// That file is the Metal 2.0 fork of this kernel: identical compute logic, with the buffer indices
// expressed as named DFB bindings (dfb::in / dfb::out) and the block counts as named compile-time
// args instead of hardcoded tt::CBIndex values and positional args. Ops whose program factory has
// been ported to the Metal 2.0 host API bind the fork; ops still on the legacy host API bind this
// file. A behavioural change to either one must be mirrored in the other.
//
// The duplication is temporary. Once the last legacy consumer is ported, delete this file and rename
// the fork over it.
//
// TODO(#52228): retire this duplication. The issue records why it exists, the full consumer
// list, and the sunset plan: https://github.com/tenstorrent/tt-metal/issues/52228

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// #include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // Use lossless tilize for fp32 inputs to preserve exact values (fast tilize truncates fp32 → tf32)
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<tt::CBIndex::c_0>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        tt::CBIndex::c_0,
        tt::CBIndex::c_16,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of tilize.cpp.
//
// Same logic, expressed against the Metal 2.0 named-binding APIs: the hardcoded tt::CBIndex::c_0 /
// c_16 buffer indices became the `dfb::in` / `dfb::out` DFB bindings, and the two positional CTAs
// became named arguments. `dfb::` handles flow straight into the kernel-lib's uint32_t template
// parameters and into compute_kernel_hw_startup via DFBBindingToken's constexpr conversion.
//
// This fork exists because the legacy original in this shared kernel pool is also bound by the three
// data_movement/tilize factories, which are still on the legacy host API; it lives alongside the
// original rather than replacing it.
//
// Binding vocabulary a Metal 2.0 KernelSpec must supply for this source:
//   dfb::in  — row-major input DFB, bound CONSUMER
//   dfb::out — tiled output DFB, bound PRODUCER
//   args::per_core_block_cnt, args::per_core_block_tile_cnt — compile-time args

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

// #include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    // Use lossless tilize for fp32 inputs to preserve exact values (fast tilize truncates fp32 → tf32)
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

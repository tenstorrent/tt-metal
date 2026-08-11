// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// #include "api/debug/dprint.h"

// Runtime-num-blocks variant of tilize.cpp.
//
// The width-in-tiles (per_core_block_tile_cnt) stays a compile-time template parameter to the
// tilize LLK (perf-critical). Only the work-split loop bound (per_core_block_cnt / nblocks_per_core)
// is read as a runtime arg so the compiled binary is invariant to the number of blocks assigned to
// a core. This lets shape-only changes hit the disk cache instead of recompiling.
void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(0);

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

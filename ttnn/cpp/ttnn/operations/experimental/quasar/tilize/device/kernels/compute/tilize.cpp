// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"  // [#48552 DIAG - remove after]

void kernel_main() {
    constexpr auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    // [#48552 DIAG - remove after] Brackets the tilize library call. If "entry" prints but "DONE" does not,
    // the hang (0x19 MEM_READ_NO_RESPONSE) is INSIDE compute_kernel_lib::tilize for this block config
    // (blocks x tiles/block); combined with the [TLZR] reader push/cap this localizes reader-vs-compute.
    DPRINT("[TLZC] entry blocks={} tiles/blk={}\n", (uint32_t)per_core_block_cnt, (uint32_t)per_core_block_tile_cnt);

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

    DPRINT("[TLZC] DONE\n");  // [#48552 DIAG - remove after]
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    // src_cb_id / out_cb_id are DFB bindings (dfb::src0 / dfb::src1); they flow into the
    // compute-kernel-lib helper via DFBAccessor's implicit (constexpr) uint32_t conversion.
    compute_kernel_hw_startup(dfb::src0, dfb::src1);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::src0,
        dfb::src1,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}

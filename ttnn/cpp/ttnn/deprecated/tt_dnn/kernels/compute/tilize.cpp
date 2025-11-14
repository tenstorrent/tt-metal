// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.h"

// #include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    compute_kernel_lib::tilize(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16, per_core_block_cnt);
}
}  // namespace NAMESPACE

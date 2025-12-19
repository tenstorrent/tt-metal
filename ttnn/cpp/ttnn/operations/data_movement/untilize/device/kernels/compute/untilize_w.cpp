// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t third_dim = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    compute_kernel_lib::untilize(
        tt::CBIndex::c_0, 1, tt::CBIndex::c_16, per_core_block_cnt * per_core_block_tile_cnt * third_dim);
}
}  // namespace NAMESPACE

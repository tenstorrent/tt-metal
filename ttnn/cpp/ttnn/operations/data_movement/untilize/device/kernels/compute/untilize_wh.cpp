// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t block_size_col = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_row = get_compile_time_arg_val(1);
    constexpr uint32_t third_dim = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    compute_kernel_lib::untilize<block_size_row>(tt::CBIndex::c_0, tt::CBIndex::c_16, block_size_col * third_dim);
}
}  // namespace NAMESPACE

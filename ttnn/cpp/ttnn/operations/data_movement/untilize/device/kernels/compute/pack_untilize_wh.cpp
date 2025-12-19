// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
#ifdef DST_ACCUM_MODE
    constexpr uint32_t max_bct = 4;
#else
    constexpr uint32_t max_bct = 8;
#endif
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_column(block_size_row, max_bct);
    constexpr uint32_t block_ct_dim = block_size_row / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = block_size_row;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // Unified untilize automatically:
    // - Detects DEST limit from DST_SYNC_MODE and DST_ACCUM_MODE
    // - Detects data format (integer vs non-integer) from unpack_dst_format
    // - Uses block-based pack_untilize for wide integer types (hardware-accelerated)
    // - Falls back to standard untilize for wide non-integer types
    compute_kernel_lib::untilize<block_size_row, tt::CBIndex::c_0, tt::CBIndex::c_16>(block_size_col * third_dim);
}

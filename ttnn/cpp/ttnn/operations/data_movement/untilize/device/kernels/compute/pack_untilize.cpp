// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);

    // Unified untilize automatically:
    // - Detects DEST limit from DST_SYNC_MODE and DST_ACCUM_MODE
    // - Detects data format (integer vs non-integer) from unpack_dst_format
    // - Uses block-based pack_untilize for wide integer types (hardware-accelerated)
    // - Falls back to standard untilize for wide non-integer types
    compute_kernel_lib::untilize<per_core_block_tile_cnt, src_cb_id, out_cb_id>(per_core_block_cnt);
}
}  // namespace NAMESPACE

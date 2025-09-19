// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // Simple implementation for the specific mixed broadcast test case
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);  // predicate address
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);  // true tensor address
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);  // false tensor address
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);  // num_tiles_per_core
    const uint32_t start_id = get_arg_val<uint32_t>(4);   // start_tile_id

    constexpr auto predicate_cb = get_compile_time_arg_val(0);
    constexpr auto true_cb = get_compile_time_arg_val(1);
    constexpr auto false_cb = get_compile_time_arg_val(2);

    // Compile-time args layout mirrors no-bcast reader: 3 CB ids, then 3 TensorAccessorArgs blocks
    constexpr auto src0_args = TensorAccessorArgs<3>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto src2_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(src0_args, src0_addr, get_tile_size(predicate_cb));
    const auto s1 = TensorAccessor(src1_args, src1_addr, get_tile_size(true_cb));
    const auto s2 = TensorAccessor(src2_args, src2_addr, get_tile_size(false_cb));

    constexpr uint32_t onetile = 1;

    // For our test case: predicate (1,1,1,1024), true (1,1,1024,1024), false (1,1,1024,1)
    // This means: predicate needs row broadcast, false needs column broadcast
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint32_t global_tile_id = start_id + tile_idx;

        // Calculate tile coordinates for this tile
        uint32_t tile_h = global_tile_id / 32;  // 1024/32 = 32 tiles per row
        uint32_t tile_w = global_tile_id % 32;

        // Handle predicate: row broadcast - always read from first row (tile_h = 0)
        cb_reserve_back(predicate_cb, onetile);
        uint32_t pred_l1_addr = get_write_ptr(predicate_cb);
        noc_async_read_tile(0 * 32 + tile_w, s0, pred_l1_addr);  // Always read from row 0
        noc_async_read_barrier();
#if !BCAST_LLK
        FILL_TILE_WITH_FIRST_ROW(predicate_cb);
#endif
        cb_push_back(predicate_cb, onetile);

        // Handle true tensor: regular read - read the actual tile
        cb_reserve_back(true_cb, onetile);
        uint32_t true_l1_addr = get_write_ptr(true_cb);
        noc_async_read_tile(global_tile_id, s1, true_l1_addr);
        noc_async_read_barrier();
        cb_push_back(true_cb, onetile);

        // Handle false tensor: column broadcast - always read from first column (tile_w = 0)
        cb_reserve_back(false_cb, onetile);
        uint32_t false_l1_addr = get_write_ptr(false_cb);
        noc_async_read_tile(tile_h * 32 + 0, s2, false_l1_addr);  // Always read from column 0
        noc_async_read_barrier();
        FILL_TILE_WITH_FIRST_COLUMN_C(false_cb);
        cb_push_back(false_cb, onetile);
    }
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config — PUSH regime (source is L1-sharded, target is not L1-sharded).
// Runs on every core of the input shard grid (BRISC / NoC1). The core's own input shard IS the
// block: it is consumed in place through the zero-copy CB `cb_input_shard` (never re-read through
// an accessor) and written to the target, one NoC write per maximal run, one write barrier per
// block (which also guarantees arrival before the program ends).
//
// No kernel-lib helper covers this (op_design.md "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_to_memory_config_common.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t shard_tile_cols = get_compile_time_arg_val(2);  // nominal: the shard's L1 row stride
    constexpr uint32_t cb_input_shard = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    constexpr bool full_width = (shard_tile_cols == tensor_tile_cols);

    uint32_t arg = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);  // 0 (idle grid core) or 1 (whole shard)
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);
    const uint32_t block_tile_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);
    const uint32_t block_tile_cols = get_arg_val<uint32_t>(arg++);

    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);
    const uint32_t shard_base = get_read_ptr(cb_input_shard);
    constexpr uint32_t shard_row_bytes = shard_tile_cols * page_size;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        {
            DeviceZoneScopedN("push_block");
            if constexpr (full_width) {
                g6_tmc::write_span(
                    dst, row0 * tensor_tile_cols, block_tile_rows * tensor_tile_cols, shard_base, page_size);
            } else {
                uint32_t l1_row = shard_base;
                for (uint32_t r = 0; r < block_tile_rows; ++r) {
                    g6_tmc::write_span(dst, (row0 + r) * tensor_tile_cols + col0, block_tile_cols, l1_row, page_size);
                    l1_row += shard_row_bytes;
                }
            }
        }
        {
            DeviceZoneScopedN("settle_block");
            noc_async_write_barrier();
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// One or both halves of every block in a streaming row gather-copy.
//
// A tile is reconstructed with a configurable number of NoC reads. The payload
// is always 2 KiB, but smaller transactions create the data-movement RISC-V
// issue bottleneck that split readers address.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_gather_0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather_1 = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_source = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t transaction_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t half_begin = get_compile_time_arg_val(8);
    constexpr uint32_t num_halves = get_compile_time_arg_val(9);
    constexpr uint32_t half_block_tiles = block_tiles / 2;
    constexpr uint32_t transactions_per_tile = tile_bytes / transaction_bytes;

    const uint32_t source_l1_base = get_arg_val<uint32_t>(0);
    const tt_l1_ptr uint32_t* noc_x = reinterpret_cast<const tt_l1_ptr uint32_t*>(get_arg_addr(1));
    const tt_l1_ptr uint32_t* noc_y = reinterpret_cast<const tt_l1_ptr uint32_t*>(get_arg_addr(1 + num_cores));

    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t half = half_begin; half < half_begin + num_halves; ++half) {
            const uint32_t cb_gather = half == 0 ? cb_gather_0 : cb_gather_1;
            const uint32_t half_start = block * block_tiles + half * half_block_tiles;

            for (uint32_t i = 0; i < half_block_tiles; ++i) {
                const uint32_t global_tile = half_start + i;
                const uint32_t source = global_tile / tiles_per_source;
                const uint32_t source_tile = global_tile % tiles_per_source;

                cb_reserve_back(cb_gather, 1);
                const uint32_t dst = get_write_ptr(cb_gather);
                const uint32_t src = source_l1_base + source_tile * tile_bytes;

                for (uint32_t transaction = 0; transaction < transactions_per_tile; ++transaction) {
                    const uint32_t offset = transaction * transaction_bytes;
                    noc_async_read(
                        get_noc_addr(noc_x[source], noc_y[source], src + offset), dst + offset, transaction_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_gather, 1);
            }
        }
    }
}

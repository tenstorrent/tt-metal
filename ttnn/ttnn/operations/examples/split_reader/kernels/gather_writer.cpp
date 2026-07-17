// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC / NoC1 writes the gathered tiles to DRAM in their original order. In
// split-reader mode it also gathers the second half of each block from the
// row-sharded L1 input. There is no transform or Tensix compute stage.

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
    constexpr bool split_reader = get_compile_time_arg_val(8) != 0;
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr uint32_t half_block_tiles = block_tiles / 2;
    constexpr uint32_t transactions_per_tile = tile_bytes / transaction_bytes;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t source_l1_base = get_arg_val<uint32_t>(1);
    const tt_l1_ptr uint32_t* noc_x = reinterpret_cast<const tt_l1_ptr uint32_t*>(get_arg_addr(2));
    const tt_l1_ptr uint32_t* noc_y = reinterpret_cast<const tt_l1_ptr uint32_t*>(get_arg_addr(2 + num_cores));

    const auto out_acc = TensorAccessor(out_args, dst_addr, tile_bytes);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if constexpr (split_reader) {
            const uint32_t half_start = block * block_tiles + half_block_tiles;
            for (uint32_t i = 0; i < half_block_tiles; ++i) {
                const uint32_t global_tile = half_start + i;
                const uint32_t source = global_tile / tiles_per_source;
                const uint32_t source_tile = global_tile % tiles_per_source;

                cb_reserve_back(cb_gather_1, 1);
                const uint32_t l1_write_addr = get_write_ptr(cb_gather_1);
                const uint32_t src = source_l1_base + source_tile * tile_bytes;
                for (uint32_t transaction = 0; transaction < transactions_per_tile; ++transaction) {
                    const uint32_t offset = transaction * transaction_bytes;
                    noc_async_read(
                        get_noc_addr(noc_x[source], noc_y[source], src + offset),
                        l1_write_addr + offset,
                        transaction_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_gather_1, 1);
            }
        }

        cb_wait_front(cb_gather_0, half_block_tiles);
        cb_wait_front(cb_gather_1, half_block_tiles);
        const uint32_t first_half_l1 = get_read_ptr(cb_gather_0);
        const uint32_t second_half_l1 = get_read_ptr(cb_gather_1);
        const uint32_t output_block_start = block * block_tiles;
        for (uint32_t i = 0; i < half_block_tiles; ++i) {
            noc_async_write(first_half_l1 + i * tile_bytes, out_acc.get_noc_addr(output_block_start + i), tile_bytes);
            noc_async_write(
                second_half_l1 + i * tile_bytes,
                out_acc.get_noc_addr(output_block_start + half_block_tiles + i),
                tile_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_gather_0, half_block_tiles);
        cb_pop_front(cb_gather_1, half_block_tiles);
    }
    noc_async_write_barrier();
}

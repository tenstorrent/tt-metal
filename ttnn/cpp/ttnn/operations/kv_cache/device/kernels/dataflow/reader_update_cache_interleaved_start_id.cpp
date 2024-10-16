// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_addr = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t B = get_arg_val<uint32_t>(3);
    const uint32_t num_batched_heads = get_arg_val<uint32_t>(4);
    const uint32_t cache_total_num_tiles = get_arg_val<uint32_t>(5);
    const uint32_t cache_batch_num_tiles = get_arg_val<uint32_t>(6);
    const uint32_t cache_head_num_tiles = get_arg_val<uint32_t>(7);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(8);
    const uint32_t input_start_id = get_arg_val<uint32_t>(9);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(10);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t granularity = get_compile_time_arg_val(4);
    constexpr uint32_t u_count = get_compile_time_arg_val(5);

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const DataFormat input_data_format = get_dataformat(input_cb_id);

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr, .page_size = cache_tile_bytes, .data_format = cache_data_format};
#ifdef INPUT_SHARDED
    cb_reserve_back(input_cb_id, Wt * num_batched_heads);
    cb_push_back(input_cb_id, Wt * num_batched_heads);
#else
    const InterleavedAddrGenFast<input_is_dram> s1 = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    uint32_t input_id = input_start_id;
#endif

    uint32_t cache_id = cache_start_id;
    uint32_t b = batch_start_id;

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
#ifndef INPUT_SHARDED
        cb_reserve_back(input_cb_id, Wt);
        uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
        for (uint32_t i = 0; i < Wt; ++i) {
            noc_async_read_tile(input_id, s1, input_l1_write_addr);
            input_l1_write_addr += input_tile_bytes;
            input_id++;
        }
        noc_async_read_barrier();
        cb_push_back(input_cb_id, Wt);
#endif
        for (uint32_t u = 0; u < u_count; ++u) {
            cb_reserve_back(cache_cb_id, Wt * granularity);
            uint32_t cache_l1_write_addr = get_write_ptr(cache_cb_id);
            for (uint32_t g = 0; g < granularity; ++g) {
                for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
                    cache_l1_write_addr += cache_tile_bytes;
                }
                cache_id += cache_batch_num_tiles;  // Input is read in by batch, then heads so skip to next batch
                b++;
                if (b == B) {
                    b = 0;
                    cache_id = cache_id - cache_total_num_tiles + cache_head_num_tiles;  // Start of next head
                }
            }

            noc_async_read_barrier();
            cb_push_back(cache_cb_id, Wt * granularity);
        }
    }
}

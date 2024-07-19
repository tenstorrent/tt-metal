// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "dprint.h"

void kernel_main() {
    const uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    const uint32_t Wt          = get_arg_val<uint32_t>(1);
    const uint32_t B           = get_arg_val<uint32_t>(2);
    const uint32_t num_batched_heads      = get_arg_val<uint32_t>(3);
    const uint32_t cache_total_num_tiles  = get_arg_val<uint32_t>(4);
    const uint32_t cache_batch_num_tiles  = get_arg_val<uint32_t>(5);
    const uint32_t cache_head_num_tiles   = get_arg_val<uint32_t>(6);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(7);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(8);
    const uint32_t Wbytes      = get_arg_val<uint32_t>(9);
    const uint32_t offset      = get_arg_val<uint32_t>(10);
    const uint32_t batch_read_offset = get_arg_val<uint32_t>(11);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t granularity = get_compile_time_arg_val(5);
    constexpr uint32_t u_count = get_compile_time_arg_val(6);


    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };

    uint32_t cache_id = cache_start_id;
    uint32_t b = batch_start_id;

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        cb_wait_front(untilized_input_cb_id, Wt);
        uint64_t input_l1_read_addr = get_noc_addr(get_read_ptr(untilized_input_cb_id)) + batch_read_offset;

        for (uint32_t u = 0; u < u_count; ++u) {
            // Operating on a granularity > 1 led to performance improvements.
            // It introduces a double-buffered pipeline between compute and writer.
            for (uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to untilize a block. Update that block in L1.
                cb_wait_front(untilized_cache_cb_id, Wt);
                cb_reserve_back(untilized_cache2_cb_id, Wt);
                uint32_t cache_l1_write_addr = get_read_ptr(untilized_cache_cb_id) + offset;
                noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
                input_l1_read_addr += Wbytes;
                noc_async_read_barrier();
                cb_push_back(untilized_cache2_cb_id, Wt);
                cb_pop_front(untilized_cache_cb_id, Wt); // NEW
            }

            for (uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to tilize an updated block. Write that block to DRAM
                cb_wait_front(cache_cb_id, Wt);
                uint32_t out_l1_read_addr = get_read_ptr(cache_cb_id);
                for(uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
                    out_l1_read_addr += cache_tile_bytes;
                }
                cache_id += cache_batch_num_tiles; // Input is read in by batch, then heads so skip to next batch
                b++;
                if (b == B) {
                    b = 0;
                    cache_id = cache_id - cache_total_num_tiles + cache_head_num_tiles; // Start of next head
                }
                noc_async_writes_flushed();
                cb_pop_front(cache_cb_id, Wt);
            }
        }
        cb_pop_front(untilized_input_cb_id, Wt);
    }
    // Delay syncing the writes to maximize perf.
    noc_async_write_barrier();
}

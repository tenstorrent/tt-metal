// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
// #include "dprint.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t B = get_arg_val<uint32_t>(2);
    const uint32_t num_batched_heads = get_arg_val<uint32_t>(3);
    const uint32_t cache_total_num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t cache_batch_num_tiles = get_arg_val<uint32_t>(5);
    const uint32_t cache_head_num_tiles = get_arg_val<uint32_t>(6);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(7);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(8);
    const uint32_t Wbytes = get_arg_val<uint32_t>(9);
    const uint32_t offset = get_arg_val<uint32_t>(10);
    const uint32_t batch_read_offset = get_arg_val<uint32_t>(11);

    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t granularity = get_compile_time_arg_val(4);
    constexpr uint32_t u_count = get_compile_time_arg_val(5);
    constexpr auto cache_args = TensorAccessorArgs<6>();

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);

    Noc noc;
    CircularBuffer cb_cache(cache_cb_id);
    CircularBuffer cb_untilized_cache(untilized_cache_cb_id);
    CircularBuffer cb_untilized_cache2(untilized_cache2_cb_id);
    CircularBuffer cb_untilized_input(untilized_input_cb_id);

    const auto s0 = TensorAccessor(cache_args, cache_addr);

    uint32_t cache_id = cache_start_id;
    uint32_t b = batch_start_id;

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        cb_untilized_input.wait_front(Wt);
        uint32_t input_l1_read_addr = cb_untilized_input.get_read_ptr() + batch_read_offset;

        for (uint32_t u = 0; u < u_count; ++u) {
            // Operating on a granularity > 1 led to performance improvements.
            // It introduces a double-buffered pipeline between compute and writer.
            for (uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to untilize a block. Update that block in L1.
                cb_untilized_cache.wait_front(Wt);
                cb_untilized_cache2.reserve_back(Wt);
                uint32_t cache_l1_write_addr = cb_untilized_cache.get_read_ptr() + offset;
                noc.async_read(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(cache_l1_write_addr),
                    Wbytes,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = input_l1_read_addr},
                    {});
                input_l1_read_addr += Wbytes;
                noc.async_read_barrier();
                cb_untilized_cache2.push_back(Wt);
                cb_untilized_cache.pop_front(Wt);  // NEW
            }

            for (uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to tilize an updated block. Write that block to DRAM
                cb_cache.wait_front(Wt);
                uint32_t out_l1_read_offset = 0;
                for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc.async_write(
                        cb_cache,
                        s0,
                        cache_tile_bytes,
                        {.offset_bytes = out_l1_read_offset},
                        {.page_id = curr_cache_id});
                    out_l1_read_offset += cache_tile_bytes;
                }
                cache_id += cache_batch_num_tiles;  // Input is read in by batch, then heads so skip to next batch
                b++;
                if (b == B) {
                    b = 0;
                    cache_id = cache_id - cache_total_num_tiles + cache_head_num_tiles;  // Start of next head
                }
                noc.async_writes_flushed();
                cb_cache.pop_front(Wt);
            }
        }
        cb_untilized_input.pop_front(Wt);
    }
    // Delay syncing the writes to maximize perf.
    noc.async_write_barrier();
}

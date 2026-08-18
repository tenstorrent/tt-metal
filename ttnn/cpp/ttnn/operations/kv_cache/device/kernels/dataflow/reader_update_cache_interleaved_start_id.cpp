// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

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

    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t granularity = get_compile_time_arg_val(2);
    constexpr uint32_t u_count = get_compile_time_arg_val(3);
    constexpr auto cache_args = TensorAccessorArgs<4>();
    constexpr auto input_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);

    Noc noc;
    CircularBuffer cb_cache(cache_cb_id);
    CircularBuffer cb_input(input_cb_id);

    const auto s0 = TensorAccessor(cache_args, cache_addr);
#ifdef INPUT_SHARDED
    cb_input.reserve_back(Wt * num_batched_heads);
    cb_input.push_back(Wt * num_batched_heads);
#else
    const auto s1 = TensorAccessor(input_args, input_addr);
    uint32_t input_id = input_start_id;
#endif

    uint32_t cache_id = cache_start_id;
    uint32_t b = batch_start_id;

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
#ifndef INPUT_SHARDED
        cb_input.reserve_back(Wt);
        uint32_t input_l1_write_offset = 0;
        for (uint32_t i = 0; i < Wt; ++i) {
            noc.async_read(
                s1, cb_input, input_tile_bytes, {.page_id = input_id}, {.offset_bytes = input_l1_write_offset});
            input_l1_write_offset += input_tile_bytes;
            input_id++;
        }
        noc.async_read_barrier();
        cb_input.push_back(Wt);
#endif
        for (uint32_t u = 0; u < u_count; ++u) {
            cb_cache.reserve_back(Wt * granularity);
            uint32_t cache_l1_write_offset = 0;
            for (uint32_t g = 0; g < granularity; ++g) {
                for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc.async_read(
                        s0,
                        cb_cache,
                        cache_tile_bytes,
                        {.page_id = curr_cache_id},
                        {.offset_bytes = cache_l1_write_offset});
                    cache_l1_write_offset += cache_tile_bytes;
                }
                cache_id += cache_batch_num_tiles;  // Input is read in by batch, then heads so skip to next batch
                b++;
                if (b == B) {
                    b = 0;
                    cache_id = cache_id - cache_total_num_tiles + cache_head_num_tiles;  // Start of next head
                }
            }

            noc.async_read_barrier();
            cb_cache.push_back(Wt * granularity);
        }
    }
}

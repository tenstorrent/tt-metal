// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t B = get_arg(args::B);
    const std::uint32_t num_batched_heads = get_arg(args::num_batched_heads);
    const std::uint32_t cache_total_num_tiles = get_arg(args::cache_total_num_tiles);
    const std::uint32_t cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    const std::uint32_t cache_head_num_tiles = get_arg(args::cache_head_num_tiles);
    const std::uint32_t cache_start_id = get_arg(args::cache_start_id);
    const std::uint32_t input_start_id = get_arg(args::input_start_id);
    const std::uint32_t batch_start_id = get_arg(args::batch_start_id);

    constexpr auto granularity = get_arg(args::granularity);
    constexpr auto u_count = get_arg(args::u_count);

    Noc noc;
    // cache = input tensor s0 (read from the cache/output tensor); input = the update source (s1).
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_input(dfb::input);

    const std::uint32_t cache_tile_bytes = dfb_cache.get_tile_size();
    const std::uint32_t input_tile_bytes = dfb_input.get_tile_size();

    const auto s0 = TensorAccessor(tensor::cache);
#ifdef INPUT_SHARDED
    dfb_input.reserve_back(Wt * num_batched_heads);
    dfb_input.push_back(Wt * num_batched_heads);
#else
    const auto s1 = TensorAccessor(tensor::input);
    std::uint32_t input_id = input_start_id;
#endif

    std::uint32_t cache_id = cache_start_id;
    std::uint32_t b = batch_start_id;

    for (std::uint32_t h = 0; h < num_batched_heads; ++h) {
#ifndef INPUT_SHARDED
        dfb_input.reserve_back(Wt);
        std::uint32_t input_l1_write_offset = 0;
        for (std::uint32_t i = 0; i < Wt; ++i) {
            noc.async_read(
                s1, dfb_input, input_tile_bytes, {.page_id = input_id}, {.offset_bytes = input_l1_write_offset});
            input_l1_write_offset += input_tile_bytes;
            input_id++;
        }
        noc.async_read_barrier();
        dfb_input.push_back(Wt);
#endif
        for (std::uint32_t u = 0; u < u_count; ++u) {
            dfb_cache.reserve_back(Wt * granularity);
            std::uint32_t cache_l1_write_offset = 0;
            for (std::uint32_t g = 0; g < granularity; ++g) {
                for (std::uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc.async_read(
                        s0,
                        dfb_cache,
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
            dfb_cache.push_back(Wt * granularity);
        }
    }
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
// #include "dprint.h"

void kernel_main() {
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t B = get_arg(args::B);
    const std::uint32_t num_batched_heads = get_arg(args::num_batched_heads);
    const std::uint32_t cache_total_num_tiles = get_arg(args::cache_total_num_tiles);
    const std::uint32_t cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    const std::uint32_t cache_head_num_tiles = get_arg(args::cache_head_num_tiles);
    const std::uint32_t cache_start_id = get_arg(args::cache_start_id);
    const std::uint32_t batch_start_id = get_arg(args::batch_start_id);
    const std::uint32_t Wbytes = get_arg(args::Wbytes);
    const std::uint32_t offset = get_arg(args::offset);
    const std::uint32_t batch_read_offset = get_arg(args::batch_read_offset);

    constexpr auto granularity = get_arg(args::granularity);
    constexpr auto u_count = get_arg(args::u_count);

    Noc noc;
    // cache holds tilized data destined for the cache tensor (the output DFB); untilized_cache /
    // untilized_cache2 alias one L1 region (compute untilizes into untilized_cache; the writer pokes
    // the update in place and hands it back as untilized_cache2 for compute to re-tilize).
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_untilized_cache(dfb::untilized_cache);
    DataflowBuffer dfb_untilized_cache2(dfb::untilized_cache2);
    DataflowBuffer dfb_untilized_input(dfb::untilized_input);

    const std::uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    const auto s0 = TensorAccessor(tensor::cache);

    std::uint32_t cache_id = cache_start_id;
    std::uint32_t b = batch_start_id;

    for (std::uint32_t h = 0; h < num_batched_heads; ++h) {
        dfb_untilized_input.wait_front(Wt);
        std::uint32_t input_l1_read_addr = dfb_untilized_input.get_read_ptr() + batch_read_offset;

        for (std::uint32_t u = 0; u < u_count; ++u) {
            // Operating on a granularity > 1 led to performance improvements.
            // It introduces a double-buffered pipeline between compute and writer.
            for (std::uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to untilize a block. Update that block in L1.
                dfb_untilized_cache.wait_front(Wt);
                dfb_untilized_cache2.reserve_back(Wt);
                std::uint32_t cache_l1_write_addr = dfb_untilized_cache.get_read_ptr() + offset;
                noc.async_read(
                    UnicastEndpoint{},
                    CoreLocalMem<std::uint32_t>(cache_l1_write_addr),
                    Wbytes,
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                     .addr = input_l1_read_addr},
                    {});
                input_l1_read_addr += Wbytes;
                noc.async_read_barrier();
                dfb_untilized_cache2.push_back(Wt);
                dfb_untilized_cache.pop_front(Wt);  // NEW
            }

            for (std::uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to tilize an updated block. Write that block to DRAM
                dfb_cache.wait_front(Wt);
                std::uint32_t out_l1_read_offset = 0;
                for (std::uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc.async_write(
                        dfb_cache,
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
                dfb_cache.pop_front(Wt);
            }
        }
        dfb_untilized_input.pop_front(Wt);
    }
    // Delay syncing the writes to maximize perf.
    noc.async_write_barrier();
}

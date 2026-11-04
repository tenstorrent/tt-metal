// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr bool is_tile = get_arg(args::is_tile) != 0;
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr uint32_t barrier_threshold = get_arg(args::barrier_threshold);
    constexpr uint32_t max_page_size = get_arg(args::one_packet) != 0 ? NOC_MAX_BURST_SIZE : NOC_MAX_BURST_SIZE + 1;

    const uint32_t rows = get_arg(args::rows);
    [[maybe_unused]] const uint32_t row_tiles = get_arg(args::row_tiles);
    [[maybe_unused]] const uint32_t row_bytes = get_arg(args::row_bytes);
    const uint32_t local_stride = get_arg(args::local_stride);
    [[maybe_unused]] const uint32_t src_offset = get_arg(args::src_offset);
    [[maybe_unused]] const uint32_t src_stride = get_arg(args::src_stride);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t local_offset = get_arg(args::local_offset);

    Noc noc;
    DataflowBuffer dfb(dfb::shard);
    CoreLocalMem<uint32_t> shard(dfb.get_write_ptr());
    const auto src = TensorAccessor(tensor::src);

    uint32_t l1_offset = local_offset;
    uint32_t row_id = start_id;
    if constexpr (is_tile) {
        uint32_t count = 0;
        for (uint32_t h = 0; h < rows; ++h) {
            uint32_t tile_id = row_id;
            for (uint32_t w = 0; w < row_tiles; ++w) {
                noc.async_read<NocOptions::DEFAULT, max_page_size>(
                    src, shard, tile_bytes, {.page_id = tile_id}, {.offset_bytes = l1_offset});
                tile_id++;
                l1_offset += tile_bytes;
                if (++count == barrier_threshold) {
                    noc.async_read_barrier();
                    count = 0;
                }
            }
            l1_offset += local_stride;
            row_id += src_stride;
        }
    } else {
        for (uint32_t h = 0; h < rows; ++h) {
            noc.async_read<NocOptions::DEFAULT, max_page_size>(
                src, shard, row_bytes, {.page_id = row_id, .offset_bytes = src_offset}, {.offset_bytes = l1_offset});
            row_id++;
            l1_offset += local_stride;
        }
    }
    noc.async_read_barrier();
}

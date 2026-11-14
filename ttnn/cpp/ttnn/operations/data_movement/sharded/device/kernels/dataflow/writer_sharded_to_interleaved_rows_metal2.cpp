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
    constexpr uint32_t max_page_size = get_arg(args::one_packet) != 0 ? NOC_MAX_BURST_SIZE : NOC_MAX_BURST_SIZE + 1;

    const uint32_t rows = get_arg(args::rows);
    [[maybe_unused]] const uint32_t row_tiles = get_arg(args::row_tiles);
    [[maybe_unused]] const uint32_t row_bytes = get_arg(args::row_bytes);
    const uint32_t local_stride = get_arg(args::local_stride);
    [[maybe_unused]] const uint32_t dst_offset = get_arg(args::dst_offset);
    [[maybe_unused]] const uint32_t dst_stride = get_arg(args::dst_stride);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t local_offset = get_arg(args::local_offset);

    Noc noc;
    DataflowBuffer dfb(dfb::shard);
    CoreLocalMem<uint32_t> shard(dfb.get_write_ptr());
    const auto dst = TensorAccessor(tensor::dst);

    uint32_t l1_offset = local_offset;
    uint32_t row_id = start_id;
    if constexpr (is_tile) {
        for (uint32_t h = 0; h < rows; ++h) {
            uint32_t tile_id = row_id;
            for (uint32_t w = 0; w < row_tiles; ++w) {
                noc.async_write<NocOptions::DEFAULT, max_page_size>(
                    shard, dst, tile_bytes, {.offset_bytes = l1_offset}, {.page_id = tile_id});
                tile_id++;
                l1_offset += tile_bytes;
            }
            l1_offset += local_stride;
            row_id += dst_stride;
        }
    } else {
        for (uint32_t h = 0; h < rows; ++h) {
            noc.async_write<NocOptions::DEFAULT, max_page_size>(
                shard, dst, row_bytes, {.offset_bytes = l1_offset}, {.page_id = row_id, .offset_bytes = dst_offset});
            row_id++;
            l1_offset += local_stride;
        }
    }
    noc.async_write_barrier();
}

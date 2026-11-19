// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local copy writer (terms: fabric_all_gather_chunk_walk.hpp), for an interleaved input: writes this chip's own shard,
// as its reader fetched it, into this chip's own output, one physically contiguous write per fabric chunk. Same CB
// batches as the reader. (With any other input the link workers write the own shard themselves.)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "fabric_all_gather_common.hpp"

void kernel_main() {
    constexpr uint32_t chunk_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_fabric_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_dram_banks = get_compile_time_arg_val(4);
    constexpr uint32_t chunks_per_cb_batch = get_compile_time_arg_val(5);
    constexpr bool kValidPrefixFromMetadata = get_compile_time_arg_val(6) != 0;
    constexpr auto output_args = TensorAccessorArgs<7>();
    constexpr auto valid_prefix_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t fabric_chunk_bytes = pages_per_fabric_chunk * page_bytes;

    // Common args: the reader's ([0] input [1] output ... [3..7] cache slot, then the chip shard geometry at 8), so
    // both compute the same chunks. Per-core args: [0] first owned chunk [1] owned chunk stride [2] rank.
    constexpr uint32_t kGeometryCommonArgs = 8;
    Noc noc;
    CircularBuffer chunk_cb(chunk_cb_id);
    const uint32_t landing_l1 = CircularBuffer(metadata_cb_id).get_write_ptr();
    const ChipShardGeometry geometry =
        read_chip_shard_geometry<kValidPrefixFromMetadata, kGeometryCommonArgs>(noc, landing_l1, valid_prefix_args);
    if (!geometry.active) {  // every chip reads the same chunk start, so every kernel of the op skips together
        return;
    }
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(1), page_bytes);
    const uint32_t first_owned_chunk = get_arg_val<uint32_t>(0);
    const uint32_t owned_chunk_stride = get_arg_val<uint32_t>(1);
    const uint32_t rank = get_arg_val<uint32_t>(2);

    ChunkCbCursor<chunks_per_cb_batch> cb_cursor;
    uint32_t chunks_in_batch = 0;
    auto pop_batch = [&]() {
        noc.async_write_barrier();
        chunk_cb.pop_front(pages_per_fabric_chunk * chunks_in_batch);
        cb_cursor.advance(chunks_in_batch);
        chunks_in_batch = 0;
    };
    for_each_fabric_chunk(
        geometry,
        num_dram_banks,
        pages_per_fabric_chunk,
        first_owned_chunk,
        owned_chunk_stride,
        kWholeChipShard,
        [&](uint32_t outer_slice, uint32_t page_in_slice, uint32_t num_pages) {
            chunk_cb.wait_front(pages_per_fabric_chunk * (chunks_in_batch + 1));
            noc.async_write(
                CoreLocalMem<uint32_t>(chunk_cb.get_read_ptr() + chunks_in_batch * fabric_chunk_bytes),
                output,
                num_pages * page_bytes,
                {},
                {.page_id = output_page_index(geometry, rank, outer_slice, page_in_slice)});
            if (++chunks_in_batch == cb_cursor.capacity()) {
                pop_batch();
            }
        });
    if (chunks_in_batch > 0) {
        pop_batch();
    }
}

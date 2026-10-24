// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"

// Move this core's output block to the storage cores that own it. Each segment is two values: the
// byte count and the output shard it lands in (storage core i holds output shard i), written through
// `output` addressed by shard.
template <typename DFB, typename Accessor>
inline void write_resharded_data(
    const Noc& noc,
    DFB& dfb_out,
    const Accessor& output,
    uint32_t num_segments_to_write_back,
    uint32_t storage_core_start_offset,
    const tt_l1_ptr uint32_t* segment_args,
    uint32_t worker_core_stride_w_bytes,
    uint32_t storage_core_stride_w_bytes,
    uint32_t block_ht) {
    const uint32_t out_single_tile_size_bytes = dfb_out.get_tile_size();
    uint32_t args_idx = 0;
    uint32_t worker_core_read_offset = 0;

    const ShardView output_shards(output);

    uint32_t num_tiles_in_write_queue = 0;

    for (uint32_t i = 0; i < num_segments_to_write_back; ++i) {
        const uint32_t write_size = segment_args[args_idx++];
        const uint32_t storage_shard_id = segment_args[args_idx++];
        const auto storage_shard = output_shards.shard(storage_shard_id, noc.get_noc_id());

        const uint32_t num_tiles_to_write_in_current_segment = write_size / out_single_tile_size_bytes * block_ht;

        uint32_t src_offset = worker_core_read_offset;
        uint32_t dst_offset = 0;
        if (i == 0) {  // For the first segment we need to add the start offset; the following segments will start at 0
                       // offset
            dst_offset += storage_core_start_offset;
        }

        for (uint32_t h = 0; h < block_ht; ++h) {
            for (uint32_t w = 0; w < num_tiles_to_write_in_current_segment; ++w) {
                num_tiles_in_write_queue += 1;
                dfb_out.wait_front(static_cast<uint16_t>(num_tiles_in_write_queue));
                noc.async_write(
                    dfb_out,
                    storage_shard,
                    out_single_tile_size_bytes,
                    {.offset_bytes = src_offset},
                    {.offset_bytes = dst_offset});
                src_offset += out_single_tile_size_bytes;
                dst_offset += out_single_tile_size_bytes;
            }
            src_offset += worker_core_stride_w_bytes;
            dst_offset += storage_core_stride_w_bytes;
        }
        worker_core_read_offset += write_size;
    }
    noc.async_write_barrier();

    // The output tiles are waited cumulatively as they are written out to the storage cores. The
    // barrier above is the last use of them, so pop the whole queue to leave the buffer
    // balanced. (The sibling helper in rms_allgather waits and pops one tile at a time instead.)
    dfb_out.pop_front(static_cast<uint16_t>(num_tiles_in_write_queue));
}

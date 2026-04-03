// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

inline void write_resharded_data(
    experimental::Noc& noc,
    experimental::CircularBuffer& cb_out,
    experimental::CircularBuffer& cb_out_resharded,
    uint32_t num_segments_to_write_back,
    uint32_t storage_core_start_offset,
    tt_l1_ptr uint32_t* segment_args,
    uint32_t worker_core_stride_w_bytes,
    uint32_t storage_core_stride_w_bytes,
    uint32_t block_ht) {
    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out.get_cb_id());
    uint32_t args_idx = 0;
    uint32_t worker_core_read_offset = 0;

    experimental::UnicastEndpoint remote;

    uint32_t num_tiles_in_write_queue = 0;

    for (uint32_t i = 0; i < num_segments_to_write_back; ++i) {
        uint32_t write_size = segment_args[args_idx++];
        uint32_t storage_core_x = segment_args[args_idx++];
        uint32_t storage_core_y = segment_args[args_idx++];

        uint32_t num_tiles_to_write_in_current_segment = write_size / out_single_tile_size_bytes * block_ht;

        uint32_t src_offset = worker_core_read_offset;
        uint32_t dst_addr = cb_out_resharded.get_write_ptr();
        if (i == 0) {  // For the first segment we need to add the start offset; the following segments will start at 0
                       // offset
            dst_addr += storage_core_start_offset;
        }

        for (uint32_t h = 0; h < block_ht; ++h) {
            for (uint32_t w = 0; w < num_tiles_to_write_in_current_segment; ++w) {
                num_tiles_in_write_queue += 1;
                cb_out.wait_front(num_tiles_in_write_queue);
                noc.async_write(
                    cb_out,
                    remote,
                    out_single_tile_size_bytes,
                    {.offset_bytes = src_offset},
                    {.noc_x = storage_core_x, .noc_y = storage_core_y, .addr = dst_addr});
                src_offset += out_single_tile_size_bytes;
                dst_addr += out_single_tile_size_bytes;
            }
            src_offset += worker_core_stride_w_bytes;
            dst_addr += storage_core_stride_w_bytes;
        }
        worker_core_read_offset += write_size;
    }
    noc.async_write_barrier();
}

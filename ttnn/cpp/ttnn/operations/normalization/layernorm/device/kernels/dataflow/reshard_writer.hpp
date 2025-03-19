// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "dataflow_api.h"

inline void write_resharded_data(
    uint32_t cb_out,
    uint32_t cb_out_resharded,
    uint32_t num_segments_to_write_back,
    uint32_t storage_core_start_offset,
    tt_l1_ptr uint32_t* segment_args,
    uint32_t worker_core_stride_w_bytes,
    uint32_t storage_core_stride_w_bytes,
    uint32_t block_ht) {
    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);
    uint32_t args_idx = 0;
    uint32_t worker_core_read_offset = 0;

    uint32_t cb_out_read_base_addr = get_read_ptr(cb_out);
    uint32_t cb_out_reshard_write_base_addr = get_write_ptr(cb_out_resharded);

    uint32_t num_tiles_in_write_queue = 0;

    for (uint32_t i = 0; i < num_segments_to_write_back; ++i) {
        uint32_t write_size = segment_args[args_idx++];
        uint32_t storage_core_x = segment_args[args_idx++];
        uint32_t storage_core_y = segment_args[args_idx++];

        uint32_t num_tiles_to_write_in_current_segment = write_size / out_single_tile_size_bytes * block_ht;

        uint32_t worker_core_read_addr = cb_out_read_base_addr + worker_core_read_offset;
        uint32_t local_storage_core_write_addr = cb_out_reshard_write_base_addr;
        if (i == 0) {  // For the first segment we need to add the start offset; the following segments will start at 0
                       // offset
            local_storage_core_write_addr += storage_core_start_offset;
        }

        uint64_t remote_storage_core_write_addr =
            get_noc_addr(storage_core_x, storage_core_y, local_storage_core_write_addr);

        for (uint32_t h = 0; h < block_ht; ++h) {
            for (uint32_t w = 0; w < num_tiles_to_write_in_current_segment; ++w) {
                num_tiles_in_write_queue += 1;
                cb_wait_front(cb_out, num_tiles_in_write_queue);
                noc_async_write(worker_core_read_addr, remote_storage_core_write_addr, out_single_tile_size_bytes);
                worker_core_read_addr += out_single_tile_size_bytes;
                remote_storage_core_write_addr += out_single_tile_size_bytes;
            }
            worker_core_read_addr += worker_core_stride_w_bytes;
            remote_storage_core_write_addr += storage_core_stride_w_bytes;
        }
        worker_core_read_offset += write_size;
    }
    noc_async_write_barrier();
}

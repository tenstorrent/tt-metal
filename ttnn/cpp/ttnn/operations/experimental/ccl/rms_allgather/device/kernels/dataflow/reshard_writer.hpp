// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

template <
    uint32_t cb_out,
    uint32_t cb_out_resharded,
    uint32_t worker_core_stride_w_bytes,
    uint32_t storage_core_stride_w_bytes>
inline void write_minimal_resharded_data(
    Noc& noc,
    uint32_t num_segments_to_write_back,
    uint32_t storage_core_start_offset,
    tt_l1_ptr uint32_t* segment_args) {
    DataflowBuffer cb_out_obj(cb_out);
    DataflowBuffer cb_out_resharded_obj(cb_out_resharded);
    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);
    uint32_t args_idx = 0;
    uint32_t worker_core_read_offset = 0;

    uint32_t cb_out_read_base_addr = cb_out_obj.get_read_ptr();
    uint32_t cb_out_reshard_write_base_addr = cb_out_resharded_obj.get_write_ptr();

    for (uint32_t i = 0; i < num_segments_to_write_back; ++i) {
        uint32_t write_size = segment_args[args_idx++];
        uint32_t storage_core_x = segment_args[args_idx++];
        uint32_t storage_core_y = segment_args[args_idx++];

        uint32_t num_tiles_to_write_in_current_segment = write_size;

        uint32_t worker_core_read_addr = cb_out_read_base_addr + worker_core_read_offset;
        uint32_t local_storage_core_write_addr = cb_out_reshard_write_base_addr;
        if (i == 0) {  // For the first segment we need to add the start offset; the following segments will start at 0
                       // offset
            local_storage_core_write_addr += storage_core_start_offset;
        }

        for (uint32_t w = 0; w < num_tiles_to_write_in_current_segment; ++w) {
            cb_out_obj.wait_front(1);
            noc.async_write(
                CoreLocalMem<uint8_t>(worker_core_read_addr),
                UnicastEndpoint{},
                out_single_tile_size_bytes,
                {},
                {.noc_x = storage_core_x, .noc_y = storage_core_y, .addr = local_storage_core_write_addr});
            worker_core_read_addr += out_single_tile_size_bytes;
            local_storage_core_write_addr += out_single_tile_size_bytes;
            cb_out_obj.pop_front(1);
        }
        worker_core_read_addr += worker_core_stride_w_bytes;
        local_storage_core_write_addr += storage_core_stride_w_bytes;
        worker_core_read_offset += write_size * out_single_tile_size_bytes;
    }
    noc.async_write_barrier();
}

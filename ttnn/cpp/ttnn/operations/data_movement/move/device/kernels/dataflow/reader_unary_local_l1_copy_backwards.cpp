// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t total_size_bytes = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t num_chunks = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t chunk_size_bytes = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t remainder_chunk_size_bytes = get_arg_val<uint32_t>(i);
    i += 1;
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);

    Noc noc;
    CircularBuffer src_cb(src_cb_id);
    CircularBuffer dst_cb(dst_cb_id);

    uint32_t src_cb_base_addr = src_cb.get_read_ptr();
    uint32_t dst_cb_base_addr = dst_cb.get_write_ptr();

    // Copy from top of src cb to top of dst cb (backwards)
    uint32_t src_cb_addr = src_cb_base_addr + total_size_bytes;
    uint32_t dst_cb_addr = dst_cb_base_addr + total_size_bytes;
    for (uint32_t i = 0; i < num_chunks; i += 1) {
        src_cb_addr -= chunk_size_bytes;
        dst_cb_addr -= chunk_size_bytes;
        CoreLocalMem<uint32_t> dst(dst_cb_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            chunk_size_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_cb_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
    }
    if (remainder_chunk_size_bytes > 0) {
        src_cb_addr -= remainder_chunk_size_bytes;
        dst_cb_addr -= remainder_chunk_size_bytes;
        CoreLocalMem<uint32_t> dst(dst_cb_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            remainder_chunk_size_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_cb_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
    }
}

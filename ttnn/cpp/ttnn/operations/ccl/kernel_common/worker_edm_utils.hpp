// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

using ttnn::ccl::ShardType;
using ttnn::ccl::WorkerXY;
// using ttnn::ccl::coord_t;

namespace ttnn {
namespace ccl {
static FORCE_INLINE coord_t coord_from_args(std::size_t& arg_idx) {
    uint32_t x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t y = get_arg_val<uint32_t>(arg_idx++);
    return coord_t(x, y);
}

enum EDM_IO_BLOCKING_MODE { BLOCKING, NON_BLOCKING };

}  // namespace ccl
}  // namespace ttnn

FORCE_INLINE void push_filler_pages_to_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_reserve_back(cb_id, num_pages);
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void pop_filler_pages_from_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_wait_front(cb_id, num_pages);
    cb_pop_front(cb_id, num_pages);
}

FORCE_INLINE void fetch_chunk(const uint32_t& cb_id,
                              const uint32_t& num_pages,
                              const uint32_t& page_size,
                              uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, page_size * num_pages);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

template <ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode = ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>
FORCE_INLINE void send_chunk_from_address(const uint32_t& local_l1_address,
                                          const uint32_t& num_pages,
                                          const uint32_t& page_size,
                                          uint64_t remote_l1_write_addr) {
    noc_async_write(local_l1_address, remote_l1_write_addr, page_size * num_pages);
    if constexpr (blocking_mode == ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier();
    }
}

template <ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode = ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>
FORCE_INLINE void send_chunk(const uint32_t& cb_id,
                             const uint32_t& num_pages,
                             const uint32_t& page_size,
                             uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    if constexpr (blocking_mode == ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier();
        cb_pop_front(cb_id, num_pages);
    }
}

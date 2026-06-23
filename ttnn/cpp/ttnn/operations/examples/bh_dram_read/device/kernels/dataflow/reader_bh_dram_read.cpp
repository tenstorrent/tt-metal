// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Reader-only kernel. One instance runs per DRAM bank. It reads the pages of
// the input tensor that reside in this core's assigned bank into a circular
// buffer and immediately discards them (no compute, no writer).
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);  // this core's bank id
    const uint32_t stride = get_arg_val<uint32_t>(3);    // number of DRAM banks

    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t onepage = 1;

    // Page size from the CB interface (works for TILE and ROW_MAJOR layouts).
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer cb(cb_id_in0);

    // page_ids {start_id, start_id + stride, ...} all map to this core's bank
    // under interleaved addressing.
    uint32_t page_id = start_id;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb.reserve_back(onepage);
        noc.async_read(s, cb, page_bytes, {.page_id = page_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(onepage);
        // Reader-only: discard immediately so the CB never fills.
        cb.wait_front(onepage);
        cb.pop_front(onepage);
        page_id += stride;
    }
}

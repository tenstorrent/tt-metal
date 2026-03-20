// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t onepage = 1;
    constexpr auto cb_id_dst = tt::CBIndex::c_2;

    experimental::Noc noc;
    experimental::CircularBuffer cb_dst(cb_id_dst);

#if DST_SHARDED
    cb_dst.wait_front(num_pages);
#else
    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const uint32_t page_bytes = get_local_cb_interface(cb_id_dst).fifo_page_size;
    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_dst.wait_front(onepage);
        noc.async_write(cb_dst, dst, page_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb_dst.pop_front(onepage);
    }
#endif
}

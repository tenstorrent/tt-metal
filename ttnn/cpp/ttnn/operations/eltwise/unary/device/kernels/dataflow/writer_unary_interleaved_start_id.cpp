// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_out);

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}

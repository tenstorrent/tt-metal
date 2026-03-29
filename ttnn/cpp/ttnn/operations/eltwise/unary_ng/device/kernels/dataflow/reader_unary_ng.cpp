// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_id_src = tt::CBIndex::c_0;

    experimental::Noc noc;
    experimental::CircularBuffer cb_src(cb_id_src);

#if SRC_SHARDED
    cb_src.reserve_back(num_pages);
    cb_src.push_back(num_pages);
#else
    constexpr uint32_t onepage = 1;
    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    const uint32_t page_bytes = get_local_cb_interface(cb_id_src).fifo_page_size;
    const auto src = TensorAccessor(src_args, src_addr, page_bytes);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_src.reserve_back(onepage);
        noc.async_read(src, cb_src, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_src.push_back(onepage);
    }
#endif
}

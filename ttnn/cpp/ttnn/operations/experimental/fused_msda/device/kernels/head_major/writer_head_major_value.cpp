// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stores each untilized block as one page of the head-major value.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t out_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(1);
constexpr uint32_t PAGE_NBYTES = get_compile_time_arg_val(2);  // 32 * D * 2
constexpr auto out_args = TensorAccessorArgs<3>();

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, out_addr, PAGE_NBYTES);
    Noc noc;
    CircularBuffer out_cb(out_cb_index);

    for (uint32_t p = first_page; p < first_page + num_pages; ++p) {
        out_cb.wait_front(TILES_PER_PAGE);
        noc.async_write(
            CoreLocalMem<uint32_t>(out_cb.get_read_ptr()),
            out_acc,
            PAGE_NBYTES,
            {.offset_bytes = 0},
            {.page_id = p, .offset_bytes = 0});
        noc.async_writes_flushed();
        out_cb.pop_front(TILES_PER_PAGE);
    }
    noc.async_write_barrier();
}

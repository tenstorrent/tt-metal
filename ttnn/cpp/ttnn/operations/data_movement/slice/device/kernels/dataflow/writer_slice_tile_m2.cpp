// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 slice tile writer. A slice-local copy of the interleaved path of
// writer_unary_interleaved_start_id.cpp (that kernel is shared by ~10 other ops
// and must stay on the legacy arg model). Bindings are Metal 2.0:
//   - CB index           -> dfb::cb_out
//   - output accessor    -> ta::dst   (address implicit; no dst_addr RTA)
//   - num_pages/start_id -> named RTAs

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t cb_id_out = dfb::cb_out;

    CircularBuffer cb_out(cb_id_out);
    const uint32_t page_bytes = cb_out.get_tile_size();
    Noc noc;

    const auto s = TensorAccessor(ta::dst);

    constexpr uint32_t onepage = 1;
    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out.wait_front(onepage);
        noc.async_write(cb_out, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        cb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
}

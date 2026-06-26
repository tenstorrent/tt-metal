// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local writer for transpose's tiled WH factory. The device-side NoC + TensorAccessor
// logic is unchanged from the legacy eltwise/unary writer; only the resource bindings move to the
// Metal 2.0 namespaces (dfb::/tensor::/args::).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t page_size = get_arg(args::page_size);

    Noc noc;
    DataflowBuffer cb(dfb::cb_out0);

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::dst);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}

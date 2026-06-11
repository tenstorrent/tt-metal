// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the CN interleaved writer (op-private copy). Only the binding mechanism changed:
// the CB id comes from the DFB token (dfb::), the output address from the TensorAccessor binding (ta::),
// page-format compile-time scalars from named compile-time args (args::, constexpr), and the per-core
// work descriptors from named runtime args (args::). The write loop is preserved unchanged.
// CN_RM continues to select the sharded-aware multi-page split helper.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t write_size = get_arg(args::write_size);

    constexpr uint32_t cb_id_out = dfb::src0;

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::dst_args);

    Noc noc;
    CircularBuffer cb(cb_id_out);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(onepage);
#ifdef CN_RM
        // Restored native sharded multi-page split (see common.hpp helper).
        // RM-only: see reader kernel for rationale; TILE-layout below uses original API.
        const uint32_t cb_read_ptr = cb.get_read_ptr();
        tt::data_movement::common::noc_async_write_sharded(cb_read_ptr, s, i, 0, write_size);
#else
        noc.async_write(cb, s, page_size, {.offset_bytes = 0}, {.page_id = i});
#endif
        noc.async_write_barrier();
        cb.pop_front(onepage);
    }
}

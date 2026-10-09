// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar unary reader: fork of eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp
// (the Metal 2.0 fork of reader_unary_interleaved_start_id.cpp), minus the unused BACKWARDS walk.
// Reads num_pages consecutive tiles, starting at page start_id, from the input tensor into the "in" DFB, one
// tile per reserve/push. TensorAccessor(tensor::src) resolves a page id to its bank (interleaved) or shard
// core (sharded), so the same code serves both. The DFB is bound with implicit sync disabled, so
// reserve_back/push_back are the credit handshake with the compute kernel.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb(dfb::in);
    // Read the page size from the DFB object, after constructing it.
    const uint32_t page_bytes = dfb.get_entry_size();
    constexpr uint32_t onepage = 1;

    const auto src = TensorAccessor(tensor::src);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb.reserve_back(onepage);
        noc.async_read(src, dfb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(onepage);
    }
}

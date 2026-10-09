// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar unary writer: fork of eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp
// (the Metal 2.0 fork of writer_unary_interleaved_start_id.cpp), minus the unused OUT_SHARDED / BACKWARDS
// variants. Drains num_pages tiles from the "out" DFB to the output tensor, starting at page start_id.
// TensorAccessor(tensor::dst) resolves a page id to its bank (interleaved) or shard core (sharded). The DFB is
// bound with implicit sync disabled, so wait_front/pop_front are the credit handshake with the compute kernel.

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
    DataflowBuffer dfb(dfb::out);
    const uint32_t page_bytes = dfb.get_entry_size();
    constexpr uint32_t onepage = 1;

    const auto dst = TensorAccessor(tensor::dst);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb.wait_front(onepage);
        noc.async_write(dfb, dst, page_bytes, {}, {.page_id = i});
        // The tile has left the DFB slot once the write is flushed, so the slot can go back to the producer.
        noc.async_writes_flushed();
        dfb.pop_front(onepage);
    }
    noc.async_write_barrier();
}

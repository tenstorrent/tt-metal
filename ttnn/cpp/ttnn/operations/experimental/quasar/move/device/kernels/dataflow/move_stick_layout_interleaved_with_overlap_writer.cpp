// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 cross-kernel split of move_stick_layout_interleaved_with_overlap.cpp (row-major /
// stick-layout interleaved overlap move). The reader stages src -> the scratch DFB and, only after
// the all-cores read handshake, push_back()s it; this writer consumes the scratch DFB (wait_front)
// and drains it to dst. Splitting the producer (reader) and consumer (writer) across two kernels
// avoids a DM-kernel self-loop on dfb::scratch, which Metal 2.0 forbids. Because the reader's
// push_back happens only after the handshake, this wait_front cannot unblock until every core has
// finished reading src, preserving the overlap-safety invariant (no core writes dst until all cores
// have read src).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_pages = get_arg(args::num_pages);
    uint32_t aligned_page_size = get_arg(args::aligned_page_size);

    constexpr uint32_t page_size = get_arg(args::page_size);

    Noc noc;
    DataflowBuffer cb(dfb::scratch);

    const auto dst_addrgen = TensorAccessor(tensor::output);

    // Drain the staged pages (CB -> dst). wait_front unblocks only after the reader's
    // post-handshake push_back, i.e. after all cores have finished reading src.
    cb.wait_front(num_pages);
    uint32_t l1_read_addr = cb.get_read_ptr();
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        CoreLocalMem<uint32_t> src(l1_read_addr);
        noc.async_write(src, dst_addrgen, page_size, {.offset_bytes = 0}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        l1_read_addr += aligned_page_size;
    }
    cb.pop_front(num_pages);
}

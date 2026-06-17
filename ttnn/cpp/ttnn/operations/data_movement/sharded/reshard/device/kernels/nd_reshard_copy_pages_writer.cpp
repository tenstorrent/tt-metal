// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reshard copy-pages consumer.
// Pops entries from the DFB bound via dfb::reshard_dfb and writes them as pages
// [start_page, end_page) to the output tensor via TensorAccessor(ta::output). The base address
// and sharding layout come from the TensorBinding (filled from the TensorArgument at enqueue).
//
// Runtime args (positional varargs):
//   arg 0: start_page
//   arg 1: end_page

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    TensorAccessor accessor(ta::output);
    DataflowBuffer buf(dfb::reshard_dfb);
    const uint32_t entry_size = buf.get_entry_size();

    for (uint32_t page_id = start_page; page_id < end_page; page_id++) {
        buf.wait_front(1);
        uint64_t dst_noc_addr = accessor.get_noc_addr(page_id);
        noc_async_write(buf.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        buf.pop_front(1);
    }
}

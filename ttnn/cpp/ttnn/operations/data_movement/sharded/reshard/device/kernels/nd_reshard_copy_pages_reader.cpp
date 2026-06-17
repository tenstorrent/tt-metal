// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 reshard copy-pages producer.
// Reads pages [start_page, end_page) from the input tensor via TensorAccessor(ta::input)
// and pushes them entry-by-entry into the DFB bound via dfb::reshard_dfb. The base address
// and sharding layout come from the TensorBinding (filled from the TensorArgument at enqueue).
//
// Runtime args (positional varargs):
//   arg 0: start_page
//   arg 1: end_page

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    TensorAccessor accessor(ta::input);
    DataflowBuffer buf(dfb::reshard_dfb);
    const uint32_t entry_size = buf.get_entry_size();

    for (uint32_t page_id = start_page; page_id < end_page; page_id++) {
        buf.reserve_back(1);
        uint64_t src_noc_addr = accessor.get_noc_addr(page_id);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        buf.push_back(1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scratch variant: writes faces 0 to 2 of each reduced 32x32 output page in one write. Face 3 is tile padding
// and are not written.

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
    const auto s = TensorAccessor(tensor::dst);
    const uint32_t face_bytes = dfb.get_entry_size() / 4;

    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        dfb.wait_front(1);
        noc.async_write(dfb, s, 3 * face_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}

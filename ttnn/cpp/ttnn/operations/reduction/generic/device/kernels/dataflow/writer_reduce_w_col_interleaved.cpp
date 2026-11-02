// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes faces 0 and 2 of each reduced 32x32 output page, the faces that hold column 0. Faces 1 and 3 are tile padding
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
        noc.async_write(dfb, s, face_bytes, {}, {.page_id = i});
        noc.async_write(
            dfb, s, face_bytes, {.offset_bytes = 2 * face_bytes}, {.page_id = i, .offset_bytes = 2 * face_bytes});
        noc.async_writes_flushed();
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // output
    uint32_t output_stick_size = get_arg(args::output_stick_size);

    // etc
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_sticks = get_arg(args::num_sticks);

    // The aligned page size of a row-major stick rides the binding token, baked in when the program is
    // built. A row of a different width is a different program-cache key, so the program is rebuilt
    // rather than reused with a stale page size.
    const auto s0 = TensorAccessor(tensor::s0);

    Noc noc;
    // The output stick is drained straight out of the buffer the reader staged it in.
    DataflowBuffer dfb_out_obj(dfb::out);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out_obj.wait_front(1);
        noc.async_write(dfb_out_obj, s0, output_stick_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out_obj.pop_front(1);
    }
}

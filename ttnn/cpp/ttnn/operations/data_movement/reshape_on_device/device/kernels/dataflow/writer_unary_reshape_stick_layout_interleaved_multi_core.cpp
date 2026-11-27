// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    uint32_t num_read_per_barrier = get_arg(args::num_read_per_barrier);
    uint32_t num_sticks_per_dfb_push = get_arg(args::num_sticks_per_dfb_push);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t new_stick_size = get_arg(args::new_stick_size);

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_output(dfb::out0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        dfb_output.wait_front(num_sticks_per_dfb_push);
        uint32_t dfb_read_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_write(
                dfb_output,
                s,
                new_stick_size,
                {.offset_bytes = dfb_read_offset},
                {.page_id = i_stick, .offset_bytes = 0});
            dfb_read_offset += new_stick_size;
            i_stick += 1;
        }
        noc.async_write_barrier();
        dfb_output.pop_front(num_sticks_per_dfb_push);
    }
}

// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
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

    constexpr uint32_t old_stick_size = get_arg(args::old_stick_size);

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_input(dfb::in0);

    uint32_t i_stick = start_id;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        dfb_input.reserve_back(num_sticks_per_dfb_push);
        uint32_t dfb_write_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_read(
                s,
                dfb_input,
                old_stick_size,
                {.page_id = i_stick, .offset_bytes = 0},
                {.offset_bytes = dfb_write_offset});
            dfb_write_offset += old_stick_size;
            i_stick++;
        }
        noc.async_read_barrier();
        dfb_input.push_back(num_sticks_per_dfb_push);
    }
}

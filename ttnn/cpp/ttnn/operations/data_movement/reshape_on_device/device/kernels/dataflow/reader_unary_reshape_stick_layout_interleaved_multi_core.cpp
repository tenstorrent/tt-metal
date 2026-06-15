// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t old_stick_size = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    constexpr auto cb_in0 = tt::CBIndex::c_0;

    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer cb_input(cb_in0);

    uint32_t i_stick = start_id;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_input.reserve_back(num_sticks_per_cb_push);
        uint32_t cb_write_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_read(
                s,
                cb_input,
                old_stick_size,
                {.page_id = i_stick, .offset_bytes = 0},
                {.offset_bytes = cb_write_offset});
            cb_write_offset += old_stick_size;
            i_stick++;
        }
        noc.async_read_barrier();
        cb_input.push_back(num_sticks_per_cb_push);
    }
}

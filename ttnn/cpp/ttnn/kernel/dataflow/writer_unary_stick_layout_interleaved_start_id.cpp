// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr auto dst0_args = TensorAccessorArgs<2>();

    const auto s0 = TensorAccessor(dst0_args, dst_addr);

    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_sticks;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_out0.wait_front(1);
        noc.async_write(cb_out0, s0, stick_size, {.offset_bytes = 0}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        cb_out0.pop_front(1);
    }
}

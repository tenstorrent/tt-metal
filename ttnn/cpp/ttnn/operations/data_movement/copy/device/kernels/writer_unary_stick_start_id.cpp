// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_shards = get_arg_val<uint32_t>(4);

    constexpr uint32_t dfb_id_out0 = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    DataflowBuffer dfb_out0(dfb_id_out0);

    Noc noc;
    const auto s0 = TensorAccessor(dst_args, dst_addr);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_sticks;
    for (uint32_t i = start_id; i != end_id; --i) {
        for (uint32_t k = num_shards - 1; k >= 0; k--) {
#else
    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        for (uint32_t k = 0; k < num_shards; k++) {
#endif
            uint32_t stick_index = i * num_shards + k;
            dfb_out0.wait_front(1);
            noc.async_write(dfb_out0, s0, stick_size, {.offset_bytes = 0}, {.page_id = stick_index, .offset_bytes = 0});
            noc.async_write_barrier();
            dfb_out0.pop_front(1);
        }
    }
}

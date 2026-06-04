// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_height = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const uint32_t input_width_offset_bytes = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto s0 = TensorAccessor(dst_args, dst_addr + input_width_offset_bytes);

    Noc noc;
    CircularBuffer cb_out(cb_id_out0);

    uint32_t stick_id = start_id;
    cb_out.wait_front(block_height);
    uint32_t cb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            cb_out, s0, block_width_bytes, {.offset_bytes = cb_read_offset}, {.page_id = stick_id, .offset_bytes = 0});
        stick_id++;
        cb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_height);
}

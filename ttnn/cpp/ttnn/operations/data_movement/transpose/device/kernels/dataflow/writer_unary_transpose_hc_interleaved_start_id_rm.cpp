// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(1);

    const uint32_t stick_size_bytes = W_size_bytes;

    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_out0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb.wait_front(num_read_per_barrier);
        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_write(
                cb, s, stick_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = i_stick, .offset_bytes = 0});
            l1_read_offset += stick_size_bytes;
            i_stick += 1;
        }
        noc.async_write_barrier();
        cb.pop_front(num_read_per_barrier);
    }
}

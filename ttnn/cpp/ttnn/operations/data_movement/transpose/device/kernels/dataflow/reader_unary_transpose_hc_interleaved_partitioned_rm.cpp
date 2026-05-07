// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t curr_c = get_arg_val<uint32_t>(4);
    uint32_t curr_h = get_arg_val<uint32_t>(5);
    uint32_t curr_n = get_arg_val<uint32_t>(6);

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(3);

    constexpr uint32_t CH = C * H;

    constexpr auto cb_in0 = tt::CBIndex::c_0;

    const uint32_t stick_size_bytes = W_size_bytes;

    constexpr auto src_args = TensorAccessorArgs<5>();
    const auto s = TensorAccessor(src_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_in0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb.reserve_back(num_read_per_barrier);
        uint32_t l1_write_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_read(
                s, cb, stick_size_bytes, {.page_id = i_stick, .offset_bytes = 0}, {.offset_bytes = l1_write_offset});
            l1_write_offset += stick_size_bytes;

            curr_c++;
            i_stick += H;
            if (curr_c == C) {  // end of channel dim
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {  // end of H dim
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    i_stick = i_stick - H + 1;
                } else {
                    i_stick = i_stick - CH + 1;
                }
            }
        }
        noc.async_read_barrier();
        cb.push_back(num_read_per_barrier);
    }
}

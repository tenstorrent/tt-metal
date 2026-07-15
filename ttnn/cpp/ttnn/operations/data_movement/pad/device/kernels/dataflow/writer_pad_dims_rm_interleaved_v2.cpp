// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_padded_aligned = get_compile_time_arg_val(2);
    constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(3);
    constexpr uint32_t accessor_page_size = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    const auto s = TensorAccessor(dst_args, dst_addr, accessor_page_size);
    Noc noc;
    CircularBuffer cb_out0_exp(cb_out0);

    uint32_t i_page = start_page_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_out0_exp.wait_front(num_sticks_per_barrier);

        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            if (num_output_pages_in_row == 1) {
                // Width fits in a single page: index the accessor with the flat page id directly.
                // `noc_async_write_sharded` derives pages-per-row from the (rank-squeezed) dspec
                // shape, which is wrong when an outer dim is sharded and the width is a single page.
                noc.async_write(
                    CoreLocalMem<uint32_t>(cb_out0_exp.get_read_ptr() + l1_read_offset),
                    s,
                    stick_size_bytes,
                    {},
                    {.page_id = i_page, .offset_bytes = 0});
            } else {
                const uint32_t stick_id = i_page / num_output_pages_in_row;
                tt::data_movement::common::noc_async_write_sharded(
                    noc,
                    cb_out0_exp.get_read_ptr() + l1_read_offset,
                    s,
                    stick_id,
                    /*offset=*/0,
                    /*size=*/stick_size_bytes);
            }
            l1_read_offset += stick_size_padded_aligned;
            i_page += num_output_pages_in_row;
        }
        noc.async_write_barrier();
        cb_out0_exp.pop_front(num_sticks_per_barrier);
    }
}

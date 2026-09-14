// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_sticks_per_core = get_arg(args::num_sticks_per_core);
    auto num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    auto start_page_id = get_arg(args::start_page_id);

    constexpr auto stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr auto stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);
    constexpr auto num_output_pages_in_row = get_arg(args::num_output_pages_in_row);

    const auto s = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer dfb_out0_exp(dfb::out0);

    uint32_t i_page = start_page_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        dfb_out0_exp.wait_front(num_sticks_per_barrier);

        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            if (num_output_pages_in_row == 1) {
                // Width fits in a single page: index the accessor with the flat page id directly.
                // `noc_async_write_sharded` derives pages-per-row from the (rank-squeezed) dspec
                // shape, which is wrong when an outer dim is sharded and the width is a single page.
                noc.async_write(
                    CoreLocalMem<uint32_t>(dfb_out0_exp.get_read_ptr() + l1_read_offset),
                    s,
                    stick_size_bytes,
                    {},
                    {.page_id = i_page, .offset_bytes = 0});
            } else {
                const uint32_t stick_id = i_page / num_output_pages_in_row;
                tt::data_movement::common::noc_async_write_sharded(
                    noc,
                    dfb_out0_exp.get_read_ptr() + l1_read_offset,
                    s,
                    stick_id,
                    /*offset=*/0,
                    /*size=*/stick_size_bytes);
            }
            l1_read_offset += stick_size_padded_aligned;
            i_page += num_output_pages_in_row;
        }
        noc.async_write_barrier();
        dfb_out0_exp.pop_front(num_sticks_per_barrier);
    }
}

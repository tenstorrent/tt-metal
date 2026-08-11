// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of pad's RM multicore-default writer (private to
// PadRmReaderWriterMultiCoreDefaultProgramFactory). The device-side NoC + TensorAccessor logic is
// unchanged; only the resource access is migrated to the Metal 2.0 named handles (dfb::/tensor::/args::).
//   - c_0 padded stream -> dfb::cb_out0 (CONSUMER of the reader's cb_in0)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const uint32_t num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    const uint32_t start_page_id = get_arg(args::start_page_id);

    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);
    constexpr uint32_t num_output_pages_in_row = get_arg(args::num_output_pages_in_row);
    constexpr uint32_t output_page_size = get_arg(args::output_page_size);
    constexpr uint32_t size_of_valid_data_in_last_output_page_in_row =
        get_arg(args::size_of_valid_data_in_last_output_page_in_row);

    const auto s = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer cb_out0(dfb::cb_out0);

    uint32_t i_page = start_page_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_out0.wait_front(num_sticks_per_barrier);

        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            uint32_t tmp_offset = l1_read_offset;
            for (uint32_t p = 0; p < num_output_pages_in_row - 1; p++) {
                noc.async_write(
                    cb_out0,
                    s,
                    output_page_size,
                    {.offset_bytes = tmp_offset},
                    {.page_id = i_page + p, .offset_bytes = 0});
                tmp_offset += output_page_size;
            }
            noc.async_write(
                cb_out0,
                s,
                size_of_valid_data_in_last_output_page_in_row,
                {.offset_bytes = tmp_offset},
                {.page_id = i_page + num_output_pages_in_row - 1, .offset_bytes = 0});
            l1_read_offset += stick_size_padded_aligned;
            i_page += num_output_pages_in_row;
        }
        noc.async_write_barrier();
        cb_out0.pop_front(num_sticks_per_barrier);
    }
}

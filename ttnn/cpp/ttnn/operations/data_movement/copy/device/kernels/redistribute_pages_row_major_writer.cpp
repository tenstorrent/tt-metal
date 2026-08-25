// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

FORCE_INLINE std::uint32_t u32_min(std::uint32_t a, std::uint32_t b) { return (a < b) ? a : b; }

void kernel_main() {
    // run-time args
    const auto start_row = get_arg(args::start_row);
    const auto num_rows_to_process = get_arg(args::num_rows_to_process);

    // compile-time args
    constexpr auto num_output_pages_in_row = get_arg(args::num_output_pages_in_row);
    constexpr auto elements_per_output_page = get_arg(args::elements_per_output_page);
    constexpr auto bytes_per_element = get_arg(args::bytes_per_element);
    constexpr auto elements_per_tensor_row = get_arg(args::elements_per_tensor_row);
    constexpr auto bytes_per_output_subblock = get_arg(args::bytes_per_output_subblock);

    Noc noc;
    // dfb::in1 is the reader->writer output-page FIFO (this kernel is the consumer).
    DataflowBuffer dfb_in1(dfb::in1);

    const auto accessor_dst = TensorAccessor(tensor::dst);

    const std::uint32_t elements_per_output_subblock = bytes_per_output_subblock / bytes_per_element;

    for (std::uint32_t row = start_row; row < start_row + num_rows_to_process; ++row) {
        std::uint32_t output_column = 0;
        while (output_column < elements_per_tensor_row) {
            std::uint32_t next_output_page_end_column =
                output_column + (elements_per_output_page - (output_column % elements_per_output_page) - 1);
            std::uint32_t output_end_column =
                u32_min(output_column + elements_per_output_subblock - 1, elements_per_tensor_row - 1);
            output_end_column = u32_min(
                next_output_page_end_column,
                output_end_column);  // output end column should be the minimum of the next output page end column,
                                     // the end column of the next output subblock and the end of the tensor row

            std::uint32_t output_page_id = row * num_output_pages_in_row + (output_column / elements_per_output_page);
            std::uint32_t output_addr_subblock_offset =
                ((output_column % elements_per_output_page) / elements_per_output_subblock) * bytes_per_output_subblock;
            std::uint32_t num_bytes_to_write = (output_end_column - output_column + 1) * bytes_per_element;
            dfb_in1.wait_front(1);
            noc.async_write(
                dfb_in1,
                accessor_dst,
                num_bytes_to_write,
                {.offset_bytes = 0},
                {.page_id = output_page_id, .offset_bytes = output_addr_subblock_offset});
            noc.async_write_barrier();
            dfb_in1.pop_front(1);
            output_column = output_end_column + 1;
        }
    }
}

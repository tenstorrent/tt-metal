// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/kernel_args.h"

FORCE_INLINE std::uint32_t u32_min(std::uint32_t a, std::uint32_t b) { return (a < b) ? a : b; }

void kernel_main() {
    // run-time args
    const auto start_row = get_arg(args::start_row);
    const auto num_rows_to_process = get_arg(args::num_rows_to_process);

    // compile-time args
    // num_output_pages_in_row is declared but unused in this kernel; carried forward
    // from the legacy CTA list unchanged to preserve behavior (an ops-team prune candidate).
    constexpr auto num_output_pages_in_row = get_arg(args::num_output_pages_in_row);
    constexpr auto num_input_pages_in_row = get_arg(args::num_input_pages_in_row);
    constexpr auto elements_per_output_page = get_arg(args::elements_per_output_page);
    constexpr auto bytes_per_element = get_arg(args::bytes_per_element);
    constexpr auto elements_per_input_page = get_arg(args::elements_per_input_page);
    constexpr auto elements_per_tensor_row = get_arg(args::elements_per_tensor_row);
    constexpr auto bytes_per_input_subblock = get_arg(args::bytes_per_input_subblock);
    constexpr auto bytes_per_output_subblock = get_arg(args::bytes_per_output_subblock);

    Noc noc;
    // dfb::in0 is a reader-private L1 scratchpad (self-loop): the reader both fills and
    // drains it. dfb::in1 is the reader->writer output-page FIFO.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    const auto accessor_src = TensorAccessor(tensor::src);

    const std::uint32_t elements_per_output_subblock = bytes_per_output_subblock / bytes_per_element;
    const std::uint32_t elements_per_input_subblock = bytes_per_input_subblock / bytes_per_element;
    dfb_in0.reserve_back(1);

    // To help understand the logic of this kernel, here is a visualization of what a subblock looks like in the
    // input/output tensor: When the tensor page is not too large (i.e., does not cause a DFB OOM error), the subblock
    // size = the width of the page, and one row of the tensor looks like:
    // | page_1 | page_2 | page_3 | page_4 | ... | page_n |
    //
    // ----------------------------------------------------------------------------------------------------------
    // In cases where we have a tensor with massive rows and massive page sizes, the subblock size will be capped at
    // 65536 * 4 bytes, which will be smaller than the page size:
    // |         page_1         |       |         page_2         | ...
    // |subblock_1|subblock_2|subblock_3|subblock_4|subblock_5|subblock_6| ...
    // Note that in the above diagram, we read in a full page one subblock at a time,
    // and no subblock spans multiple pages. That is, if our page size is not divisible by our max subblock size of
    // 65536 * 4 bytes, then the last subblock read from that page will contain less data than the subblock size
    // (i.e., the overlapping region of subblock_3 and page_1 in the diagram above).
    // Thus, the start of a page will always align with the start of a subblock. This is required to guarantee
    // aligned noc reads/writes.

    const std::uint32_t input_l1_write_addr = dfb_in0.get_write_ptr();

    for (std::uint32_t row = start_row; row < start_row + num_rows_to_process; ++row) {
        std::uint32_t input_start_column = 0;
        std::uint32_t input_end_column = input_start_column + elements_per_input_subblock - 1;
        std::uint32_t output_start_column = 0;
        std::uint32_t output_end_column = output_start_column + elements_per_output_subblock - 1;
        while (input_start_column < elements_per_tensor_row) {
            if (input_start_column >= output_start_column) {  // We need to read in a new input subblock
                std::uint32_t input_page_id =
                    row * num_input_pages_in_row + (input_start_column / elements_per_input_page);
                std::uint32_t input_subblock_offset =
                    ((input_start_column % elements_per_input_page) / elements_per_input_subblock) *
                    bytes_per_input_subblock;
                std::uint32_t num_bytes_to_read = (input_end_column - input_start_column + 1) * bytes_per_element;
                CoreLocalMem<std::uint32_t> dst(input_l1_write_addr);
                noc.async_read(
                    accessor_src,
                    dst,
                    num_bytes_to_read,
                    {.page_id = input_page_id, .offset_bytes = input_subblock_offset},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
            }
            if (input_end_column >= output_end_column) {  // Case where we are finishing writing an output subblock
                std::uint32_t bytes_to_write_to_output_subblock;
                std::uint32_t l1_output_subblock_write_addr_offset;
                std::uint32_t l1_input_subblock_read_addr_offset;
                if (output_start_column >= input_start_column) {
                    dfb_in1.reserve_back(
                        1);  // We are writing a new output subblock, so we need to reserve a slot on the output DFB
                    bytes_to_write_to_output_subblock =
                        (output_end_column - output_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset = 0;
                    l1_input_subblock_read_addr_offset =
                        (output_start_column - input_start_column) *
                        bytes_per_element;  // part of the input subblock was already read in previous iterations
                } else {
                    bytes_to_write_to_output_subblock =
                        (output_end_column - input_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset =
                        (input_start_column - output_start_column) *
                        bytes_per_element;  // part of the output subblock was already written in previous iterations
                    l1_input_subblock_read_addr_offset = 0;
                }

                std::uint32_t l1_output_subblock_write_addr =
                    dfb_in1.get_write_ptr();  // write the output subblock to the output DFB
                tt::data_movement::common::tt_memmove<false, false, true, 0>(
                    noc,
                    l1_output_subblock_write_addr + l1_output_subblock_write_addr_offset,
                    input_l1_write_addr + l1_input_subblock_read_addr_offset,
                    bytes_to_write_to_output_subblock);

                if (input_end_column == output_end_column) {
                    // We have processed the entire input subblock, so we must update the start and end indices of the
                    // input subblock as well
                    input_start_column = input_end_column + 1;
                    std::uint32_t next_input_page_end_column =
                        input_start_column +
                        (elements_per_input_page - (input_start_column % elements_per_input_page) - 1);
                    input_end_column =
                        u32_min(input_start_column + elements_per_input_subblock - 1, elements_per_tensor_row - 1);
                    input_end_column = u32_min(
                        next_input_page_end_column,
                        input_end_column);  // input end column should be the minimum of the next input page end column,
                                            // the end column of the next input subblock and the end of the tensor row
                }
                // We have processed the entire output subblock, so we must update the start and end indices of the
                // output subblock
                output_start_column = output_end_column + 1;
                std::uint32_t next_output_page_end_column =
                    output_start_column +
                    (elements_per_output_page - (output_start_column % elements_per_output_page) - 1);
                output_end_column =
                    u32_min(output_start_column + elements_per_output_subblock - 1, elements_per_tensor_row - 1);
                output_end_column = u32_min(
                    next_output_page_end_column,
                    output_end_column);  // output end column should be the minimum of the next output page end column,
                                         // the end column of the next output subblock and the end of the tensor row
                // We have processed the entire output subblock, so we must commit it to the output DFB
                dfb_in1.push_back(1);
            } else {  // Case where we are finishing reading in an input subblock
                std::uint32_t bytes_to_write_to_output_subblock;
                std::uint32_t l1_output_subblock_write_addr_offset;
                std::uint32_t l1_input_subblock_read_addr_offset;
                if (output_start_column >= input_start_column) {
                    dfb_in1.reserve_back(
                        1);  // We are writing a new output subblock, so we need to reserve a slot on the output DFB
                    bytes_to_write_to_output_subblock =
                        (input_end_column - output_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset = 0;
                    l1_input_subblock_read_addr_offset = (output_start_column - input_start_column) * bytes_per_element;
                } else {
                    bytes_to_write_to_output_subblock = (input_end_column - input_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset =
                        (input_start_column - output_start_column) * bytes_per_element;
                    l1_input_subblock_read_addr_offset = 0;
                }

                std::uint32_t l1_output_subblock_write_addr =
                    dfb_in1.get_write_ptr();  // Write the output subblock to the output DFB
                tt::data_movement::common::tt_memmove<false, false, true, 0>(
                    noc,
                    l1_output_subblock_write_addr + l1_output_subblock_write_addr_offset,
                    input_l1_write_addr + l1_input_subblock_read_addr_offset,
                    bytes_to_write_to_output_subblock);

                // We have processed the entire input subblock, so we must update the start and end indices of the input
                // subblock as well
                input_start_column = input_end_column + 1;
                std::uint32_t next_input_page_end_column =
                    input_start_column + (elements_per_input_page - (input_start_column % elements_per_input_page) - 1);
                input_end_column =
                    u32_min(input_start_column + elements_per_input_subblock - 1, elements_per_tensor_row - 1);
                input_end_column = u32_min(
                    next_input_page_end_column,
                    input_end_column);  // input end column should be the minimum of the next input page end column,
                                        // the end column of the next input subblock and the end of the tensor row
            }
        }
    }

    dfb_in0.push_back(1);
    dfb_in0.wait_front(1);
    dfb_in0.pop_front(1);
}

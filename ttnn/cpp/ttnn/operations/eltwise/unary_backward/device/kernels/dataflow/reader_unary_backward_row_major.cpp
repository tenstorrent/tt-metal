// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-major reader for the unary-backward device operation: reads grad_output into dfb::grad and
// input into dfb::input. It follows the block scheme of the RM_INTERLEAVED path in
// eltwise/unary/device/kernels/dataflow/reader_unary.cpp:
//
//   * A block is rows_per_tile consecutive rows. Narrow rows are packed several to one
//     tile-sized DFB entry; a row wider than a tile is split into chunks_per_row chunks of
//     chunk_elements, one DFB entry each.
//   * Eltwise compute is position-independent, and a DFB entry round-trips through the same face
//     layout on unpack and pack, so an element lands back at the DFB offset it was read into.
//
// Unlike unary it does not assume a page is a whole row -- see row_major_pages.hpp -- because two
// operands (and the output) may be paged differently.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "row_major_pages.hpp"

void kernel_main() {
    const uint32_t num_blocks = get_arg(args::num_blocks);
    const uint32_t start_block = get_arg(args::start_block);
    const uint32_t row_elements = get_arg(args::row_elements);
    const uint32_t chunk_elements = get_arg(args::chunk_elements);
    const uint32_t chunks_per_row = get_arg(args::chunks_per_row);
    const uint32_t rows_per_tile = get_arg(args::rows_per_tile);
    const uint32_t total_rows = get_arg(args::total_rows);
    const RowMajorPaging grad_paging{
        get_arg(args::grad_page_elements),
        get_arg(args::grad_pages_per_row),
        get_arg(args::grad_element_bytes),
        get_arg(args::grad_alignment)};
    const RowMajorPaging input_paging{
        get_arg(args::input_page_elements),
        get_arg(args::input_pages_per_row),
        get_arg(args::input_element_bytes),
        get_arg(args::input_alignment)};

    constexpr uint32_t onepage = 1;

    const auto grad = TensorAccessor(tensor::grad);
    const auto input = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb_grad(dfb::grad);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_scratch(dfb::scratch);
    const uint32_t scratch = aligned_scratch(
        dfb_scratch, (grad_paging.alignment > input_paging.alignment) ? grad_paging.alignment : input_paging.alignment);

    const uint32_t end_block = start_block + num_blocks;
    for (uint32_t block = start_block; block < end_block; ++block) {
        const uint32_t base_row = block * rows_per_tile;
        const uint32_t remaining = total_rows - base_row;
        const uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;

        for (uint32_t j = 0; j < chunks_per_row; ++j) {
            const uint32_t first = j * chunk_elements;
            const uint32_t last = (first + chunk_elements < row_elements) ? first + chunk_elements : row_elements;

            dfb_grad.reserve_back(onepage);
            dfb_input.reserve_back(onepage);
            const uint32_t grad_dfb = dfb_grad.get_write_ptr();
            const uint32_t input_dfb = dfb_input.get_write_ptr();
            for (uint32_t r = 0; r < actual_rows; ++r) {
                const uint32_t row = base_row + r;
                const uint32_t dfb_base = r * (last - first);
                const uint32_t grad_region_end = grad_dfb + ((dfb_base + (last - first)) * grad_paging.element_bytes);
                const uint32_t input_region_end =
                    input_dfb + ((dfb_base + (last - first)) * input_paging.element_bytes);
                for_each_row_piece(
                    grad_paging,
                    row,
                    first,
                    last,
                    dfb_base,
                    [&](uint32_t page, uint32_t src, uint32_t dst, uint32_t bytes) {
                        read_piece(noc, grad, grad_paging, page, src, grad_dfb + dst, bytes, grad_region_end, scratch);
                    });
                for_each_row_piece(
                    input_paging,
                    row,
                    first,
                    last,
                    dfb_base,
                    [&](uint32_t page, uint32_t src, uint32_t dst, uint32_t bytes) {
                        read_piece(
                            noc, input, input_paging, page, src, input_dfb + dst, bytes, input_region_end, scratch);
                    });
            }
            noc.async_read_barrier();
            dfb_grad.push_back(onepage);
            dfb_input.push_back(onepage);
        }
    }
}

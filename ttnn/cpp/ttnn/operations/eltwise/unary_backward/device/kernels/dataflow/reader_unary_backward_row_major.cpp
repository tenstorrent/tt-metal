// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-major reader for the unary-backward device operation: reads grad_output into c_0 and input
// into c_1. It follows the block scheme of the RM_INTERLEAVED path in
// eltwise/unary/device/kernels/dataflow/reader_unary.cpp:
//
//   * A block is rows_per_tile consecutive rows. Narrow rows are packed several to one
//     tile-sized CB page; a row wider than a tile is split into chunks_per_row chunks of
//     chunk_elements, one CB page each.
//   * Eltwise compute is position-independent, and a CB page round-trips through the same face
//     layout on unpack and pack, so an element lands back at the CB offset it was read into.
//
// Unlike unary it does not assume a page is a whole row -- see row_major_pages.hpp -- because two
// operands (and the output) may be paged differently.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "row_major_pages.hpp"

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t start_block = get_arg_val<uint32_t>(arg++);
    const uint32_t row_elements = get_arg_val<uint32_t>(arg++);
    const uint32_t chunk_elements = get_arg_val<uint32_t>(arg++);
    const uint32_t chunks_per_row = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t total_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(arg++);
    const RowMajorPaging grad_paging{
        get_arg_val<uint32_t>(arg),
        get_arg_val<uint32_t>(arg + 1),
        get_arg_val<uint32_t>(arg + 2),
        get_arg_val<uint32_t>(arg + 3)};
    arg += 4;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const RowMajorPaging input_paging{
        get_arg_val<uint32_t>(arg),
        get_arg_val<uint32_t>(arg + 1),
        get_arg_val<uint32_t>(arg + 2),
        get_arg_val<uint32_t>(arg + 3)};

    constexpr auto cb_id_grad = tt::CBIndex::c_0;
    constexpr auto cb_id_input = tt::CBIndex::c_1;
    constexpr auto cb_id_scratch = tt::CBIndex::c_3;
    constexpr uint32_t onepage = 1;

    constexpr auto grad_args = TensorAccessorArgs<0, 0>();
    constexpr auto input_args =
        TensorAccessorArgs<grad_args.next_compile_time_args_offset(), grad_args.next_common_runtime_args_offset()>();
    const auto grad = TensorAccessor(grad_args, grad_addr);
    const auto input = TensorAccessor(input_args, input_addr);

    DataflowBuffer dfb_grad(cb_id_grad);
    DataflowBuffer dfb_input(cb_id_input);
    const uint32_t scratch = aligned_scratch(cb_id_scratch);

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
            const uint32_t grad_cb = get_write_ptr(cb_id_grad);
            const uint32_t input_cb = get_write_ptr(cb_id_input);
            for (uint32_t r = 0; r < actual_rows; ++r) {
                const uint32_t row = base_row + r;
                const uint32_t cb_base = r * (last - first);
                const uint32_t grad_region_end = grad_cb + ((cb_base + (last - first)) * grad_paging.element_bytes);
                const uint32_t input_region_end = input_cb + ((cb_base + (last - first)) * input_paging.element_bytes);
                for_each_row_piece(
                    grad_paging,
                    row,
                    first,
                    last,
                    cb_base,
                    [&](uint32_t page, uint32_t src, uint32_t dst, uint32_t bytes) {
                        read_piece(grad, grad_paging, page, src, grad_cb + dst, bytes, grad_region_end, scratch);
                    });
                for_each_row_piece(
                    input_paging,
                    row,
                    first,
                    last,
                    cb_base,
                    [&](uint32_t page, uint32_t src, uint32_t dst, uint32_t bytes) {
                        read_piece(input, input_paging, page, src, input_cb + dst, bytes, input_region_end, scratch);
                    });
            }
            noc_async_read_barrier();
            dfb_grad.push_back(onepage);
            dfb_input.push_back(onepage);
        }
    }
}

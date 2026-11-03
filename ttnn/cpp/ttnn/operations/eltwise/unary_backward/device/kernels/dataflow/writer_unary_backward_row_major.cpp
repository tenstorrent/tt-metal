// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-major writer for the unary-backward device operation: the counterpart of
// reader_unary_backward_row_major.cpp, draining dfb::dst into the output. It is the RM_INTERLEAVED path
// of eltwise/unary/device/kernels/dataflow/writer_unary.cpp, generalized the same way as the
// reader so an output paged differently from the operands (e.g. width-sharded) is written by
// element rather than by page -- see row_major_pages.hpp.

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
    const RowMajorPaging dst_paging{
        get_arg(args::dst_page_elements),
        get_arg(args::dst_pages_per_row),
        get_arg(args::dst_element_bytes),
        get_arg(args::dst_alignment)};

    constexpr uint32_t onepage = 1;

    const auto dst = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_dst(dfb::dst);
    DataflowBuffer dfb_scratch(dfb::scratch);
    const uint32_t scratch = aligned_scratch(dfb_scratch, dst_paging.alignment);

    const uint32_t end_block = start_block + num_blocks;
    for (uint32_t block = start_block; block < end_block; ++block) {
        const uint32_t base_row = block * rows_per_tile;
        const uint32_t remaining = total_rows - base_row;
        const uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;

        for (uint32_t j = 0; j < chunks_per_row; ++j) {
            const uint32_t first = j * chunk_elements;
            const uint32_t last = (first + chunk_elements < row_elements) ? first + chunk_elements : row_elements;

            dfb_dst.wait_front(onepage);
            const uint32_t dfb_addr = dfb_dst.get_read_ptr();
            for (uint32_t r = 0; r < actual_rows; ++r) {
                const uint32_t dfb_base = r * (last - first);
                for_each_row_piece(
                    dst_paging,
                    base_row + r,
                    first,
                    last,
                    dfb_base,
                    [&](uint32_t page, uint32_t dst_offset, uint32_t src, uint32_t bytes) {
                        write_piece(noc, dst, dst_paging, page, dst_offset, dfb_addr + src, bytes, scratch);
                    });
            }
            noc.async_writes_flushed();
            dfb_dst.pop_front(onepage);
        }
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Row-major writer for the unary-backward device operation: the counterpart of
// reader_unary_backward_row_major.cpp, draining c_2 into the output. It is the RM_INTERLEAVED path
// of eltwise/unary/device/kernels/dataflow/writer_unary.cpp, generalized the same way as the
// reader so an output paged differently from the operands (e.g. width-sharded) is written by
// element rather than by page -- see row_major_pages.hpp.

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
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg++);
    const RowMajorPaging dst_paging{
        get_arg_val<uint32_t>(arg),
        get_arg_val<uint32_t>(arg + 1),
        get_arg_val<uint32_t>(arg + 2),
        get_arg_val<uint32_t>(arg + 3)};

    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    constexpr auto cb_id_scratch = tt::CBIndex::c_4;
    constexpr uint32_t onepage = 1;

    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    DataflowBuffer dfb_dst(cb_id_dst);
    const uint32_t scratch = aligned_scratch(cb_id_scratch, dst_paging.alignment);

    const uint32_t end_block = start_block + num_blocks;
    for (uint32_t block = start_block; block < end_block; ++block) {
        const uint32_t base_row = block * rows_per_tile;
        const uint32_t remaining = total_rows - base_row;
        const uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;

        for (uint32_t j = 0; j < chunks_per_row; ++j) {
            const uint32_t first = j * chunk_elements;
            const uint32_t last = (first + chunk_elements < row_elements) ? first + chunk_elements : row_elements;

            dfb_dst.wait_front(onepage);
            const uint32_t cb = get_read_ptr(cb_id_dst);
            for (uint32_t r = 0; r < actual_rows; ++r) {
                const uint32_t cb_base = r * (last - first);
                for_each_row_piece(
                    dst_paging,
                    base_row + r,
                    first,
                    last,
                    cb_base,
                    [&](uint32_t page, uint32_t dst_offset, uint32_t src, uint32_t bytes) {
                        write_piece(dst, dst_paging, page, dst_offset, cb + src, bytes, scratch);
                    });
            }
            noc_async_writes_flushed();
            dfb_dst.pop_front(onepage);
        }
    }
    noc_async_write_barrier();
}

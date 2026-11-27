// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include <algorithm>

bool contains_element(uint32_t* arr, uint32_t size, uint32_t val) {
    return std::find(arr, arr + size, val) != arr + size;
}

void kernel_main() {
    // Compile-time args
    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t index_total_size = get_arg(args::index_total_size);
    constexpr uint32_t index_size = get_arg(args::index_size);
    constexpr bool is_last_dim = get_arg(args::is_last_dim) == 1;

    // Run-time args
    uint32_t start_row = get_arg(args::start_row);
    uint32_t end_row = get_arg(args::end_row);
    uint32_t num_rows_in_dim = get_arg(args::num_rows_in_dim);
    uint32_t dim_size = get_arg(args::dim_size);
    uint32_t col_shard_id = get_arg(args::col_shard_id);
    uint32_t row_page_stride = get_arg(args::row_page_stride);

    // Derived
    constexpr uint32_t onepage = 1;

    Noc noc;
    DataflowBuffer dfb_src(dfb::src);
    DataflowBuffer dfb_index(dfb::index);

    const auto s0 = TensorAccessor(tensor::input);
    const auto s1 = TensorAccessor(tensor::index);

    // Read the entire index tensor into L1
    dfb_index.reserve_back(onepage);
    uint32_t index_addr = dfb_index.get_write_ptr();
    noc.async_read(s1, dfb_index, index_total_size, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_index.push_back(onepage);
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_addr);

    // Read input tensor pages
    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        // Performance optimization: use pre-filled page instead of input page
        bool use_filled_page = false;
        if constexpr (!is_last_dim) {
            uint32_t dim_index = (row_id / num_rows_in_dim) % dim_size;
            use_filled_page = contains_element(index_ptr, index_size, dim_index);
        }

        if (!use_filled_page) {
            // Read input page
            dfb_src.reserve_back(onepage);
            noc.async_read(
                s0,
                dfb_src,
                input_page_size,
                {.page_id = row_id * row_page_stride + col_shard_id, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_src.push_back(onepage);
        }
    }
}

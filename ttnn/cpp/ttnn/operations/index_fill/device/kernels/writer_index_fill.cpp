// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <algorithm>

bool contains_element(uint32_t* arr, uint32_t size, uint32_t val) {
    return std::find(arr, arr + size, val) != arr + size;
}

void kernel_main() {
    // Compile-time args
    // arg 0: input (CB) page size in bytes — always the input shard row size
    constexpr uint32_t in_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t index_size = get_compile_time_arg_val(1);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr bool is_last_dim = get_compile_time_arg_val(3) == 1;
    constexpr auto dst_args = TensorAccessorArgs<4>();

    // Run-time args
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_row = get_arg_val<uint32_t>(1);
    uint32_t end_row = get_arg_val<uint32_t>(2);
    uint32_t num_rows_in_dim = get_arg_val<uint32_t>(3);
    uint32_t dim_size = get_arg_val<uint32_t>(4);
    uint32_t fill_value_ = get_arg_val<uint32_t>(5);
    uint32_t in_col_shard_id = get_arg_val<uint32_t>(6);  // input col shard (last-dim bounds)
    // Output write params (independent of input layout)
    uint32_t out_col_shard_id = get_arg_val<uint32_t>(7);     // output col shard base index
    uint32_t out_row_page_stride = get_arg_val<uint32_t>(8);  // output page stride per row
    uint32_t out_col_byte_offset = get_arg_val<uint32_t>(9);  // byte offset within output page
    uint32_t out_num_col_shards = get_arg_val<uint32_t>(10);  // output writes per input row
    uint32_t out_write_size = get_arg_val<uint32_t>(11);      // bytes per output write

    // Derived
    static_assert(elem_size == 2 || elem_size == 4, "Unsupported elem_size");
    using IntType = std::conditional_t<(elem_size == 2), uint16_t, uint32_t>;

    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t index_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t fill_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t onepage = 1;
    // row_size: number of elements in the input shard page.
    // Used for: pre-fill loop size, and last-dim col_start bounds check.
    constexpr uint32_t row_size = in_page_size / elem_size;

    // col_start: first global column owned by this core's input shard.
    // For INTERLEAVED/HEIGHT input: in_col_shard_id=0 → col_start=0 (bounds check always true).
    const uint32_t col_start = in_col_shard_id * row_size;

    IntType fill_value = static_cast<IntType>(fill_value_);

    const auto s = TensorAccessor(dst_args, output_buffer_address);

    // Pre-fill a page in L1 with fill_value to avoid re-filling on every row.
    cb_reserve_back(fill_cb_id, onepage);
    uint32_t fill_addr = get_write_ptr(fill_cb_id);
    auto* fill_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(fill_addr);
    if constexpr (!is_last_dim) {
        for (uint32_t i = 0; i < row_size; ++i) {
            fill_ptr[i] = fill_value;
        }
    }
    cb_push_back(fill_cb_id, onepage);

    // Wait for index tensor to be available
    cb_wait_front(index_cb_id, onepage);
    uint32_t index_addr = get_read_ptr(index_cb_id);
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_addr);

    // Write output pages.
    // out_num_col_shards > 1 when splitting a full input row into multiple output shards
    // (INTERLEAVED/HEIGHT input → WIDTH/BLOCK output).  In all other cases it equals 1.
    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        bool use_filled_page = false;
        if constexpr (!is_last_dim) {
            uint32_t dim_index = (row_id / num_rows_in_dim) % dim_size;
            use_filled_page = contains_element(index_ptr, index_size, dim_index);
        }

        if (use_filled_page) {
            for (uint32_t kout = 0; kout < out_num_col_shards; ++kout) {
                uint64_t out_noc =
                    s.get_noc_addr(row_id * out_row_page_stride + out_col_shard_id + kout) + out_col_byte_offset;
                noc_async_write(fill_addr + kout * out_write_size, out_noc, out_write_size);
            }
            noc_async_write_barrier();
        } else {
            cb_wait_front(src_cb_id, onepage);
            uint32_t input_addr = get_read_ptr(src_cb_id);

            // For last dim: fill only the columns this shard owns in the input page.
            // For INTERLEAVED/HEIGHT in → WIDTH/BLOCK out: col_start=0, row_size=full_N,
            // so all matching indices are filled before the page is split into output writes.
            if constexpr (is_last_dim) {
                auto* input_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(input_addr);
                for (uint32_t i = 0; i < index_size; ++i) {
                    uint32_t global_col = index_ptr[i];
                    if (global_col >= col_start && global_col < col_start + row_size) {
                        input_ptr[global_col - col_start] = fill_value;
                    }
                }
            }

            for (uint32_t kout = 0; kout < out_num_col_shards; ++kout) {
                uint64_t out_noc =
                    s.get_noc_addr(row_id * out_row_page_stride + out_col_shard_id + kout) + out_col_byte_offset;
                noc_async_write(input_addr + kout * out_write_size, out_noc, out_write_size);
            }
            noc_async_write_barrier();

            cb_pop_front(src_cb_id, onepage);
        }
    }
}

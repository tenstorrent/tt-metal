// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

//  output[idx][:] = weights[input[idx]][:];

FORCE_INLINE uint32_t
logical_to_tile_storage_index(uint32_t logical_idx, uint32_t tile_width, uint32_t face_height, uint32_t face_width) {
    uint32_t row = logical_idx / tile_width;
    uint32_t col = logical_idx % tile_width;
    uint32_t faces_per_row = tile_width / face_width;
    uint32_t face_row = row / face_height;
    uint32_t face_col = col / face_width;
    uint32_t face_id = face_row * faces_per_row + face_col;
    uint32_t sub_row = row % face_height;
    uint32_t sub_col = col % face_width;
    uint32_t face_hw = face_height * face_width;
    return face_id * face_hw + sub_row * face_width + sub_col;
}

void kernel_main() {
    uint32_t input_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t weight_buffer_src_addr = get_arg_val<uint32_t>(1);
    uint32_t input_page_id = get_arg_val<uint32_t>(2);
    uint32_t num_of_pages = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t weight_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t elems_per_page = get_compile_time_arg_val(3);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_buf_alignment = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t input_is_tile_layout = get_compile_time_arg_val(7);
    constexpr uint32_t tile_width = get_compile_time_arg_val(8);
    constexpr uint32_t face_height = get_compile_time_arg_val(9);
    constexpr uint32_t face_width = get_compile_time_arg_val(10);

    constexpr auto input_args = TensorAccessorArgs<11>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const auto input = TensorAccessor(input_args, input_buffer_src_addr, input_page_size);
    const auto weights = TensorAccessor(weights_args, weight_buffer_src_addr, weight_page_size);

    cb_reserve_back(input_cb_index, 1);
    uint32_t index_cb_addr = get_write_ptr(input_cb_index);
    volatile tt_l1_ptr input_token_t* index_cb_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(index_cb_addr);

    for (uint32_t page_id = input_page_id; page_id < input_page_id + num_of_pages; page_id++) {
        auto input_pages = input.pages(page_id, page_id + 1);
        auto input_page_iter = input_pages.begin();

        noc_async_read(input_page_iter->noc_addr(), index_cb_addr, input_page_size);
        noc_async_read_barrier();

        for (uint32_t index = 0; index < elems_per_page; ++index) {
            uint32_t storage_index = index;
            if (input_is_tile_layout) {
                storage_index = logical_to_tile_storage_index(index, tile_width, face_height, face_width);
            }
            input_token_t weights_flatten_idx = index_cb_ptr[storage_index];

            cb_reserve_back(output_cb_index, 1);
            uint32_t output_cb_addr = get_write_ptr(output_cb_index);

            uint64_t weight_noc_addr = get_token_noc_addr(weights_flatten_idx, weights);
            noc_async_read<weight_page_size>(weight_noc_addr, output_cb_addr, weight_page_size);
            noc_async_read_barrier();

            cb_push_back(output_cb_index, 1);
        }
    }
}

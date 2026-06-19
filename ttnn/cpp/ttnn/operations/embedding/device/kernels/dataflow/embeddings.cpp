// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"

void kernel_main() {
    Noc noc;

    const std::uint32_t input_buffer_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t weight_buffer_src_addr = get_arg_val<uint32_t>(1);
    const std::uint32_t batch_offset = get_arg_val<uint32_t>(2);
    const std::uint32_t weights_offset = get_arg_val<uint32_t>(3);
    const std::uint32_t num_rows = get_arg_val<uint32_t>(4);

    const std::uint32_t index_idx = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t weight_stick_size = get_compile_time_arg_val(4);

    constexpr uint32_t rows_per_block = get_compile_time_arg_val(5);  // Input elems per block
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(6);

    constexpr uint32_t chunk_size = get_compile_time_arg_val(7);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t last_chunk_size = get_compile_time_arg_val(9);

    constexpr auto input_args = TensorAccessorArgs<10>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_buffer_src_addr);
    const auto weights = TensorAccessor(weights_args, weight_buffer_src_addr);

    prepare_local_cache(noc, cb_id_in2, weights, weight_stick_size, /*pad_token_arg_idx=*/6);

    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);

    cb_in1.reserve_back(1);
    uint32_t input_l1_addr = cb_in1.get_write_ptr();
    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    uint32_t curr_row = batch_offset;  // Number of pages/rows we have read from input so far
    uint32_t offset = weights_offset;  // Which input elem we are on (bytes offset from start of row)
    uint32_t index = index_idx;
    uint32_t input_elem_size_bytes = input_block_size_bytes / rows_per_block;

    bool read_indices = true;
    for (uint32_t i = 0; i < num_rows; ++i) {
        if (read_indices) {
            noc.async_read(
                input,
                CoreLocalMem<uint32_t>(input_l1_addr),
                input_block_size_bytes,
                {.page_id = curr_row, .offset_bytes = offset},
                {});
            noc.async_read_barrier();
            read_indices = false;
        }
        input_token_t token = input_l1_ptr[index];

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_in0.reserve_back(1);
            uint32_t l1_write_addr = cb_in0.get_write_ptr();
            uint32_t current_chunk_size = (chunk < num_chunks - 1) ? chunk_size : last_chunk_size;
            uint32_t chunk_offset = chunk * chunk_size;
            read_token_async(noc, token, weights, l1_write_addr, current_chunk_size, chunk_offset);
            noc.async_read_barrier();
            cb_in0.push_back(1);
        }

        index++;
        uint32_t total_bytes_into_page = offset + index * input_elem_size_bytes;
        bool end_of_block = index == rows_per_block;
        bool end_of_page = total_bytes_into_page == input_page_size;
        if (end_of_block || end_of_page) {
            offset += input_block_size_bytes;
            if (end_of_page) {
                offset = 0;
                curr_row++;
            }
            index = 0;
            read_indices = true;
        }
    }
}

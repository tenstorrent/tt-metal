// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"

void kernel_main() {
    const uint32_t input_buffer_src_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_buffer_src_addr = get_arg_val<uint32_t>(1);
    const uint32_t input_start_id = get_arg_val<uint32_t>(2);
    const uint32_t input_start_offset = get_arg_val<uint32_t>(3);
    const uint32_t weight_offset = get_arg_val<uint32_t>(4);
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t weight_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t weight_block_size = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(6);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(8);

    constexpr auto input_args = TensorAccessorArgs<9>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    auto input = TensorAccessor(input_args, input_buffer_src_addr, input_page_size);
    auto weights = TensorAccessor(weights_args, weight_buffer_src_addr + weight_offset, weight_stick_size);

    prepare_local_cache(cb_id_in2, weights, weight_block_size, /*pad_token_arg_idx=*/6);

    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_id_in1);

    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    uint32_t curr_row = input_start_id;
    uint32_t offset = input_start_offset;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint64_t noc_input_src_addr = get_noc_addr(curr_row, input) + offset;
        noc_async_read<input_block_size_bytes>(noc_input_src_addr, input_l1_addr, input_block_size_bytes);
        noc_async_read_barrier();

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_reserve_back(cb_id_in0, tiles_per_chunk);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            // Calculate the chunk size and offset within the embedding vector
            uint32_t weight_chunk_size = weight_block_size / num_chunks;
            uint32_t weight_chunk_offset = chunk * weight_chunk_size;

            for (uint32_t k = 0; k < tile_height; ++k) {
                input_token_t token = input_l1_ptr[k];
                uint64_t src_noc_addr = get_token_noc_addr(token, weights);

                noc_async_read(src_noc_addr + weight_chunk_offset, l1_write_addr, weight_chunk_size);
                l1_write_addr += weight_chunk_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, tiles_per_chunk);
        }

        offset += input_block_size_bytes;
        if (offset == input_page_size) {
            offset = 0;
            curr_row++;
        }
    }
}

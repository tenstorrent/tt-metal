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
    constexpr uint32_t weight_block_size = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(6);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(8);

    constexpr auto input_args = TensorAccessorArgs<9>();
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    auto input = TensorAccessor(input_args, input_buffer_src_addr);
    auto weights = TensorAccessor(weights_args, weight_buffer_src_addr + weight_offset);

    prepare_local_cache(noc, cb_id_in2, weights, weight_block_size, /*pad_token_arg_idx=*/6);

    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);

    cb_in1.reserve_back(1);
    uint32_t input_l1_addr = cb_in1.get_write_ptr();

    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    uint32_t curr_row = input_start_id;
    uint32_t offset = input_start_offset;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        noc.async_read<NocOptions::DEFAULT, input_block_size_bytes>(
            input,
            CoreLocalMem<uint32_t>(input_l1_addr),
            input_block_size_bytes,
            {.page_id = curr_row, .offset_bytes = offset},
            {});
        noc.async_read_barrier();

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_in0.reserve_back(tiles_per_chunk);
            uint32_t l1_write_addr = cb_in0.get_write_ptr();

            // Calculate the chunk size and offset within the embedding vector
            uint32_t weight_chunk_size = weight_block_size / num_chunks;
            uint32_t weight_chunk_offset = chunk * weight_chunk_size;

            for (uint32_t k = 0; k < tile_height; ++k) {
                input_token_t token = input_l1_ptr[k];
                read_token_async(noc, token, weights, l1_write_addr, weight_chunk_size, weight_chunk_offset);
                l1_write_addr += weight_chunk_size;
            }
            noc.async_read_barrier();
            cb_in0.push_back(tiles_per_chunk);
        }

        offset += input_block_size_bytes;
        if (offset == input_page_size) {
            offset = 0;
            curr_row++;
        }
    }
}

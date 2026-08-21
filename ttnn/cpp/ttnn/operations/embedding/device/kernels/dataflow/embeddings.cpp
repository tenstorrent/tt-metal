// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common_metal2.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    const std::uint32_t batch_offset = get_arg(args::batch_offset);
    const std::uint32_t weights_offset = get_arg(args::weights_offset);
    const std::uint32_t num_rows = get_arg(args::num_rows);

    const std::uint32_t index_idx = get_arg(args::index_idx);

    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t weight_stick_size = get_arg(args::weight_stick_size);

    constexpr uint32_t rows_per_block = get_arg(args::rows_per_block);  // Input elems per block
    constexpr uint32_t input_block_size_bytes = get_arg(args::input_block_size_bytes);

    constexpr uint32_t chunk_size = get_arg(args::chunk_size);
    constexpr uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr uint32_t last_chunk_size = get_arg(args::last_chunk_size);

    const auto input = TensorAccessor(tensor::input);
    const auto weights = TensorAccessor(tensor::weights);

    // The local weight cache exists only for the embeddings types that serve some weight rows out of
    // local SRAM, so it is bound (and named) only on those builds.
#if defined PADDED
    prepare_local_cache(noc, scratch::local_cache, weights, weight_stick_size, get_arg(args::pad_token));
#elif defined BINARY
    prepare_local_cache(noc, scratch::local_cache, weights, weight_stick_size);
#endif

    // dfb_in0 stages the weight sticks this reader gathers; a writer kernel drains it to the output
    // tensor, except when the output is sharded and the staging buffer is the output shard itself.
    DataflowBuffer dfb_in0(dfb::in0);
    // The index scratchpad is this reader's private page for the block of indices it is working
    // through: it fetches the block into it and decodes tokens straight back out, with no hand-off to
    // another kernel. Volatile because the NoC writes the region behind the compiler's back.
    Scratchpad<volatile input_token_t> indices(scratch::indices);

    uint32_t curr_row = batch_offset;  // Number of pages/rows we have read from input so far
    uint32_t offset = weights_offset;  // Which input elem we are on (bytes offset from start of row)
    uint32_t index = index_idx;
    uint32_t input_elem_size_bytes = input_block_size_bytes / rows_per_block;

    bool read_indices = true;
    for (uint32_t i = 0; i < num_rows; ++i) {
        if (read_indices) {
            noc.async_read(input, indices, input_block_size_bytes, {.page_id = curr_row, .offset_bytes = offset}, {});
            noc.async_read_barrier();
            read_indices = false;
        }
        input_token_t token = indices[index];

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            dfb_in0.reserve_back(1);
            uint32_t l1_write_addr = dfb_in0.get_write_ptr();
            uint32_t current_chunk_size = (chunk < num_chunks - 1) ? chunk_size : last_chunk_size;
            uint32_t chunk_offset = chunk * chunk_size;
            read_token_async(noc, token, weights, l1_write_addr, current_chunk_size, chunk_offset);
            noc.async_read_barrier();
            dfb_in0.push_back(1);
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

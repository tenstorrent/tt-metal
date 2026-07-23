// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"

void kernel_main() {
    Noc noc;

    const auto input_start_id = get_arg(args::input_start_id);
    const auto input_start_offset = get_arg(args::input_start_offset);
    // Byte offset of this core's weight block within the full weight page (non-zero only for
    // block/width-sharded output). Applied to the weights-accessor reads rather than folded into the
    // accessor base, since the tensor binding fixes the base to the buffer base.
    const auto weight_offset = get_arg(args::weight_offset);
    const auto num_blocks = get_arg(args::num_blocks);

    constexpr auto input_page_size = get_arg(args::input_page_size);
    constexpr auto weight_block_size = get_arg(args::weight_block_size);
    constexpr auto tiles_per_chunk = get_arg(args::tiles_per_chunk);
    constexpr auto input_block_size_bytes = get_arg(args::input_block_size_bytes);
    constexpr auto num_chunks = get_arg(args::num_chunks);
    constexpr auto last_chunk_tiles = get_arg(args::last_chunk_tiles);

    auto input = TensorAccessor(tensor::input);
    auto weights = TensorAccessor(tensor::weights);

    prepare_local_cache(noc, weights, weight_block_size, weight_offset);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    dfb_in1.reserve_back(1);
    uint32_t input_l1_addr = dfb_in1.get_write_ptr();

    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    // Per-row byte counts for the full and (possibly partial) last chunk.
    constexpr uint32_t num_tiles_per_block = (num_chunks - 1) * tiles_per_chunk + last_chunk_tiles;
    constexpr uint32_t bytes_per_tile_row = weight_block_size / num_tiles_per_block;
    constexpr uint32_t full_chunk_bytes = tiles_per_chunk * bytes_per_tile_row;
    constexpr uint32_t last_chunk_bytes = last_chunk_tiles * bytes_per_tile_row;

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
            const bool is_last = (chunk + 1 == num_chunks);
            const uint32_t this_chunk_tiles = is_last ? last_chunk_tiles : tiles_per_chunk;
            const uint32_t weight_chunk_size = is_last ? last_chunk_bytes : full_chunk_bytes;
            const uint32_t weight_chunk_offset = chunk * full_chunk_bytes;

            dfb_in0.reserve_back(this_chunk_tiles);
            uint32_t l1_write_addr = dfb_in0.get_write_ptr();

            for (uint32_t k = 0; k < tile_height; ++k) {
                input_token_t token = input_l1_ptr[k];
                read_token_async(
                    noc, token, weights, l1_write_addr, weight_chunk_size, weight_chunk_offset, weight_offset);
                l1_write_addr += weight_chunk_size;
            }
            noc.async_read_barrier();
            dfb_in0.push_back(this_chunk_tiles);
        }

        offset += input_block_size_bytes;
        if (offset == input_page_size) {
            offset = 0;
            curr_row++;
        }
    }
    // dfb_in1 is reserved once as an index scratch buffer (no downstream consumer); commit the
    // reservation so the DFB is left balanced.
    dfb_in1.push_back(1);
}

// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 of every core: first the bias tiles of the chunks this core's units cover, once; then for
// each unit (a 32-token block times a chunk of output columns), each slot's 32 expert-row segments
// and the slots' weight tiles. A unit's reads all go out before one barrier, so their latencies
// overlap.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t token_blocks = get_named_compile_time_arg_val("token_blocks");
    constexpr uint32_t rows_cb_id = get_named_compile_time_arg_val("rows_cb");
    constexpr uint32_t weights_cb_id = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("bias_cb");
    constexpr uint32_t index_cb_id = get_named_compile_time_arg_val("index_cb");
    constexpr auto rows_args = TensorAccessorArgs<0>();
    constexpr auto index_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();

    const uint32_t rows_address = get_arg_val<uint32_t>(0);
    const uint32_t index_address = get_arg_val<uint32_t>(1);
    const uint32_t weights_address = get_arg_val<uint32_t>(2);
    const uint32_t bias_address = get_arg_val<uint32_t>(3);
    const uint32_t first_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    constexpr uint32_t block_tokens = 32;
    constexpr uint32_t chunks = hidden_tiles / chunk_tiles;
    // One bf16 row segment per tile column: 32 values.
    constexpr uint32_t segment_bytes = chunk_tiles * block_tokens * sizeof(uint16_t);
    constexpr uint32_t index_block_bytes = block_tokens * sizeof(uint32_t);

    const Noc noc;
    DataflowBuffer rows_cb(rows_cb_id);
    DataflowBuffer weights_cb(weights_cb_id);
    DataflowBuffer bias_cb(bias_cb_id);
    DataflowBuffer index_cb(index_cb_id);
    const auto rows_accessor = TensorAccessor(rows_args, rows_address);
    const auto index_accessor = TensorAccessor(index_args, index_address);
    const auto weights_accessor = TensorAccessor(weights_args, weights_address);
    const auto bias_accessor = TensorAccessor(bias_args, bias_address);
    constexpr uint32_t bias_tile_bytes = get_tile_size(bias_cb_id);
    constexpr uint32_t weight_tile_bytes = get_tile_size(weights_cb_id);

    // Units run token-block-major, so a core with fewer units than chunks covers a run of chunks that
    // may wrap. Every core reads the same few bias pages: each starts at a different tile of its run,
    // so the cores do not queue on one DRAM bank.
    const uint32_t covered_chunks = num_units < chunks ? num_units : chunks;
    const uint32_t covered_tiles = covered_chunks * chunk_tiles;
    const uint32_t first_chunk = first_unit % chunks;
    bias_cb.reserve_back(hidden_tiles);
    for (uint32_t step = 0; step < covered_tiles; ++step) {
        const uint32_t covered_tile = (step + first_unit) % covered_tiles;
        const uint32_t chunk = (first_chunk + covered_tile / chunk_tiles) % chunks;
        const uint32_t tile = chunk * chunk_tiles + covered_tile % chunk_tiles;
        noc.async_read(
            bias_accessor, bias_cb, bias_tile_bytes, {.page_id = tile}, {.offset_bytes = tile * bias_tile_bytes});
    }
    noc.async_read_barrier();
    bias_cb.push_back(hidden_tiles);

    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb.get_write_ptr());
    uint32_t loaded_block = token_blocks;
    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        const uint32_t block = unit / chunks;
        const uint32_t chunk = unit % chunks;
        if (block != loaded_block) {
            for (uint32_t k = 0; k < top_k; ++k) {
                noc.async_read(
                    index_accessor,
                    index_cb,
                    index_block_bytes,
                    {.page_id = k, .offset_bytes = block * index_block_bytes},
                    {.offset_bytes = k * index_block_bytes});
            }
            noc.async_read_barrier();
            loaded_block = block;
        }

        rows_cb.reserve_back(top_k * chunk_tiles);
        weights_cb.reserve_back(top_k);
        for (uint32_t k = 0; k < top_k; ++k) {
            for (uint32_t token = 0; token < block_tokens; ++token) {
                noc.async_read(
                    rows_accessor,
                    rows_cb,
                    segment_bytes,
                    {.page_id = indices[k * block_tokens + token], .offset_bytes = chunk * segment_bytes},
                    {.offset_bytes = (k * block_tokens + token) * segment_bytes});
            }
            noc.async_read(
                weights_accessor,
                weights_cb,
                weight_tile_bytes,
                {.page_id = k * token_blocks + block},
                {.offset_bytes = k * weight_tile_bytes});
        }
        noc.async_read_barrier();
        rows_cb.push_back(top_k * chunk_tiles);
        weights_cb.push_back(top_k);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams this core's share of the paged key cache to compute, one block at a time. The top-k
// and kv gather stages are still boilerplate.

#define COMPRESS_RATE 4
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    [[maybe_unused]] constexpr uint32_t k = get_arg(args::k);
    constexpr uint32_t num_query_group_tiles = get_arg(args::num_query_group_tiles);  // Hi / 8 * D / 32 8x32 tiles
    constexpr uint32_t num_weight_tiles = get_arg(args::num_weight_tiles);            // Hi / 32 1x32 tiles
    constexpr uint32_t page_block_size = get_arg(args::page_block_size);
    constexpr uint32_t num_tiles_per_block_of_key = get_arg(args::num_tiles_per_block_of_key);

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t num_cores = get_arg(args::num_cores);

    // ---- Tensors ----
    // query and head_weights have no accessor: dfb::query and dfb::weights are borrowed from this
    // core's replicated shards.
    const auto key_cache = TensorAccessor(tensor::key_cache);
    const auto kv_cache = TensorAccessor(tensor::kv_cache);
    const auto page_table = TensorAccessor(tensor::page_table);
    const auto cur_pos = TensorAccessor(tensor::cur_pos);
#ifdef HAS_VALID_LENGTH
    // tensor::valid_length only exists when the optional tensor is bound.
    [[maybe_unused]] const auto valid_length = TensorAccessor(tensor::valid_length);
#endif

    // ---- Dataflow buffers ----
    DataflowBuffer query_rm_dfb(dfb::query_rm);  // producer -> compute (borrowed, 8-row strips of query)
    DataflowBuffer key_dfb(dfb::key);            // producer -> compute
    DataflowBuffer weights_dfb(dfb::weights);    // producer -> compute (borrowed, already resident)
    DataflowBuffer ctrl_dfb(dfb::ctrl);          // producer -> compute
    DataflowBuffer indices_dfb(dfb::indices);    // consumer <- compute
    DataflowBuffer kv_dfb(dfb::kv);              // producer -> writer
    DataflowBuffer cur_pos_dfb(dfb::cur_pos);    // producer -> writer

    Noc noc;

    // Scoring inputs for compute. The query and weight rows are already in L1, so publishing them
    // is the whole job.
    query_rm_dfb.push_back(num_query_group_tiles);
    weights_dfb.push_back(num_weight_tiles);

    cur_pos_dfb.reserve_back(1);
    noc.async_read(cur_pos, cur_pos_dfb, cur_pos_dfb.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t cur_pos_value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_dfb.get_write_ptr())[0];
    cur_pos_dfb.push_back(1);

    // cur_pos is inclusive, so cur_pos + 1 tokens have been compressed into this many keys.
    const uint32_t num_keys = (cur_pos_value + 1) / COMPRESS_RATE;
    const uint32_t total_num_blocks = tt::data_movement::common::div_up(num_keys, page_block_size);
    const uint32_t work_per_core = tt::data_movement::common::div_up(total_num_blocks, num_cores);
    const uint32_t start_block = std::min(work_per_core * core_index, total_num_blocks);
    const uint32_t end_block = std::min(start_block + work_per_core, total_num_blocks);

    // Entry 0 tells compute how many blocks to score. Entry 1 stays reserved and never pushed: it is
    // this kernel's scratch for page-table chunks.
    DPRINT(
        "cur_pos: {}, core_index:{} total_num_blocks: {}, work_per_core: {}, start_block: {}, end_block: {}, "
        "num_blocks_to_process: {}\n",
        cur_pos_value,
        core_index,
        total_num_blocks,
        work_per_core,
        start_block,
        end_block,
        end_block - start_block);
    ctrl_dfb.reserve_back(2);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_dfb.get_write_ptr())[0] = end_block - start_block;
    ctrl_dfb.push_back(1);
    volatile tt_l1_ptr uint32_t* page_table_chunk =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_dfb.get_write_ptr());

    // Each 64 B DRAM-aligned chunk of the page table holds 16 uint32_t entries.
    uint32_t loaded_chunk = UINT32_MAX;
    for (uint32_t block = start_block; block < end_block; ++block) {
        const uint32_t chunk = block / 16;
        if (chunk != loaded_chunk) {
            noc.async_read(page_table, ctrl_dfb, 64, {.page_id = 0, .offset_bytes = 64 * chunk}, {.offset_bytes = 0});
            noc.async_read_barrier();
            loaded_chunk = chunk;
        }

        // key_cache is DRAM ND-sharded with one block per shard, so a block is contiguous in one bank.
        const uint32_t physical_block = page_table_chunk[block % 16];
        const uint32_t first_tile_id = physical_block * num_tiles_per_block_of_key;

        key_dfb.reserve_back(num_tiles_per_block_of_key);
        noc.async_read(
            key_cache,
            key_dfb,
            num_tiles_per_block_of_key * key_dfb.get_entry_size(),
            {.page_id = first_tile_id},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        key_dfb.push_back(num_tiles_per_block_of_key);
    }

    // Top-k indices from compute, used to pick which kv_cache rows to gather.
    indices_dfb.wait_front(1);

    // Gather the selected kv_cache rows for the writer.
    kv_dfb.reserve_back(1);
    noc.async_read(kv_cache, kv_dfb, kv_dfb.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    kv_dfb.push_back(1);

    indices_dfb.pop_front(1);
}

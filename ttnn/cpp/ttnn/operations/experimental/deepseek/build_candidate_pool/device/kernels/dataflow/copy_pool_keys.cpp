// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies half (compile arg ``half``) of this core's share of the candidate blocks' index keys into
// pool_keys, in candidate order. With WRITE_ROWS it also writes the kv cache row of every pool key
// in the core's share to pool_rows.
//
// key_cache and pool_keys are bfloat16 [blocks, 1, page_block_size, D] in 32x32 tiles. A candidate
// block is candidate_block_size rows starting at a multiple of it, so in each tile along the row it
// is one contiguous run of cbs 32 B face rows in the left face and one in the right face. The run
// sits at the same place in the destination tile pattern, so keys are copied run by run with no
// reformatting.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t cbs = get_arg(args::candidate_block_size);
    constexpr uint32_t page_block_size = get_arg(args::page_block_size);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t stage_chunks = get_arg(args::stage_chunks);
    constexpr uint32_t half = get_arg(args::half);
    // Blocks per core are a multiple of this, so every core's pool_rows run starts 64 B aligned.
    constexpr uint32_t align_blocks = get_arg(args::align_blocks);
    // Lookup scratch offsets, all 64 B aligned.
    constexpr uint32_t chunk_offset = get_arg(args::chunk_offset);
    constexpr uint32_t list_offset = get_arg(args::list_offset);
    constexpr uint32_t phys_offset = get_arg(args::phys_offset);
    [[maybe_unused]] constexpr uint32_t rows_offset = get_arg(args::rows_offset);

    constexpr uint32_t kFaceRowBytes = 16 * sizeof(uint16_t);
    constexpr uint32_t kFaceBytes = 16 * kFaceRowBytes;
    constexpr uint32_t chunk_bytes = cbs * kFaceRowBytes;
    constexpr uint32_t tiles_per_block = page_block_size / 32 * tiles_per_row;
    constexpr uint32_t chunks_per_block = 2 * tiles_per_row;
    static_assert(stage_chunks % (2 * chunks_per_block) == 0, "the stage must split into two banks of whole blocks");
    constexpr uint32_t bank_blocks = stage_chunks / 2 / chunks_per_block;

    // ---- Runtime args ----
    const uint32_t core_index = get_arg(args::core_index);
    const uint32_t num_cores = get_arg(args::num_cores);

    // ---- Tensors ----
    const auto candidate_blocks = TensorAccessor(tensor::candidate_blocks);  // [1, topk_blocks] INT32
    const auto pool_count = TensorAccessor(tensor::pool_count);              // [1] INT32
    const auto page_table = TensorAccessor(tensor::page_table);              // [1, max_blocks] INT32
    const auto key_cache = TensorAccessor(tensor::key_cache);
    const auto pool_keys = TensorAccessor(tensor::pool_keys);
#ifdef WRITE_ROWS
    const auto pool_rows = TensorAccessor(tensor::pool_rows);  // [1, topk_blocks * cbs] INT32
#endif

    // ---- Dataflow buffers ----
    DataflowBuffer lookup_dfb(dfb::lookup);  // pool_count, a page-table chunk, list slice, physical blocks, rows
    DataflowBuffer stage_dfb(dfb::stage);    // key runs in flight, two banks

    Noc noc;

    lookup_dfb.reserve_back(1);
    const uint32_t lookup_addr = lookup_dfb.get_write_ptr();
    noc.async_read(pool_count, lookup_dfb, 64, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr)[0];

    // Only blocks holding the pool's count keys are used, split evenly over the cores.
    const uint32_t num_used = tt::data_movement::common::div_up(count, cbs);
    const uint32_t per_core =
        tt::data_movement::common::div_up(tt::data_movement::common::div_up(num_used, num_cores), align_blocks) *
        align_blocks;
    const uint32_t start = std::min(core_index * per_core, num_used);
    const uint32_t end = std::min(start + per_core, num_used);
    if (start == end) {
        return;
    }

    // This core's slice of the list, read from the 64 B boundary at or below it.
    const uint32_t first_byte = start * sizeof(uint32_t) / 64 * 64;
    const uint32_t last_byte = tt::data_movement::common::div_up(end * sizeof(uint32_t), 64) * 64;
    noc.async_read(
        candidate_blocks,
        lookup_dfb,
        last_byte - first_byte,
        {.page_id = 0, .offset_bytes = first_byte},
        {.offset_bytes = list_offset});
    noc.async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        lookup_addr + list_offset + start * sizeof(uint32_t) - first_byte);

    // Physical page block of each of the core's candidate blocks. The list ascends, so each 64 B
    // page-table chunk (16 entries) is read once.
    volatile tt_l1_ptr uint32_t* chunk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + chunk_offset);
    volatile tt_l1_ptr uint32_t* phys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + phys_offset);
#ifdef PAGE_TABLE_ROW
    // This session's page-table row, staged where the chunks then land.
    const auto page_table_row_gen = TensorAccessor(tensor::page_table_row);
    noc.async_read(page_table_row_gen, lookup_dfb, 64, {.page_id = 0}, {.offset_bytes = chunk_offset});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t page_table_row = chunk[0];
#else
    const uint32_t page_table_row = 0;
#endif
    uint32_t loaded_chunk = UINT32_MAX;
    for (uint32_t c = start; c < end; ++c) {
        const uint32_t logical_block = list[c - start] * cbs / page_block_size;
        if (logical_block / 16 != loaded_chunk) {
            loaded_chunk = logical_block / 16;
            noc.async_read(
                page_table,
                lookup_dfb,
                64,
                {.page_id = page_table_row, .offset_bytes = 64 * loaded_chunk},
                {.offset_bytes = chunk_offset});
            noc.async_read_barrier();
            invalidate_l1_cache();
        }
        phys[c - start] = chunk[logical_block % 16];
    }

#ifdef WRITE_ROWS
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lookup_addr + rows_offset);
    for (uint32_t c = start; c < end; ++c) {
        const uint32_t key = list[c - start] * cbs;
        const uint32_t first_row = phys[c - start] * page_block_size + key % page_block_size;
        for (uint32_t i = 0; i < cbs; ++i) {
            rows[(c - start) * cbs + i] = first_row + i;
        }
    }
    (void)ckernel::load_blocking(&rows[(end - start) * cbs - 1]);
    noc.async_write(
        lookup_dfb,
        pool_rows,
        (end - start) * cbs * sizeof(uint32_t),
        {.offset_bytes = rows_offset},
        {.page_id = 0, .offset_bytes = start * cbs * sizeof(uint32_t)});
#endif

    // Keys: the first half of the core's blocks on one kernel, the rest on the other.
    const uint32_t mid = start + (end - start + 1) / 2;
    const uint32_t key_start = half == 0 ? start : mid;
    const uint32_t key_end = half == 0 ? mid : end;
    stage_dfb.reserve_back(stage_chunks);
    uint32_t bank = 0;
    for (uint32_t c = key_start; c < key_end; c += bank_blocks, bank ^= 1) {
        const uint32_t n = std::min(bank_blocks, key_end - c);
        const uint32_t base = bank * bank_blocks * chunks_per_block;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t row = list[c + j - start] * cbs % page_block_size;
            const uint32_t first_tile = phys[c + j - start] * tiles_per_block + row / 32 * tiles_per_row;
            const uint32_t run_offset = (row % 32) / 16 * 2 * kFaceBytes + (row % 16) * kFaceRowBytes;
            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(
                        key_cache,
                        stage_dfb,
                        chunk_bytes,
                        {.page_id = first_tile + t, .offset_bytes = run_offset + f * kFaceBytes},
                        {.offset_bytes = (base + j * chunks_per_block + 2 * t + f) * chunk_bytes});
                }
            }
        }
        noc.async_read_barrier();
        // Drains the other bank's writes, which were overlapping these reads, so it can be read into next.
        noc.async_writes_flushed();
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t entry = (c + j) * cbs;
            const uint32_t row = entry % page_block_size;
            const uint32_t first_tile = entry / page_block_size * tiles_per_block + row / 32 * tiles_per_row;
            const uint32_t run_offset = (row % 32) / 16 * 2 * kFaceBytes + (row % 16) * kFaceRowBytes;
            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_write(
                        stage_dfb,
                        pool_keys,
                        chunk_bytes,
                        {.offset_bytes = (base + j * chunks_per_block + 2 * t + f) * chunk_bytes},
                        {.page_id = first_tile + t, .offset_bytes = run_offset + f * kFaceBytes});
                }
            }
        }
    }
    noc.async_write_barrier();
}

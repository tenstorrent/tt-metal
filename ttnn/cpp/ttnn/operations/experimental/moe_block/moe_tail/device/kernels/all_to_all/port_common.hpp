// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the all-to-all port kernels (port_staging_reader, port_fabric_sender): their compile-time args 0 - 5 (then
// each kernel's tensor accessor and row block args) and skip_padding, the CB / semaphore ids they share with the
// workers, and the walk over a destination block's runs.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

// the staging CB between the two port kernels (moe_tail_program_factory's all_to_all_cb::CB_PORT_STAGING)
constexpr uint32_t staging_cb = get_compile_time_arg_val(0);
// (compile-time arg 1: the pages' bytes - page_bytes below)
// a run: up to pages_per_run pages of one DRAM bank (the fabric's max payload / a tile)
constexpr uint32_t pages_per_run = get_compile_time_arg_val(2);
constexpr uint32_t num_dram_banks = get_compile_time_arg_val(3);
// the staging CB moves whole groups of runs_per_group runs (the reader's read barrier, the sender's flush)
constexpr uint32_t runs_per_group = get_compile_time_arg_val(4);
// a fused arrival increment every arrival_increment_every runs of a destination (and on its last)
constexpr uint32_t arrival_increment_every = get_compile_time_arg_val(5);
constexpr bool skip_padding = get_named_compile_time_arg_val("skip_padding") != 0;
// the staging buffer's, the output's and the staging CB's pages: bfloat16 tiles
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
static_assert(page_bytes == moe_block_tiles::kBf16TileBytes, "the all-to-all moves bfloat16 tiles");
constexpr uint32_t PORT_ACCESSOR_ARGS = 6;  // (each port kernel's tensor accessor, then its row block args)
// (all_to_all_cb::CB_Z_READY: the workers' z-ready words, the same address on workers and ports; the workers' zero-gate
// semaphore id)
constexpr uint32_t CB_Z_READY = 5, ZERO_GATE_SEM = 0;

// a port's runs of a destination block's row tile: link `link` of num_links takes the bank offsets
// bank_offset = link, link + num_links, .. (< min(num_dram_banks, block_tiles)); part 1 / 2 the first / second half of
// those; a run = up to pages_per_run pages run_first_page + i num_dram_banks (run_first_page = bank_offset,
// bank_offset + pages_per_run num_dram_banks, ..): callback(run_first_page, pages)
template <typename Callback>
FORCE_INLINE void for_each_run(
    uint32_t block_tiles, uint32_t link, uint32_t num_links, uint32_t part, Callback&& callback) {
    const uint32_t banks_used = block_tiles < num_dram_banks ? block_tiles : num_dram_banks;
    const uint32_t my_offsets = banks_used > link ? (banks_used - link + num_links - 1) / num_links : 0;
    const uint32_t first_offset = part == 2 ? my_offsets / 2 : 0, end_offset = part == 1 ? my_offsets / 2 : my_offsets;
    for (uint32_t i = first_offset; i < end_offset; ++i) {
        const uint32_t bank_offset = link + i * num_links;
        for (uint32_t run_first_page = bank_offset; run_first_page < block_tiles;
             run_first_page += pages_per_run * num_dram_banks) {
            const uint32_t pages_left = (block_tiles - run_first_page + num_dram_banks - 1) / num_dram_banks;
            callback(run_first_page, pages_left < pages_per_run ? pages_left : pages_per_run);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader (blocks_rm_gather): load_block.
// Unit (n, c, h) owns the pages_per_unit = W * I consecutive input pages of J values that start at
// unit * pages_per_unit. Every page lands in its own slot of slot_bytes (the page's bank stride), so
// local and remote address agree mod 64 as Blackhole DRAM reads require. One read barrier and one
// push per block.
// read_nocs = 2 issues every other round of num_banks pages on the second NoC: on Blackhole the two
// NoCs reach a DRAM bank through different endpoints, so each bank serves both request streams.
// stagger rotates the issue order of every block so cores that start on the same bank do not walk
// the banks in lockstep.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t block_h = get_compile_time_arg_val(1);
constexpr uint32_t pages_per_unit = get_compile_time_arg_val(2);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(4);
constexpr uint32_t read_nocs = get_compile_time_arg_val(5);
constexpr uint32_t num_banks = get_compile_time_arg_val(6);
constexpr auto input_args = TensorAccessorArgs<7>();

static_assert(slot_bytes >= in_page_bytes);
static_assert(read_nocs >= 1 && read_nocs <= 3);

template <typename Accessor>
FORCE_INLINE void issue_page_range(
    const Accessor& input, uint32_t first_page, uint32_t slots, uint32_t k_begin, uint32_t k_end) {
    uint32_t slot = slots + k_begin * slot_bytes;
    for (uint32_t k = k_begin; k < k_end; ++k) {
        const uint32_t page = first_page + k;
        if constexpr (read_nocs == 3) {
            uint8_t noc = noc_index ^ ((page / num_banks) & 1);
            while (!noc_cmd_buf_ready(noc, read_cmd_buf)) {
                noc ^= 1;
            }
            noc_async_read<in_page_bytes>(input.get_noc_addr(page, 0, noc), slot, in_page_bytes, noc);
        } else if constexpr (read_nocs == 2) {
            const uint8_t noc = noc_index ^ ((page / num_banks) & 1);
            noc_async_read<in_page_bytes>(input.get_noc_addr(page, 0, noc), slot, in_page_bytes, noc);
        } else {
            noc_async_read<in_page_bytes>(input.get_noc_addr(page), slot, in_page_bytes);
        }
        slot += slot_bytes;
    }
}

template <typename Accessor>
FORCE_INLINE void issue_block_reads(const Accessor& input, uint32_t& page, uint32_t num_pages, uint32_t stagger) {
    const uint32_t slots = get_write_ptr(cb_input_slots);
    const uint32_t first = stagger < num_pages ? stagger : stagger % num_pages;
    issue_page_range(input, page, slots, first, num_pages);
    issue_page_range(input, page, slots, 0, first);
    page += num_pages;
}

FORCE_INLINE void block_read_barrier() {
    noc_async_read_barrier(noc_index);
    if constexpr (read_nocs >= 2) {
        noc_async_read_barrier(1 - noc_index);
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block(const Accessor& input, uint32_t& page, uint32_t extent, uint32_t stagger) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, block_h));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_reads(input, page, extent * pages_per_unit, stagger));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", block_read_barrier());
    // Nominal count on a ragged last block too: the writer waits and pops the same quantum.
    cb_push_back(cb_input_slots, block_h);
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    const uint32_t stagger = get_arg_val<uint32_t>(2);
    uint32_t page = start_unit * pages_per_unit;

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);

    uint32_t block_idx = 0;
    for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
        const uint32_t extent = remaining < block_h ? remaining : block_h;
        load_block<true>(input, page, extent, stagger);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-RD-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_h ? remaining : block_h;
            load_block<false>(input, page, extent, stagger);
            remaining -= extent;
        }
    }
}

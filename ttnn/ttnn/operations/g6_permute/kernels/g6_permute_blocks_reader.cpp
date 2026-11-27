// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader (blocks_rm_gather): load_block.
// Unit (n, c, h) owns the pages_per_unit = W * I consecutive input pages of J values that start at
// unit * pages_per_unit. Every page lands in its own slot of slot_bytes (the page's bank stride), so
// local and remote address agree mod 64 as Blackhole DRAM reads require. One read barrier and one
// push per block.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t block_h = get_compile_time_arg_val(1);
constexpr uint32_t pages_per_unit = get_compile_time_arg_val(2);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(4);
constexpr auto input_args = TensorAccessorArgs<5>();

static_assert(slot_bytes >= in_page_bytes);

template <typename Accessor>
FORCE_INLINE void issue_block_reads(const Accessor& input, uint32_t& page, uint32_t num_pages) {
    uint32_t slot = get_write_ptr(cb_input_slots);
    for (uint32_t k = 0; k < num_pages; ++k) {
        noc_async_read<in_page_bytes>(input.get_noc_addr(page), slot, in_page_bytes);
        slot += slot_bytes;
        ++page;
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block(const Accessor& input, uint32_t& page, uint32_t extent) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, block_h));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_reads(input, page, extent * pages_per_unit));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", noc_async_read_barrier());
    // Nominal count on a ragged last block too: the writer waits and pops the same quantum.
    cb_push_back(cb_input_slots, block_h);
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    uint32_t page = start_unit * pages_per_unit;

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);

    uint32_t block_idx = 0;
    for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
        const uint32_t extent = remaining < block_h ? remaining : block_h;
        load_block<true>(input, page, extent);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-RD-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_h ? remaining : block_h;
            load_block<false>(input, page, extent);
            remaining -= extent;
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader of the ROW_MAJOR regimes (blocks_rm_gather, nhwc_rm_transpose): load_block.
// A unit is a run of consecutive input pages, and a core's units are consecutive, so a block is one
// page range. Units come in rows of units_per_row whose last unit holds tail_pages pages and the
// others pages_per_unit: blocks_rm_gather has one unit (n, c, h) of W * I pages per row;
// nhwc_rm_transpose cuts each row (n, h) of W pages into units of block_w pages and a shorter tail.
// Every page lands in its own slot of slot_bytes (the page's bank stride), so local and remote
// address agree mod 64 as Blackhole DRAM reads require. One read barrier and one push per block.
// Each page costs one request at the DRAM port it enters, and a port retires requests at a fixed
// rate, so the reader deals a share alt_num / alt_den of its pages to the second NoC, which enters
// every bank through a different port. Pages go to a NoC a whole bank round (num_banks consecutive
// pages, one per bank) at a time, so every bank sees the same split.
// bank_major issues a block chunk_pages at a time, each chunk bank by bank: pages k, k + num_banks, ...
// of one bank sit at consecutive bank addresses, so a core's requests to a bank arrive as one run, and
// one set_state per (bank, NoC) run serves every read of it. chunk_pages bounds the run length, which
// a DRAM port pays for in head-of-line blocking of the other cores' requests.
// stagger rotates the issue order of every block (pages, or banks when bank_major) so cores that start
// on the same bank do not walk the banks in lockstep.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t block_h = get_compile_time_arg_val(1);
constexpr uint32_t pages_per_unit = get_compile_time_arg_val(2);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(4);
constexpr uint32_t alt_num = get_compile_time_arg_val(5);
constexpr uint32_t alt_den = get_compile_time_arg_val(6);
constexpr uint32_t num_banks = get_compile_time_arg_val(7);
constexpr bool bank_major = get_compile_time_arg_val(8) != 0;
constexpr uint32_t chunk_pages = get_compile_time_arg_val(9);
constexpr uint32_t units_per_row = get_compile_time_arg_val(10);
constexpr uint32_t tail_pages = get_compile_time_arg_val(11);
constexpr auto input_args = TensorAccessorArgs<12>();

static_assert(slot_bytes >= in_page_bytes);
static_assert(!bank_major || chunk_pages % num_banks == 0, "a chunk is whole bank rounds");
static_assert(alt_num < alt_den && (alt_den & (alt_den - 1)) == 0, "alt share is a power-of-two fraction below 1");
static_assert(units_per_row >= 1 && tail_pages >= 1 && tail_pages <= pages_per_unit);

// Pages of the next `extent` units; row_pos is the next unit's position in its row.
FORCE_INLINE uint32_t block_pages(uint32_t extent, uint32_t& row_pos) {
    if constexpr (tail_pages == pages_per_unit) {
        return extent * pages_per_unit;
    } else {
        uint32_t pages = extent * pages_per_unit;
        for (uint32_t u = 0; u < extent; ++u) {
            if (++row_pos == units_per_row) {
                row_pos = 0;
                pages -= pages_per_unit - tail_pages;
            }
        }
        return pages;
    }
}

// Round r goes to the second NoC when a Bresenham accumulator stepping by alt_num wraps mod alt_den.
FORCE_INLINE uint8_t page_noc(uint32_t page) {
    if constexpr (alt_num == 0) {
        return noc_index;
    } else {
        const uint32_t round = page / num_banks;
        return ((round * alt_num) & (alt_den - 1)) >= alt_den - alt_num ? 1 - noc_index : noc_index;
    }
}

template <typename Accessor>
FORCE_INLINE void issue_page_range(
    const Accessor& input, uint32_t first_page, uint32_t slots, uint32_t k_begin, uint32_t k_end) {
    uint32_t slot = slots + k_begin * slot_bytes;
    for (uint32_t k = k_begin; k < k_end; ++k) {
        const uint32_t page = first_page + k;
        if constexpr (alt_num == 0) {
            noc_async_read<in_page_bytes>(input.get_noc_addr(page), slot, in_page_bytes);
        } else {
            const uint8_t noc = page_noc(page);
            noc_async_read<in_page_bytes>(input.get_noc_addr(page, 0, noc), slot, in_page_bytes, noc);
        }
        slot += slot_bytes;
    }
}

// count pages k, k + num_banks, ... of the block on one NoC; slot_bytes is also the bank address stride.
template <typename Accessor>
FORCE_INLINE void issue_bank_run(
    const Accessor& input, uint32_t first_page, uint32_t slots, uint32_t k, uint32_t count, uint8_t noc) {
    if (count == 0) {
        return;
    }
    const uint64_t src = input.get_noc_addr(first_page + k, 0, noc);
    noc_async_read_one_packet_set_state(src, in_page_bytes, 0, noc);
    uint32_t src_addr = static_cast<uint32_t>(src);
    uint32_t slot = slots + k * slot_bytes;
    for (uint32_t r = 0; r < count; ++r) {
        noc_async_read_one_packet_with_state(src_addr, slot, 0, noc);
        src_addr += slot_bytes;
        slot += num_banks * slot_bytes;
    }
}

// Each bank's run is split between the NoCs; alt_carry spreads the rounding of alt_num / alt_den over
// the runs of the core.
template <typename Accessor>
FORCE_INLINE void issue_chunk_bank_major(
    const Accessor& input,
    uint32_t first_page,
    uint32_t slots,
    uint32_t num_pages,
    uint32_t stagger,
    uint32_t& alt_carry) {
    const uint32_t runs = num_pages < num_banks ? num_pages : num_banks;
    for (uint32_t b = 0; b < runs; ++b) {
        const uint32_t k = (b + stagger) % runs;
        const uint32_t count = (num_pages - k + num_banks - 1) / num_banks;
        uint32_t alt_count = 0;
        if constexpr (alt_num != 0) {
            alt_carry += count * alt_num;
            alt_count = alt_carry / alt_den;
            alt_carry %= alt_den;
        }
        const uint32_t own_count = count - alt_count;
        issue_bank_run(input, first_page, slots, k, own_count, noc_index);
        issue_bank_run(input, first_page, slots, k + own_count * num_banks, alt_count, 1 - noc_index);
    }
}

template <typename Accessor>
FORCE_INLINE void issue_block_bank_major(
    const Accessor& input,
    uint32_t first_page,
    uint32_t slots,
    uint32_t num_pages,
    uint32_t stagger,
    uint32_t& alt_carry) {
    for (uint32_t done = 0; done < num_pages; done += chunk_pages) {
        const uint32_t left = num_pages - done;
        issue_chunk_bank_major(
            input,
            first_page + done,
            slots + done * slot_bytes,
            left < chunk_pages ? left : chunk_pages,
            stagger,
            alt_carry);
    }
}

template <typename Accessor>
FORCE_INLINE void issue_block_reads(
    const Accessor& input, uint32_t& page, uint32_t num_pages, uint32_t stagger, uint32_t& alt_carry) {
    const uint32_t slots = get_write_ptr(cb_input_slots);
    if constexpr (bank_major) {
        issue_block_bank_major(input, page, slots, num_pages, stagger, alt_carry);
    } else {
        const uint32_t first = stagger < num_pages ? stagger : stagger % num_pages;
        issue_page_range(input, page, slots, first, num_pages);
        issue_page_range(input, page, slots, 0, first);
    }
    page += num_pages;
}

FORCE_INLINE void block_read_barrier() {
    noc_async_read_barrier(noc_index);
    if constexpr (alt_num != 0) {
        noc_async_read_barrier(1 - noc_index);
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block(
    const Accessor& input, uint32_t& page, uint32_t num_pages, uint32_t stagger, uint32_t& alt_carry) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_slots, block_h));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_reads(input, page, num_pages, stagger, alt_carry));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", block_read_barrier());
    // Nominal count on a ragged last block too: the writer waits and pops the same quantum.
    cb_push_back(cb_input_slots, block_h);
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    const uint32_t stagger = get_arg_val<uint32_t>(2);
    uint32_t row_pos = get_arg_val<uint32_t>(3);
    uint32_t alt_carry = 0;

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);

    uint32_t block_idx = 0;
    for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
        const uint32_t extent = remaining < block_h ? remaining : block_h;
        load_block<true>(input, page, block_pages(extent, row_pos), stagger, alt_carry);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-RD-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_h ? remaining : block_h;
            load_block<false>(input, page, block_pages(extent, row_pos), stagger, alt_carry);
            remaining -= extent;
        }
    }
}

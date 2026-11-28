// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone writer (BRISC, NoC1): store_block over this core's run of page ids
// start_page + i * page_stride, then drain_writes.
//
// Raw dataflow by design: no kernel_lib helper writes a CB block to DRAM pages behind one flush per
// block (op_design.md, "Helpers considered and rejected"). It goes one level lower, to raw NoC
// command-buffer writes (g6_clone_issue.hpp), for a measured win the dataflow API cannot express:
// before each cb_wait_front the block's first write packet is fully programmed into write_cmd_buf
// (CTRL, source slot, destination, coordinate, length), so after the handshake the send is one store;
// the public *_set_state / *_with_state calls cannot program addresses without sending. The block's
// other packets keep CTRL (and the length, for whole pages of at most one packet), and the issue
// counters are bumped once per block. With the reader's lean issue this took one-page copies from
// 727-736 to 629-644 ns (case bench, Blackhole p150b; perf_experiments/issue_path); the armed write
// alone cut the post-handshake issue from 113 to 25 cycles. Do not swap these calls back for
// noc_async_write without re-measuring. DM_DYNAMIC_NOC (noc_split) keeps the dataflow API path.

#include "api/dataflow/dataflow_api.h"
#include "g6_clone_zones.hpp"
#include "g6_clone_page_pieces.hpp"
#include "g6_clone_issue.hpp"

namespace {

// The noc_split path (DM_DYNAMIC_NOC): dataflow API calls on both NoCs.
template <
    uint32_t cb_page_staging,
    uint32_t block_pages,
    uint8_t bank_endpoint_noc,
    uint32_t page_bytes,
    typename Accessor>
FORCE_INLINE void store_block_split(
    const Accessor& out_accessor,
    uint32_t first_page,
    uint32_t block_extent,
    uint32_t bank_vc,
    uint32_t other_noc_pages) {
    constexpr uint8_t other_noc = 1 - noc_index;
    cb_wait_front(cb_page_staging, block_pages);
    const uint32_t l1_read_addr = get_read_ptr(cb_page_staging);
    const uint64_t bank_noc_addr = out_accessor.get_noc_addr(first_page, 0, bank_endpoint_noc);
    const uint32_t own_extent = block_extent - (other_noc_pages < block_extent ? other_noc_pages : block_extent);
    const uint32_t own_bytes = own_extent * page_bytes;
    if (own_bytes != 0) {
        noc_async_write(l1_read_addr, bank_noc_addr, own_bytes, noc_index, bank_vc);
    }
    if (own_extent != block_extent) {
        noc_async_write(
            l1_read_addr + own_bytes,
            bank_noc_addr + own_bytes,
            (block_extent - own_extent) * page_bytes,
            other_noc,
            bank_vc);
    }
    noc_async_writes_flushed();
    noc_async_writes_flushed(other_noc);
    cb_pop_front(cb_page_staging, block_pages);
}

template <
    uint32_t cb_page_staging,
    uint32_t block_pages,
    bool logical_rows,
    uint32_t page_stride,
    uint8_t bank_endpoint_noc,
    bool bank_stagger,
    uint32_t page_bytes,
    typename Accessor,
    typename Walk>
FORCE_INLINE void store_block(
    const Accessor& out_accessor,
    Walk& walk,
    uint32_t first_page,
    uint32_t block_extent,
    uint32_t range_start,
    uint32_t range_pages,
    uint32_t bank_vc) {
    // Consecutive-page walks address each page on this NoC's endpoint table: bank_endpoint_noc's when it
    // names a NoC (a one-page plan on its bank's row), else this kernel's own.
    constexpr uint8_t page_endpoint_noc = bank_endpoint_noc < NUM_NOCS ? bank_endpoint_noc : noc_index;
    constexpr bool one_packet_pages = !logical_rows && page_bytes <= NOC_MAX_BURST_SIZE;
    const uint32_t l1_block = get_read_ptr(cb_page_staging);
    uint32_t packets = 1;
    if constexpr (page_stride != 1) {
        // A bank run: the block is one span in one bank.
        ASSERT(
            out_accessor.num_contiguous_pages(first_page, first_page + (block_extent - 1) * page_stride + 1) ==
            block_extent);
        const uint64_t bank_noc_addr = out_accessor.get_noc_addr(first_page, 0, bank_endpoint_noc);
        const uint32_t bytes = block_extent * page_bytes;
        const uint32_t first_len = bytes < NOC_MAX_BURST_SIZE ? bytes : NOC_MAX_BURST_SIZE;
        g6_issue::write_arm(bank_vc, l1_block, bank_noc_addr, first_len);
        {
            G6C_ZONE("g6c_wr_wait");
            cb_wait_front(cb_page_staging, block_pages);
        }
        G6C_ZONE("g6c_wr_issue");
        g6_issue::write_fire();
        if (bytes != first_len) {
            packets += g6_issue::write_span(l1_block + first_len, bank_noc_addr + first_len, bytes - first_len);
        }
    } else {
        // Page 0 of the block, wrapped like the loop below wraps.
        uint32_t page = first_page;
        if constexpr (bank_stagger) {
            if (page >= range_start + range_pages) {
                if (page == range_start + range_pages) {
                    walk = Walk(range_start);
                }
                page -= range_pages;
            }
        }
        const uint64_t page0_noc_addr = out_accessor.get_noc_addr(page, 0, page_endpoint_noc);
        if constexpr (logical_rows) {
            const g6_clone::PagePieces& pieces = walk.pieces();
            g6_issue::write_arm(bank_vc, l1_block + pieces.offset0, page0_noc_addr + pieces.offset0, pieces.length0);
        } else {
            g6_issue::write_arm(
                bank_vc, l1_block, page0_noc_addr, page_bytes < NOC_MAX_BURST_SIZE ? page_bytes : NOC_MAX_BURST_SIZE);
        }
        {
            G6C_ZONE("g6c_wr_wait");
            cb_wait_front(cb_page_staging, block_pages);
        }
        G6C_ZONE("g6c_wr_issue");
        g6_issue::write_fire();
        // The rest of page 0.
        if constexpr (logical_rows) {
            const g6_clone::PagePieces& pieces = walk.pieces();
            if (pieces.length1 != 0) {
                g6_issue::write_packet<true>(
                    l1_block + pieces.offset1, page0_noc_addr + pieces.offset1, pieces.length1);
                ++packets;
            }
            walk.advance();
        } else if constexpr (!one_packet_pages) {
            packets += g6_issue::write_span(
                l1_block + NOC_MAX_BURST_SIZE, page0_noc_addr + NOC_MAX_BURST_SIZE, page_bytes - NOC_MAX_BURST_SIZE);
        }
        // Pages 1.. of the block.
        uint32_t l1_read_addr = l1_block + page_bytes;
        for (uint32_t i = 1; i < block_extent; ++i) {
            page = first_page + i;
            if constexpr (bank_stagger) {
                if (page >= range_start + range_pages) {
                    if (page == range_start + range_pages) {
                        walk = Walk(range_start);
                    }
                    page -= range_pages;
                }
            }
            const uint64_t page_noc_addr = out_accessor.get_noc_addr(page, 0, page_endpoint_noc);
            if constexpr (logical_rows) {
                const g6_clone::PagePieces& pieces = walk.pieces();
                g6_issue::write_packet<true>(
                    l1_read_addr + pieces.offset0, page_noc_addr + pieces.offset0, pieces.length0);
                ++packets;
                if (pieces.length1 != 0) {
                    g6_issue::write_packet<true>(
                        l1_read_addr + pieces.offset1, page_noc_addr + pieces.offset1, pieces.length1);
                    ++packets;
                }
                walk.advance();
            } else if constexpr (one_packet_pages) {
                g6_issue::write_packet<false>(l1_read_addr, page_noc_addr, page_bytes);
                ++packets;
            } else {
                packets += g6_issue::write_span(l1_read_addr, page_noc_addr, page_bytes);
            }
            l1_read_addr += page_bytes;
        }
    }
    g6_issue::count_writes(packets);
    {
        G6C_ZONE("g6c_wr_flush");
        g6_issue::writes_flushed();
    }
    cb_pop_front(cb_page_staging, block_pages);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr bool logical_rows = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t page_stride = get_compile_time_arg_val(3);
    constexpr uint8_t bank_endpoint_noc = get_compile_time_arg_val(4);
    constexpr bool bank_stagger = get_compile_time_arg_val(5) != 0;
    constexpr bool noc_split = get_compile_time_arg_val(6) != 0;
    constexpr bool per_core_vc = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t walk_ct_base = 8;
    constexpr auto out_args = TensorAccessorArgs<walk_ct_base + g6_clone::PARTIAL_TILE_CT_ARGS>();
    constexpr uint32_t page_bytes = out_args.get_aligned_page_size();
    static_assert(page_stride == 1 || page_stride == NUM_DRAM_BANKS, "g6_clone: a run steps 1 page or 1 bank cycle");
    static_assert(page_stride == 1 || !logical_rows, "g6_clone: logical_rows walks consecutive page ids");
    static_assert(
        bank_endpoint_noc < NUM_NOCS || (page_stride == 1 && bank_endpoint_noc == NUM_NOCS),
        "g6_clone: bank_endpoint_noc names a NoC, or NUM_NOCS (own NoC) on a consecutive-page walk");
    static_assert(page_stride == 1 || !bank_stagger, "g6_clone: a bank run is already one bank");
    static_assert(!noc_split || page_stride != 1, "g6_clone: only a bank run splits over both NoCs");
    static_assert(!noc_split || NOC_MODE == DM_DYNAMIC_NOC, "g6_clone: issuing on both NoCs needs DM_DYNAMIC_NOC");
    static_assert(!per_core_vc || page_stride != 1, "g6_clone: only a bank run picks its VC per core");
    static_assert(noc_split || NOC_MODE == DM_DEDICATED_NOC, "g6_clone: raw writes need DM_DEDICATED_NOC");

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);
    const uint32_t rotate = bank_stagger ? get_arg_val<uint32_t>(2) : 0;
    const uint32_t bank_vc = per_core_vc ? get_arg_val<uint32_t>(3) : NOC_UNICAST_WRITE_VC;
    const uint32_t other_noc_pages = noc_split ? get_arg_val<uint32_t>(4) : 0;

    const auto out_accessor = TensorAccessor(out_args, out_addr);
    g6_clone::PartialTileWalk<walk_ct_base> walk(start_page + rotate);

    uint32_t first_page = start_page + rotate;
    uint32_t pages_left = num_pages_this_core;
    while (pages_left != 0) {
        const uint32_t block_extent = pages_left < block_pages ? pages_left : block_pages;
        if constexpr (noc_split) {
            store_block_split<cb_page_staging, block_pages, bank_endpoint_noc, page_bytes>(
                out_accessor, first_page, block_extent, bank_vc, other_noc_pages);
        } else {
            store_block<
                cb_page_staging,
                block_pages,
                logical_rows,
                page_stride,
                bank_endpoint_noc,
                bank_stagger,
                page_bytes>(out_accessor, walk, first_page, block_extent, start_page, num_pages_this_core, bank_vc);
        }
        first_page += block_extent * page_stride;
        pages_left -= block_extent;
    }

    // The next program starts right after this one returns: every write must be acked.
    {
        G6C_ZONE("g6c_wr_drain");
        if constexpr (noc_split) {
            noc_async_write_barrier();
            noc_async_write_barrier(1 - noc_index);
        } else {
            g6_issue::writes_barrier();
        }
    }
}

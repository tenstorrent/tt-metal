// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone reader, issue-path variant "lean" (perf experiment issue_path). Same CT/RT layout, split and
// traffic as the op's reader (kernels/g6_clone_reader.cpp); only the cost of issuing changes:
//  - first block: no cb_reserve_back. The firmware zeroes the CB's sync registers before every launch, so the
//    CB is empty at kernel start and holds cb_depth_blocks >= 1 blocks;
//  - rotate (RT arg 2) is read only with bank_stagger (the host passes 0 otherwise);
//  - consecutive-page walks issue raw read packets (g6_clone_issue.hpp): a full page of at most one packet costs
//    RET/TARG/COORD + SEND, its length programmed once per block; the issue counter is bumped once per block;
//  - the push reads the CB's tiles-received register before the barrier, so after the data lands it is one store.
// Raw NoC register writes bypass noc_async_read_page / noc_async_read (dataflow_api.h); there is no kernel_lib
// helper for this copy (op_design.md, "Helpers considered and rejected"). A bank run (one transfer per block)
// keeps noc_async_read: its issue cost is one request per 16 KB. DM_DYNAMIC_NOC (noc_split) keeps the op's code.

#include "api/dataflow/dataflow_api.h"
#include "g6_clone_zones.hpp"
#include "g6_clone_page_pieces.hpp"
#include "g6_clone_issue.hpp"

namespace {

template <
    uint32_t cb_page_staging,
    uint32_t block_pages,
    bool logical_rows,
    uint32_t page_stride,
    uint8_t bank_endpoint_noc,
    bool bank_stagger,
    bool noc_split,
    uint32_t page_bytes,
    typename Accessor,
    typename Walk>
FORCE_INLINE void load_block(
    const Accessor& x_accessor,
    Walk& walk,
    uint32_t first_page,
    uint32_t block_extent,
    uint32_t range_start,
    uint32_t range_pages,
    uint32_t bank_vc,
    uint32_t other_noc_pages,
    bool first_block) {
    constexpr uint8_t other_noc = 1 - noc_index;
    {
        G6C_ZONE("g6c_rd_reserve");
        if (!first_block) {
            cb_reserve_back(cb_page_staging, block_pages);
        }
    }
    uint32_t l1_write_addr = get_write_ptr(cb_page_staging);
    {
        G6C_ZONE("g6c_rd_issue");
        if constexpr (page_stride != 1) {
            ASSERT(
                x_accessor.num_contiguous_pages(first_page, first_page + (block_extent - 1) * page_stride + 1) ==
                block_extent);
            const uint64_t bank_noc_addr = x_accessor.get_noc_addr(first_page, 0, bank_endpoint_noc);
            if constexpr (noc_split) {
                const uint32_t own_extent =
                    block_extent - (other_noc_pages < block_extent ? other_noc_pages : block_extent);
                const uint32_t own_bytes = own_extent * page_bytes;
                if (own_bytes != 0) {
                    noc_async_read(bank_noc_addr, l1_write_addr, own_bytes, noc_index, bank_vc);
                }
                if (own_extent != block_extent) {
                    noc_async_read(
                        bank_noc_addr + own_bytes,
                        l1_write_addr + own_bytes,
                        (block_extent - own_extent) * page_bytes,
                        other_noc,
                        bank_vc);
                }
            } else {
                noc_async_read(bank_noc_addr, l1_write_addr, block_extent * page_bytes, noc_index, bank_vc);
            }
        } else {
            constexpr bool one_packet_pages = !logical_rows && page_bytes <= NOC_MAX_BURST_SIZE;
            if constexpr (one_packet_pages) {
                g6_issue::read_set_len(page_bytes);
            }
            uint32_t packets = 0;
            for (uint32_t i = 0; i < block_extent; ++i) {
                uint32_t page = first_page + i;
                if constexpr (bank_stagger) {
                    // The rotated walk wraps once, from the range's end back to its start.
                    if (page >= range_start + range_pages) {
                        if (page == range_start + range_pages) {
                            walk = Walk(range_start);
                        }
                        page -= range_pages;
                    }
                }
                const uint64_t page_noc_addr = x_accessor.get_noc_addr(page);
                if constexpr (logical_rows) {
                    const g6_clone::PagePieces& pieces = walk.pieces();
                    g6_issue::read_packet<true>(
                        page_noc_addr + pieces.offset0, l1_write_addr + pieces.offset0, pieces.length0);
                    ++packets;
                    if (pieces.length1 != 0) {
                        g6_issue::read_packet<true>(
                            page_noc_addr + pieces.offset1, l1_write_addr + pieces.offset1, pieces.length1);
                        ++packets;
                    }
                    walk.advance();
                } else if constexpr (one_packet_pages) {
                    g6_issue::read_packet<false>(page_noc_addr, l1_write_addr, page_bytes);
                    ++packets;
                } else {
                    packets += g6_issue::read_span(page_noc_addr, l1_write_addr, page_bytes);
                }
                l1_write_addr += page_bytes;
            }
            g6_issue::count_reads(packets);
        }
    }
    g6_issue::EarlyPush push(cb_page_staging, block_pages);
    {
        G6C_ZONE("g6c_rd_barrier");
        if constexpr (page_stride != 1) {
            noc_async_read_barrier();
            if constexpr (noc_split) {
                noc_async_read_barrier(other_noc);
            }
        } else {
            g6_issue::reads_barrier();
        }
    }
    push.push<cb_page_staging, block_pages>();
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
    constexpr auto x_args = TensorAccessorArgs<walk_ct_base + g6_clone::PARTIAL_TILE_CT_ARGS>();
    constexpr uint32_t page_bytes = x_args.get_aligned_page_size();
    static_assert(page_stride == 1 || page_stride == NUM_DRAM_BANKS, "g6_clone: a run steps 1 page or 1 bank cycle");
    static_assert(page_stride == 1 || !logical_rows, "g6_clone: logical_rows walks consecutive page ids");
    static_assert(bank_endpoint_noc < NUM_NOCS, "g6_clone: bank_endpoint_noc names a NoC");
    static_assert(page_stride == 1 || !bank_stagger, "g6_clone: a bank run is already one bank");
    static_assert(!noc_split || page_stride != 1, "g6_clone: only a bank run splits over both NoCs");
    static_assert(!noc_split || NOC_MODE == DM_DYNAMIC_NOC, "g6_clone: issuing on both NoCs needs DM_DYNAMIC_NOC");
    static_assert(!per_core_vc || page_stride != 1, "g6_clone: only a bank run picks its VC per core");
    static_assert(
        page_stride != 1 || NOC_MODE == DM_DEDICATED_NOC, "g6_clone issue_path: raw reads need DM_DEDICATED_NOC");

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);
    const uint32_t rotate = bank_stagger ? get_arg_val<uint32_t>(2) : 0;
    const uint32_t bank_vc = per_core_vc ? get_arg_val<uint32_t>(3) : NOC_UNICAST_WRITE_VC;
    const uint32_t other_noc_pages = noc_split ? get_arg_val<uint32_t>(4) : 0;

    const auto x_accessor = TensorAccessor(x_args, x_addr);
    g6_clone::PartialTileWalk<walk_ct_base> walk(start_page + rotate);

    uint32_t first_page = start_page + rotate;
    uint32_t pages_left = num_pages_this_core;
    bool first_block = true;
    while (pages_left != 0) {
        const uint32_t block_extent = pages_left < block_pages ? pages_left : block_pages;
        load_block<
            cb_page_staging,
            block_pages,
            logical_rows,
            page_stride,
            bank_endpoint_noc,
            bank_stagger,
            noc_split,
            page_bytes>(
            x_accessor,
            walk,
            first_page,
            block_extent,
            start_page,
            num_pages_this_core,
            bank_vc,
            other_noc_pages,
            first_block);
        first_page += block_extent * page_stride;
        pages_left -= block_extent;
        first_block = false;
    }
}

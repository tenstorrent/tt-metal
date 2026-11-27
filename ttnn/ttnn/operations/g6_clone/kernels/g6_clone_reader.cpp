// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone reader (NCRISC, NoC0): load_block over this core's run of page ids
// start_page + i * page_stride.
//
// Raw dataflow by design: no kernel_lib helper moves DRAM pages into a CB behind one
// barrier per block (op_design.md, "Helpers considered and rejected"). The bytes never
// reach an unpacker, so every bit pattern of every dtype survives.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_clone_page_pieces.hpp"

namespace {

// One block: reserve the nominal block, issue the block's reads back to back, one barrier,
// push the nominal block. A ragged block narrows only the reads, so every push is
// block_pages and the CB ring closes on its capacity (cb_depth_blocks * block_pages).
// A bank run (page_stride = bank count) sits back to back in one bank, so it is one read, addressed
// to the bank's endpoint on bank_endpoint_noc's table (the one on this core's row), with its request
// on virtual channel bank_vc. With noc_split, the block's last other_noc_pages pages come back on the
// other NoC instead.
// With logical_rows, each page moves only its logical pieces to their own in-page offsets.
template <
    uint32_t cb_page_staging,
    uint32_t block_pages,
    bool logical_rows,
    uint32_t page_stride,
    uint8_t bank_endpoint_noc,
    bool bank_stagger,
    bool noc_split,
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
    uint32_t other_noc_pages) {
    constexpr uint8_t other_noc = 1 - noc_index;
    {
        DeviceZoneScopedN("g6c_rd_reserve");
        cb_reserve_back(cb_page_staging, block_pages);
    }
    {
        DeviceZoneScopedN("g6c_rd_issue");
        const uint32_t page_bytes = x_accessor.get_aligned_page_size();
        uint32_t l1_write_addr = get_write_ptr(cb_page_staging);
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
                if constexpr (logical_rows) {
                    const g6_clone::PagePieces& pieces = walk.pieces();
                    const uint64_t page_noc_addr = x_accessor.get_noc_addr(page);
                    noc_async_read<NOC_MAX_BURST_SIZE>(
                        page_noc_addr + pieces.offset0, l1_write_addr + pieces.offset0, pieces.length0);
                    if (pieces.length1 != 0) {
                        noc_async_read<NOC_MAX_BURST_SIZE>(
                            page_noc_addr + pieces.offset1, l1_write_addr + pieces.offset1, pieces.length1);
                    }
                    walk.advance();
                } else {
                    noc_async_read_page(page, x_accessor, l1_write_addr);
                }
                l1_write_addr += page_bytes;
            }
        }
    }
    {
        DeviceZoneScopedN("g6c_rd_barrier");
        noc_async_read_barrier();
        if constexpr (noc_split) {
            noc_async_read_barrier(other_noc);
        }
    }
    cb_push_back(cb_page_staging, block_pages);
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
    static_assert(page_stride == 1 || page_stride == NUM_DRAM_BANKS, "g6_clone: a run steps 1 page or 1 bank cycle");
    static_assert(page_stride == 1 || !logical_rows, "g6_clone: logical_rows walks consecutive page ids");
    static_assert(bank_endpoint_noc < NUM_NOCS, "g6_clone: bank_endpoint_noc names a NoC");
    static_assert(page_stride == 1 || !bank_stagger, "g6_clone: a bank run is already one bank");
    static_assert(!noc_split || page_stride != 1, "g6_clone: only a bank run splits over both NoCs");
    static_assert(!noc_split || NOC_MODE == DM_DYNAMIC_NOC, "g6_clone: issuing on both NoCs needs DM_DYNAMIC_NOC");
    static_assert(!per_core_vc || page_stride != 1, "g6_clone: only a bank run picks its VC per core");

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);
    // bank_stagger: the walk starts `rotate` pages into the range, so the grid's first requests
    // spread over the banks, and wraps once (0 without bank_stagger).
    const uint32_t rotate = get_arg_val<uint32_t>(2);
    // Bank runs only: the request VC (RT arg 3 with per_core_vc, else the API default), and how many
    // pages of each block the other NoC carries (RT arg 4 with noc_split, else none).
    const uint32_t bank_vc = per_core_vc ? get_arg_val<uint32_t>(3) : NOC_UNICAST_WRITE_VC;
    const uint32_t other_noc_pages = noc_split ? get_arg_val<uint32_t>(4) : 0;

    // Page size comes from the accessor args (the buffer's aligned page), the same value
    // the host used as the CB page size.
    const auto x_accessor = TensorAccessor(x_args, x_addr);
    g6_clone::PartialTileWalk<walk_ct_base> walk(start_page + rotate);

    const uint32_t num_blocks_this_core = (num_pages_this_core + block_pages - 1) / block_pages;
    uint32_t first_page = start_page + rotate;
    uint32_t pages_left = num_pages_this_core;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_extent = pages_left < block_pages ? pages_left : block_pages;
        load_block<cb_page_staging, block_pages, logical_rows, page_stride, bank_endpoint_noc, bank_stagger, noc_split>(
            x_accessor, walk, first_page, block_extent, start_page, num_pages_this_core, bank_vc, other_noc_pages);
        first_page += block_extent * page_stride;
        pages_left -= block_extent;
    }
}

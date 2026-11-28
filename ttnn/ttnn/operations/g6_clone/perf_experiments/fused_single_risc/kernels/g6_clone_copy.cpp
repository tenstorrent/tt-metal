// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone fused copy kernel: ONE data-movement kernel per core, on one RISC, replaces the
// reader + writer pair. It reads a block of this core's run of page ids
// start_page + i * page_stride into an L1 slot, waits for the reads, writes the slot back out,
// and waits for the write acks before it returns (BB-03). No CB handshake and no second RISC.
//
// Raw dataflow by design, as in the reader / writer it replaces: no kernel_lib helper moves DRAM
// pages to DRAM pages through one L1 slot behind one barrier per direction. The staging CB is
// only the L1 allocation of the slots (get_write_ptr at kernel start = the CB's base); nothing
// pushes or pops it, since producer and consumer are the same RISC.
//
// NoCs: reads issue on read_noc and writes on write_noc (CT args). read_noc == write_noc runs in the
// dedicated mode of that NoC. read_noc != write_noc (the reader/writer pair's traffic shape: reads
// NoC0, writes NoC1) runs either
//   - in DM_DYNAMIC_NOC mode (the sanctioned two-NoC mode: every issue and barrier keeps its
//     counters in L1, shared with the other RISC), or
//   - with own_other_noc: in the dedicated mode of the kernel's NoC, the kernel taking the other
//     NoC's read / non-posted-write counters over at its start (as noc_local_state_init, which the
//     kernel wrapper runs for the kernel's own NoC before the go signal, does). Sound only because
//     this kernel is the core's ONLY data-movement kernel: the idle RISC's firmware issues nothing
//     on either NoC while it runs (it polls L1 for this RISC's done), so the NIU counters of the
//     other NoC move only with this kernel's transactions, and every one of them is barriered before
//     return. Raw LLK, bypassing the dynamic-mode L1 counters, which cost ~45 ns on a one-page copy
//     (Blackhole p150b, Q4: dynamic 715 ns, own_other 669-674 ns).
//
// Several blocks per core (no INPUTS shape; the L1 cap or BLOCKS_PER_CORE_KNOB): with
// cb_depth_blocks == 2 the slots alternate, and block k's writes overlap block k + 1's reads.
// Before block k + 1's reads land in the slot block k - 1 was written from, block k - 1's writes
// must have left L1 (flushed); the flush is taken before block k's writes issue, so it never
// waits on them.

#include "api/dataflow/dataflow_api.h"
#include "g6_clone_zones.hpp"
#include "g6_clone_page_pieces.hpp"

namespace {

// Per-direction transfer of one block between an L1 slot and DRAM. kWrite selects the direction.
// A bank run (page_stride = bank count) is contiguous in one bank, so it moves as one transfer,
// addressed to the bank's endpoint on bank_endpoint_noc's table; with noc_split, the block's last
// other_noc_pages pages move on the other NoC. A consecutive-page walk moves page by page (with
// logical_rows, only each page's logical pieces at their in-page offsets).
template <
    bool kWrite,
    uint8_t own_noc,
    uint32_t page_stride,
    uint8_t bank_endpoint_noc,
    bool logical_rows,
    bool bank_stagger,
    bool noc_split,
    typename Accessor,
    typename Walk>
FORCE_INLINE void move_block(
    const Accessor& accessor,
    Walk& walk,
    uint32_t l1_addr,
    uint32_t first_page,
    uint32_t block_extent,
    uint32_t range_start,
    uint32_t range_pages,
    uint32_t vc,
    uint32_t other_noc_pages) {
    constexpr uint8_t other_noc = 1 - own_noc;
    const uint32_t page_bytes = accessor.get_aligned_page_size();
    if constexpr (page_stride != 1) {
        ASSERT(
            accessor.num_contiguous_pages(first_page, first_page + (block_extent - 1) * page_stride + 1) ==
            block_extent);
        const uint64_t bank_noc_addr = accessor.get_noc_addr(first_page, 0, bank_endpoint_noc);
        if constexpr (noc_split) {
            const uint32_t own_extent =
                block_extent - (other_noc_pages < block_extent ? other_noc_pages : block_extent);
            const uint32_t own_bytes = own_extent * page_bytes;
            if (own_bytes != 0) {
                if constexpr (kWrite) {
                    noc_async_write(l1_addr, bank_noc_addr, own_bytes, own_noc, vc);
                } else {
                    noc_async_read(bank_noc_addr, l1_addr, own_bytes, own_noc, vc);
                }
            }
            if (own_extent != block_extent) {
                const uint32_t rest_bytes = (block_extent - own_extent) * page_bytes;
                if constexpr (kWrite) {
                    noc_async_write(l1_addr + own_bytes, bank_noc_addr + own_bytes, rest_bytes, other_noc, vc);
                } else {
                    noc_async_read(bank_noc_addr + own_bytes, l1_addr + own_bytes, rest_bytes, other_noc, vc);
                }
            }
        } else {
            if constexpr (kWrite) {
                noc_async_write(l1_addr, bank_noc_addr, block_extent * page_bytes, own_noc, vc);
            } else {
                noc_async_read(bank_noc_addr, l1_addr, block_extent * page_bytes, own_noc, vc);
            }
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
                const uint64_t page_noc_addr = accessor.get_noc_addr(page, 0, own_noc);
                if constexpr (kWrite) {
                    noc_async_write<NOC_MAX_BURST_SIZE>(
                        l1_addr + pieces.offset0, page_noc_addr + pieces.offset0, pieces.length0, own_noc);
                    if (pieces.length1 != 0) {
                        noc_async_write<NOC_MAX_BURST_SIZE>(
                            l1_addr + pieces.offset1, page_noc_addr + pieces.offset1, pieces.length1, own_noc);
                    }
                } else {
                    noc_async_read<NOC_MAX_BURST_SIZE>(
                        page_noc_addr + pieces.offset0, l1_addr + pieces.offset0, pieces.length0, own_noc);
                    if (pieces.length1 != 0) {
                        noc_async_read<NOC_MAX_BURST_SIZE>(
                            page_noc_addr + pieces.offset1, l1_addr + pieces.offset1, pieces.length1, own_noc);
                    }
                }
                walk.advance();
            } else {
                if constexpr (kWrite) {
                    noc_async_write_page(page, accessor, l1_addr, 0, 0, own_noc);
                } else {
                    noc_async_read_page(page, accessor, l1_addr, 0, own_noc);
                }
            }
            l1_addr += page_bytes;
        }
    }
}

template <uint8_t read_noc, bool noc_split>
FORCE_INLINE void wait_reads() {
    noc_async_read_barrier(read_noc);
    if constexpr (noc_split) {
        noc_async_read_barrier(1 - read_noc);
    }
}

template <uint8_t write_noc, bool noc_split>
FORCE_INLINE void flush_writes() {
    noc_async_writes_flushed(write_noc);
    if constexpr (noc_split) {
        noc_async_writes_flushed(1 - write_noc);
    }
}

template <uint8_t write_noc, bool noc_split>
FORCE_INLINE void wait_writes() {
    noc_async_write_barrier(write_noc);
    if constexpr (noc_split) {
        noc_async_write_barrier(1 - write_noc);
    }
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
    constexpr bool per_core_read_vc = get_compile_time_arg_val(7) != 0;
    constexpr bool per_core_write_vc = get_compile_time_arg_val(8) != 0;
    constexpr uint8_t read_noc = get_compile_time_arg_val(9);
    constexpr uint8_t write_noc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_depth_blocks = get_compile_time_arg_val(11);
    constexpr bool own_other_noc = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t walk_ct_base = 13;
    constexpr auto x_args = TensorAccessorArgs<walk_ct_base + g6_clone::PARTIAL_TILE_CT_ARGS>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(page_stride == 1 || page_stride == NUM_DRAM_BANKS, "g6_clone: a run steps 1 page or 1 bank cycle");
    static_assert(page_stride == 1 || !logical_rows, "g6_clone: logical_rows walks consecutive page ids");
    static_assert(bank_endpoint_noc < NUM_NOCS, "g6_clone: bank_endpoint_noc names a NoC");
    static_assert(read_noc < NUM_NOCS && write_noc < NUM_NOCS, "g6_clone: read_noc / write_noc name a NoC");
    static_assert(page_stride == 1 || !bank_stagger, "g6_clone: a bank run is already one bank");
    static_assert(!noc_split || page_stride != 1, "g6_clone: only a bank run splits over both NoCs");
    static_assert(
        (!noc_split && read_noc == write_noc) || NOC_MODE == DM_DYNAMIC_NOC || own_other_noc,
        "g6_clone: issuing on both NoCs from one kernel needs DM_DYNAMIC_NOC or own_other_noc");
    static_assert(!own_other_noc || NOC_MODE == DM_DEDICATED_NOC, "g6_clone: own_other_noc is a dedicated-mode build");
    static_assert(
        !(per_core_read_vc || per_core_write_vc) || page_stride != 1,
        "g6_clone: only a bank run picks its VC per core");
    static_assert(cb_depth_blocks == 1 || cb_depth_blocks == 2, "g6_clone: one or two staging slots");

    if constexpr (own_other_noc) {
        // The kernel wrapper synced noc_index's counters before the go signal. Take over, the same
        // way, only the other NoC's counters this kernel's barriers compare: its reads' (when reads
        // issue there) and its non-posted writes' (when writes do). A NoC split issues both
        // directions on both NoCs.
        constexpr uint8_t other_noc = 1 - noc_index;
        if constexpr (noc_split || read_noc == other_noc) {
            noc_reads_num_issued[other_noc] = NOC_STATUS_READ_REG(other_noc, NIU_MST_RD_RESP_RECEIVED);
        }
        if constexpr (noc_split || write_noc == other_noc) {
            const uint32_t writes_sent = NOC_STATUS_READ_REG(other_noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
            const uint32_t writes_acked = NOC_STATUS_READ_REG(other_noc, NIU_MST_WR_ACK_RECEIVED);
            noc_nonposted_writes_num_issued[other_noc] = writes_sent;
            noc_nonposted_writes_acked[other_noc] = writes_acked;
        }
    }

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);
    // bank_stagger: the walk starts `rotate` pages into the range and wraps once (0 without it).
    const uint32_t rotate = get_arg_val<uint32_t>(2);
    // Bank runs only: per-direction request VCs (RT args 3, 4 with per_core_*_vc, else the API
    // default) and how many pages of each block the other NoC carries (RT arg 5 with noc_split).
    const uint32_t read_vc = per_core_read_vc ? get_arg_val<uint32_t>(3) : NOC_UNICAST_WRITE_VC;
    const uint32_t write_vc = per_core_write_vc ? get_arg_val<uint32_t>(4) : NOC_UNICAST_WRITE_VC;
    const uint32_t other_noc_pages = noc_split ? get_arg_val<uint32_t>(5) : 0;

    const auto x_accessor = TensorAccessor(x_args, x_addr);
    const auto out_accessor = TensorAccessor(out_args, out_addr);
    g6_clone::PartialTileWalk<walk_ct_base> read_walk(start_page + rotate);
    g6_clone::PartialTileWalk<walk_ct_base> write_walk(start_page + rotate);

    // The slots: the staging CB's L1, untouched by any push / pop since the program started.
    const uint32_t slot0_addr = get_write_ptr(cb_page_staging);
    const uint32_t slot_bytes = block_pages * x_accessor.get_aligned_page_size();

    const uint32_t num_blocks_this_core = (num_pages_this_core + block_pages - 1) / block_pages;
    uint32_t read_first_page = start_page + rotate;
    uint32_t write_first_page = read_first_page;
    uint32_t pages_left_to_read = num_pages_this_core;

    // Block 0 in.
    uint32_t read_slot = 0;
    uint32_t read_extent = pages_left_to_read < block_pages ? pages_left_to_read : block_pages;
    {
        G6C_ZONE("g6c_fu_rd_issue");
        move_block<false, read_noc, page_stride, bank_endpoint_noc, logical_rows, bank_stagger, noc_split>(
            x_accessor,
            read_walk,
            slot0_addr,
            read_first_page,
            read_extent,
            start_page,
            num_pages_this_core,
            read_vc,
            other_noc_pages);
    }
    {
        G6C_ZONE("g6c_fu_rd_barrier");
        wait_reads<read_noc, noc_split>();
    }

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        // Block block_idx is in slot read_slot; write it out.
        const uint32_t write_slot_addr = slot0_addr + read_slot * slot_bytes;
        const uint32_t write_extent = read_extent;
        if constexpr (cb_depth_blocks == 2) {
            if (block_idx != 0) {
                // Block block_idx - 1's writes must have left the other slot before the next reads
                // land there; they were issued a whole read ago, so this rarely waits.
                G6C_ZONE("g6c_fu_wr_flush");
                flush_writes<write_noc, noc_split>();
            }
        }
        {
            G6C_ZONE("g6c_fu_wr_issue");
            move_block<true, write_noc, page_stride, bank_endpoint_noc, logical_rows, bank_stagger, noc_split>(
                out_accessor,
                write_walk,
                write_slot_addr,
                write_first_page,
                write_extent,
                start_page,
                num_pages_this_core,
                write_vc,
                other_noc_pages);
        }
        write_first_page += write_extent * page_stride;
        read_first_page += read_extent * page_stride;
        pages_left_to_read -= read_extent;
        if (pages_left_to_read != 0) {
            if constexpr (cb_depth_blocks == 1) {
                // One slot: this block's writes must leave it before the next block's reads land.
                G6C_ZONE("g6c_fu_wr_flush");
                flush_writes<write_noc, noc_split>();
            } else {
                read_slot ^= 1;
            }
            read_extent = pages_left_to_read < block_pages ? pages_left_to_read : block_pages;
            {
                G6C_ZONE("g6c_fu_rd_issue");
                move_block<false, read_noc, page_stride, bank_endpoint_noc, logical_rows, bank_stagger, noc_split>(
                    x_accessor,
                    read_walk,
                    slot0_addr + read_slot * slot_bytes,
                    read_first_page,
                    read_extent,
                    start_page,
                    num_pages_this_core,
                    read_vc,
                    other_noc_pages);
            }
            {
                G6C_ZONE("g6c_fu_rd_barrier");
                wait_reads<read_noc, noc_split>();
            }
        }
    }

    // The next program starts right after this one returns: every write must be acked.
    {
        G6C_ZONE("g6c_fu_wr_drain");
        wait_writes<write_noc, noc_split>();
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post split_noc_reads candidate — ONE data-movement kernel source, instantiated on BOTH DM RISCs
// (NCRISC on NoC0 and BRISC on NoC1 by default; the host picks). Each instance owns, by CT args:
//   reads : F (optional, into cb_f) and X streams [x_lo, x_hi) (into cb_x, slot (i - x_lo)*B + c)
//   writes: output streams [w_lo, w_hi) (from cb_out, slot (j - w_lo)*B + c)
//   expand: the coefficient read + expansion (single producer of cb_coef_bcast; needs writes)
// Every CB keeps exactly one producer and one consumer: the host gives each RISC its own X / output CB and
// the compute kernel (split_compute.cpp) waits / reserves on both.
//
// Schedule per block (same SegmentWalker derivation as compute):
//   reads (if any): reserve, issue all, ONE read barrier, push   — identical to the op's reader
//   writes (if any): when the instance also reads, the write of block b is done after block b+1's reads are
//                    pushed (one-block lag: the output of block b cannot exist before the reads of b are
//                    pushed, so waiting on it right after them would serialize read(b+1) behind compute(b));
//                    a write-only instance writes block b in-order, exactly like the op's writer.
//   expand: segment 0's set up front (write-only instance) or folded into block 0's read barrier (read+write
//           instance); segment s+1's set right after segment s's first block is written (op's look-ahead).
// No raw LLK: plain dataflow_api (noc_async_read / write, CB sync).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"
#include "mhc_skip_noc.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr bool read_f = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t x_lo = get_compile_time_arg_val(4);
    constexpr uint32_t x_hi = get_compile_time_arg_val(5);
    constexpr uint32_t w_lo = get_compile_time_arg_val(6);
    constexpr uint32_t w_hi = get_compile_time_arg_val(7);
    constexpr bool expand_here = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t cb_f = get_compile_time_arg_val(9);
    constexpr uint32_t cb_x = get_compile_time_arg_val(10);
    constexpr uint32_t cb_out = get_compile_time_arg_val(11);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(12);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(13);
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(14);
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(15);
    constexpr uint32_t f_page_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t coef_page_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(19);
    // Alternate-NoC routing (needs NOC_MODE::DM_DYNAMIC_NOC on both DM kernels): F / X streams >= x_alt_from /
    // output streams >= w_alt_from go on the OTHER NoC (1 - noc_index); barriers then cover both NoCs.
    constexpr bool f_alt = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t x_alt_from = get_compile_time_arg_val(21);
    constexpr uint32_t w_alt_from = get_compile_time_arg_val(22);
    constexpr uint32_t sched = get_compile_time_arg_val(23);  // read+write instance: 0 lag-1, 1/2 event loop
    constexpr auto f_args = TensorAccessorArgs<24>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_x = x_hi - x_lo;
    constexpr uint32_t num_w = w_hi - w_lo;
    constexpr bool has_reads = read_f || num_x > 0;
    constexpr bool has_writes = num_w > 0;
    constexpr bool reads_alt = (read_f && f_alt) || x_alt_from < x_hi;
    constexpr bool writes_alt = w_alt_from < w_hi;
    const uint8_t my_noc = noc_index;
    const uint8_t alt_noc = 1 - noc_index;
    constexpr uint32_t row_tiles = n * col_tiles_per_row;  // X / X' page stride per token row
    static_assert(!expand_here || has_writes, "split_dm: the coefficient expander must also own writes");

    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t post_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_unit = get_arg_val<uint32_t>(5);
    const uint32_t num_units = get_arg_val<uint32_t>(6);

    const auto f_acc = TensorAccessor(f_args, f_addr, f_page_bytes);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);
    const auto o_acc = TensorAccessor(o_args, o_addr, x_page_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page_bytes);
    mhc_post::CoefExpander<
        n,
        coef_tiles_per_stream,
        post_tiles_per_row,
        comb_tiles_per_row,
        coef_page_bytes,
        cb_coef_raw,
        cb_coef_bcast,
        decltype(post_acc)>
        coefs(post_acc, comb_acc);

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    mhc_post::SegmentWalker ahead = walker;  // coefficient look-ahead (same derivation)

    struct Pending {
        uint32_t row, col_start, valid;
        bool first_of_segment;
    };

    auto write_block = [&](const Pending& p) {
        {
            MaybeDeviceZoneScope("dm_write_wait");
            cb_wait_front(cb_out, num_w * block_col_tiles);
        }
        {
            MaybeDeviceZoneScope("dm_write_issue");
            const uint32_t base = get_read_ptr(cb_out);
            for (uint32_t j = w_lo; j < w_hi; ++j) {
                const uint32_t page0 = p.row * row_tiles + j * col_tiles_per_row + p.col_start;
                const uint32_t slot0 = base + (j - w_lo) * block_col_tiles * x_page_bytes;
                const uint8_t noc = j >= w_alt_from ? alt_noc : my_noc;
                for (uint32_t c = 0; c < p.valid; ++c) {
                    data_write(slot0 + c * x_page_bytes, o_acc.get_noc_addr(page0 + c, 0, noc), x_page_bytes, noc);
                }
            }
        }
        {
            MaybeDeviceZoneScope("dm_write_barrier");
            noc_async_write_barrier(my_noc);
            if constexpr (writes_alt) {
                noc_async_write_barrier(alt_noc);
            }
        }
        cb_pop_front(cb_out, num_w * block_col_tiles);
        if constexpr (expand_here) {
            if (p.first_of_segment && !ahead.done()) {
                MaybeDeviceZoneScope("dm_coef_load");
                coefs.load(ahead.next().row);  // next segment's set, ahead of compute
            }
        }
    };

    // Block iterator over the same SegmentWalker derivation (one per direction in the event-loop schedule).
    struct BlockIter {
        mhc_post::SegmentWalker w;
        mhc_post::Segment seg{0, 0, 0};
        uint32_t blocks = 0, idx = 0;
        explicit BlockIter(const mhc_post::SegmentWalker& walker) : w(walker) {}
        bool done() const { return idx >= blocks && w.done(); }
        Pending next() {
            if (idx >= blocks) {
                seg = w.next();
                blocks = mhc_post::num_blocks(seg.col_tiles, block_col_tiles);
                idx = 0;
            }
            const uint32_t b = idx++;
            return Pending{
                seg.row,
                seg.col0 + b * block_col_tiles,
                mhc_post::block_valid_col_tiles(seg.col_tiles, block_col_tiles, b),
                b == 0};
        }
    };

    bool first_block = true;
    auto read_block = [&](const Pending& p) {
        {
            MaybeDeviceZoneScope("dm_read_reserve");
            if constexpr (read_f) {
                cb_reserve_back(cb_f, block_col_tiles);
            }
            if constexpr (num_x > 0) {
                cb_reserve_back(cb_x, num_x * block_col_tiles);
            }
        }
        {
            MaybeDeviceZoneScope("dm_read_issue");
            if constexpr (read_f) {
                const uint32_t f_base = get_write_ptr(cb_f);
                const uint32_t f_page0 = p.row * col_tiles_per_row + p.col_start;
                const uint8_t noc = f_alt ? alt_noc : my_noc;
                for (uint32_t c = 0; c < p.valid; ++c) {
                    data_read(f_acc.get_noc_addr(f_page0 + c, 0, noc), f_base + c * f_page_bytes, f_page_bytes, noc);
                }
            }
            if constexpr (num_x > 0) {
                const uint32_t x_base = get_write_ptr(cb_x);
                for (uint32_t i = x_lo; i < x_hi; ++i) {
                    const uint32_t x_page0 = p.row * row_tiles + i * col_tiles_per_row + p.col_start;
                    const uint32_t x_slot0 = x_base + (i - x_lo) * block_col_tiles * x_page_bytes;
                    const uint8_t noc = i >= x_alt_from ? alt_noc : my_noc;
                    for (uint32_t c = 0; c < p.valid; ++c) {
                        data_read(
                            x_acc.get_noc_addr(x_page0 + c, 0, noc), x_slot0 + c * x_page_bytes, x_page_bytes, noc);
                    }
                }
            }
            if constexpr (expand_here) {
                if (first_block) {
                    coefs.issue_raw_reads(ahead.next().row);  // shares block 0's barrier
                }
            }
        }
        {
            MaybeDeviceZoneScope("dm_read_barrier");
            noc_async_read_barrier(my_noc);
            if constexpr (reads_alt) {
                noc_async_read_barrier(alt_noc);
            }
        }
        if constexpr (read_f) {
            cb_push_back(cb_f, block_col_tiles);
        }
        if constexpr (num_x > 0) {
            cb_push_back(cb_x, num_x * block_col_tiles);
        }
        if constexpr (expand_here) {
            if (first_block) {
                MaybeDeviceZoneScope("dm_coef_load");
                coefs.expand();
            }
        }
        first_block = false;
    };

    if constexpr (expand_here && !has_reads) {
        if (!ahead.done()) {
            MaybeDeviceZoneScope("dm_coef_load");
            coefs.load(ahead.next().row);
        }
    }

    BlockIter rd(walker);
    if constexpr (!(has_reads && has_writes)) {
        // single-direction instance: the op's in-order loop
        while (!rd.done()) {
            const Pending p = rd.next();
            if constexpr (has_reads) {
                read_block(p);
            }
            if constexpr (has_writes) {
                write_block(p);
            }
        }
    } else if constexpr (sched == 0) {
        // lag-1: write block b after reading block b+1
        Pending pending{0, 0, 0, false};
        bool have_pending = false;
        while (!rd.done()) {
            const Pending p = rd.next();
            read_block(p);
            if (have_pending) {
                write_block(pending);
            }
            pending = p;
            have_pending = true;
        }
        if (have_pending) {
            write_block(pending);
        }
    } else {
        // event loop: whichever side is ready (sched 1 = reads first, 2 = writes first on a tie)
        BlockIter wr(walker);
        auto read_ready = [&]() {
            bool ok = true;
            if constexpr (read_f) {
                ok = cb_pages_reservable_at_back(cb_f, block_col_tiles);
            }
            if constexpr (num_x > 0) {
                ok = ok && cb_pages_reservable_at_back(cb_x, num_x * block_col_tiles);
            }
            return ok;
        };
        auto write_ready = [&]() { return cb_pages_available_at_front(cb_out, num_w * block_col_tiles); };
        uint32_t reads_issued = 0, writes_done = 0;
        while (!rd.done() || !wr.done()) {
            const bool can_read = !rd.done() && read_ready();
            // a write needs its block's reads pushed first (compute cannot have produced it otherwise)
            const bool can_write = !wr.done() && writes_done < reads_issued && write_ready();
            if (can_read && (sched == 1 || !can_write)) {
                read_block(rd.next());
                ++reads_issued;
            } else if (can_write) {
                write_block(wr.next());
                ++writes_done;
            }
        }
    }
}

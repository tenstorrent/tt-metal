// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post data movement — ONE source for both DM RISCs (CT arg `role`). Perf 1 (split_noc_reads) replaced the
// separate reader / writer kernels with this pair.
//
//   role 0 (NCRISC, NoC0) — the reader. Sole producer of cb_sublayer_tiles / cb_residual_tiles. Per block of B
//        columns: reserve both windows, read the n X streams (and F, unless the block is read-helped), ONE
//        barrier, wait for the helper's F reads if helped, nominal pushes.
//   role 1 (BRISC, NoC1) — the writer. Sole consumer of cb_output_tiles and sole producer of cb_coef_raw /
//        cb_coef_bcast (load_coefficients, mhc_post_coef_expand.hpp). Optionally the READ HELPER: it reads the F
//        block of every read-helped block (block index >= help_from) straight into the window the reader
//        reserved, on NoC0 (dynamic-NoC mode), and never touches cb_sublayer_tiles' CB API. The window address
//        is derived, not exchanged: every reserve of cb_sublayer_tiles is the full nominal block and the CB is
//        exactly depth_in blocks, so block k lives at fifo_base + (k % depth_in) * B * f_page.
//        Hand-off: two local L1 semaphores holding monotonic block counters — rd_go (reader -> helper: window k
//        reserved) and rd_done (helper -> reader: F of block k landed).
//        The writer has three independent duties (help requests, output blocks, coefficient loads), so it
//        runs a non-blocking event loop; with read help the coefficient load is a resumable job expanded one
//        term per pass, so a pending help request waits at most one half-tile expansion.
//
// Why the help is F on NoC0 from block 1 (measured, Blackhole p150, 110 cores, bf16, device kernel ns): the DM
// floor is DRAM-bound and per-core DRAM service is uneven; a second RISC issuing a share of the reads on the
// reader's NoC shortens the slow cores. T1280 C4096 268 -> 246 us, T1024 C5120 275 -> 245 us, T640 C7168
// 218 -> 213 us. X-stream help measured the same (within 2%); write help (NCRISC writing output streams) and
// any help on NoC1 measured slower. Help from block 0 regressed (it delays the first coefficient set).
// The host enables help only when (a) the busiest core has >= HELP_MIN_BLOCKS blocks — measured regression with
// 3-5 blocks per core: T640 C1792 +18%, T256 C1792 +7%, T2048 C1792 +4% — and (b) X is not float32 (compute-bound
// datapath: T1000 C7168 X fp32 / F bf16 +3.2% with help). With help off this kernel measures within +-2% of the
// former reader / writer pair.
// No raw LLK: dataflow_api only (noc_async_*, CB sync, L1 semaphores).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"

namespace {
FORCE_INLINE uint32_t sem_read(uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}
FORCE_INLINE void sem_write(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = v; }

struct Blk {
    uint32_t row, col_start, valid;
};

// Block sequence of one core's unit range (segments -> blocks of B columns), same derivation on every RISC.
template <uint32_t B>
struct BlockIter {
    mhc_post::SegmentWalker w;
    mhc_post::Segment seg{0, 0, 0};
    uint32_t blocks = 0, idx = 0;
    explicit BlockIter(const mhc_post::SegmentWalker& walker) : w(walker) {}
    bool done() const { return idx >= blocks && w.done(); }
    Blk next() {
        if (idx >= blocks) {
            seg = w.next();
            blocks = mhc_post::num_blocks(seg.col_tiles, B);
            idx = 0;
        }
        const uint32_t b = idx++;
        return Blk{seg.row, seg.col0 + b * B, mhc_post::block_valid_col_tiles(seg.col_tiles, B, b)};
    }
};
}  // namespace

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t n = get_compile_time_arg_val(0);                  // streams
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);  // Ct = C / 32
    constexpr uint32_t B = get_compile_time_arg_val(2);                  // block_col_tiles
    constexpr uint32_t role = get_compile_time_arg_val(3);               // 0 reader (NCRISC), 1 writer (BRISC)
    constexpr bool read_help = get_compile_time_arg_val(4) != 0;         // BRISC reads F of helped blocks
    constexpr uint32_t help_from = get_compile_time_arg_val(5);          // first helped block index (per core)
    constexpr uint32_t depth_in = get_compile_time_arg_val(6);           // blocks in cb_sublayer_tiles
    constexpr uint32_t cb_f = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(10);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(11);
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(12);  // ceil(n / 32)
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(13);  // ceil(n*n / 32)
    constexpr uint32_t f_page = get_compile_time_arg_val(14);
    constexpr uint32_t x_page = get_compile_time_arg_val(15);  // X and X' (same dtype, same page)
    constexpr uint32_t coef_page = get_compile_time_arg_val(16);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(17);  // P = ceil((n+1)/2)
    constexpr uint32_t sem_rd_go = get_compile_time_arg_val(18);
    constexpr uint32_t sem_rd_done = get_compile_time_arg_val(19);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(20);  // 32
    constexpr auto f_args = TensorAccessorArgs<21>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * mhc_post::FACE_HW, "mhc_post: expansion assumes 32x32 tiles of 16x16 faces");

    constexpr uint32_t row_tiles = n * col_tiles_per_row;
    constexpr uint32_t f_block_bytes = B * f_page;

    // ---- runtime args (identical on both RISCs) ----
    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t post_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_unit = get_arg_val<uint32_t>(5);
    const uint32_t num_units = get_arg_val<uint32_t>(6);

    const uint32_t rd_go = get_semaphore(sem_rd_go);
    const uint32_t rd_done = get_semaphore(sem_rd_done);

    // F reads of one block into the window at f_win, on `noc` (no barrier).
    const auto f_acc = TensorAccessor(f_args, f_addr, f_page);
    auto issue_f_reads = [&](const Blk& p, uint32_t f_win, uint8_t noc) {
        const uint32_t f_page0 = p.row * col_tiles_per_row + p.col_start;
        for (uint32_t c = 0; c < p.valid; ++c) {
            noc_async_read(f_acc.get_noc_addr(f_page0 + c, 0, noc), f_win + c * f_page, f_page, noc);
        }
    };

    const mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);

    if constexpr (role == 0) {
        // ================= reader =================
        const auto x_acc = TensorAccessor(x_args, x_addr, x_page);
        BlockIter<B> rd(walker);
        for (uint32_t k = 0; !rd.done(); ++k) {
            {
                MaybeDeviceZoneScope("reader_reserve");  // back-pressure from compute
                cb_reserve_back(cb_f, B);
                cb_reserve_back(cb_x, n * B);
            }
            const Blk p = rd.next();
            const bool helped = read_help && k >= help_from;
            if (helped) {
                sem_write(rd_go, k + 1);
            }
            {
                MaybeDeviceZoneScope("reader_issue");
                if (!helped) {
                    issue_f_reads(p, get_write_ptr(cb_f), noc_index);
                }
                const uint32_t x_base = get_write_ptr(cb_x);
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t x_page0 = p.row * row_tiles + i * col_tiles_per_row + p.col_start;
                    const uint32_t slot0 = x_base + i * B * x_page;
                    for (uint32_t c = 0; c < p.valid; ++c) {
                        noc_async_read(x_acc.get_noc_addr(x_page0 + c), slot0 + c * x_page, x_page);
                    }
                }
            }
            {
                MaybeDeviceZoneScope("reader_barrier");
                noc_async_read_barrier();
            }
            if (helped) {
                MaybeDeviceZoneScope("reader_wait_help");  // the helper's F reads of block k
                while (sem_read(rd_done) < k + 1) {
                }
            }
            cb_push_back(cb_f, B);
            cb_push_back(cb_x, n * B);
        }
    } else {
        // ================= writer (+ coefficient expander, + read helper) =================
        const auto o_acc = TensorAccessor(o_args, o_addr, x_page);
        const auto post_acc = TensorAccessor(post_args, post_addr, coef_page);
        const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page);
        mhc_post::CoefExpander<
            n,
            coef_tiles_per_stream,
            post_tiles_per_row,
            comb_tiles_per_row,
            coef_page,
            cb_coef_raw,
            cb_coef_bcast,
            decltype(post_acc)>
            coefs(post_acc, comb_acc);
        const uint8_t help_noc = 1 - noc_index;  // the reader's NoC (dynamic-NoC mode when read_help)

        // Coefficient loads (Perf 2, coef_prefetch: EAGER look-ahead): segment 0's set up front, then segment s+1's
        // as soon as set s's job is finished and cb_coef_bcast has room for a whole set (non-blocking check — the
        // event loop never blocks on a reserve; COEF_DEPTH = 2 sets in flight). The former schedule started set
        // s+1 only after writing segment s's first output block, which stalled row-straddling cores on their second
        // set (measured compute_wait_coef 7.8 us at T640 C1792, 15.3 us on the slowest T1280 C4096 core; eager:
        // T512 C2560 83.3 -> 71.5 us, T1024 C1792 113.8 -> 108.5 us). The look-ahead walker has the same derivation.
        // With read help the set is expanded one term per event-loop pass; without, in one go (it then has nothing
        // to interleave with, and a set that lands sooner feeds short segments sooner).
        constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
        mhc_post::SegmentWalker ahead = walker;  // non-empty: the host gives every core >= 1 unit
        auto coef_pending = [&]() { return coefs.active() || !ahead.done(); };
        auto coef_progress = [&]() {
            if (!coefs.active() && !ahead.done() && cb_pages_reservable_at_back(cb_coef_bcast, num_coef_tiles)) {
                MaybeDeviceZoneScope("writer_coef_start");  // raw post / comb read + barrier
                coefs.start(ahead.next().row);
            }
            if constexpr (read_help) {
                if (coefs.active()) {
                    coefs.step();  // one term per pass: a pending help request waits at most one half-tile
                }
            } else if (coefs.active()) {
                MaybeDeviceZoneScope("writer_coef_expand");  // nothing to interleave with: the whole set at once
                while (coefs.active()) {
                    coefs.step();
                }
            }
        };
        coef_progress();

        BlockIter<B> wr(walker);
        BlockIter<B> rh(walker);  // blocks the helper reads F for
        for (uint32_t k = 0; k < help_from && !rh.done(); ++k) {
            rh.next();  // the reader reads these blocks alone
        }
        uint32_t helped_k = help_from;  // next helped block index
        const uint32_t f_base = get_write_ptr(cb_f);
        while (!wr.done() || (read_help && !rh.done()) || coef_pending()) {
            if constexpr (read_help) {
                if (!rh.done() && sem_read(rd_go) >= helped_k + 1) {
                    {
                        MaybeDeviceZoneScope("writer_help_read");
                        issue_f_reads(rh.next(), f_base + (helped_k % depth_in) * f_block_bytes, help_noc);
                        noc_async_read_barrier(help_noc);
                    }
                    ++helped_k;
                    sem_write(rd_done, helped_k);
                }
            }
            if (!wr.done() && cb_pages_available_at_front(cb_out, n * B)) {
                cb_wait_front(cb_out, n * B);
                const Blk p = wr.next();
                {
                    MaybeDeviceZoneScope("writer_issue");
                    const uint32_t out_base = get_read_ptr(cb_out);
                    for (uint32_t j = 0; j < n; ++j) {
                        const uint32_t page0 = p.row * row_tiles + j * col_tiles_per_row + p.col_start;
                        const uint32_t slot0 = out_base + j * B * x_page;
                        for (uint32_t c = 0; c < p.valid; ++c) {
                            noc_async_write(slot0 + c * x_page, o_acc.get_noc_addr(page0 + c), x_page);
                        }
                    }
                }
                {
                    MaybeDeviceZoneScope("writer_barrier");
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_out, n * B);
            }
            if (coef_pending()) {
                coef_progress();
            }
        }
    }
}

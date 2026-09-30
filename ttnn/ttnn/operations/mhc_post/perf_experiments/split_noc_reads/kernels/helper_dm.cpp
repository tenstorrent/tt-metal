// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post split_noc_reads candidate "helper DMA" — ONE DM kernel source for both DM RISCs, CB topology and the
// compute kernel IDENTICAL to the op (cb_sublayer / cb_residual produced by the reader, cb_output consumed by the
// writer; the op's mhc_post_compute.cpp runs unchanged). The idle half of each DM RISC lends its NoC issue to the
// other side as a plain DMA helper that never touches the CB API of the CB it helps:
//
//   role 0 (NCRISC, primary reader): owns cb_sublayer / cb_residual (reserve / push). Reads F (unless f_help) and
//        X streams [0, x_help_lo). Optionally HELPS the writer: writes output streams [w_help_lo, n) of the block
//        the writer has at its front.
//   role 1 (BRISC, primary writer): owns cb_output (wait / pop) and the coefficient expansion. Writes output
//        streams [0, w_help_lo). Optionally HELPS the reader: reads X streams [x_help_lo, n) (and F if f_help) into
//        the window the reader has reserved.
//
// The helper derives the window address itself: every reserve / wait of the helped CB is the full nominal block,
// so block k lives at fifo_base + (k % depth) * block_bytes. Hand-off through 4 local L1 semaphores (monotonic block
// counters): rd_go (reader -> helper: window k reserved), rd_done (helper -> reader: its reads of k landed),
// wr_go (writer -> helper: output block k at the front), wr_done (helper -> writer: its writes of k completed).
// Both roles run a non-blocking event loop (poll CB state and the counters), because either may be waiting on the
// other: a blocking wait on one side could deadlock against the other side's help request.
// No raw LLK: dataflow_api only (noc_async_*, CB sync, semaphores).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"
#include "mhc_skip_noc.hpp"

namespace {
FORCE_INLINE uint32_t sem_read(uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}
FORCE_INLINE void sem_write(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = v; }
}  // namespace

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t role = get_compile_time_arg_val(3);  // 0 primary reader, 1 primary writer
    constexpr uint32_t x_help_lo = get_compile_time_arg_val(4);
    constexpr bool f_help = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t w_help_lo = get_compile_time_arg_val(6);
    constexpr bool help_alt = get_compile_time_arg_val(7) != 0;  // this RISC's HELP traffic on the other NoC
    constexpr uint32_t depth_in = get_compile_time_arg_val(8);
    constexpr uint32_t depth_out = get_compile_time_arg_val(9);
    constexpr uint32_t cb_f = get_compile_time_arg_val(10);
    constexpr uint32_t cb_x = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(13);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(14);
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(15);
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(16);
    constexpr uint32_t f_page = get_compile_time_arg_val(17);
    constexpr uint32_t x_page = get_compile_time_arg_val(18);
    constexpr uint32_t coef_page = get_compile_time_arg_val(19);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(20);
    constexpr uint32_t sem_rd_go = get_compile_time_arg_val(21);
    constexpr uint32_t sem_rd_done = get_compile_time_arg_val(22);
    constexpr uint32_t sem_wr_go = get_compile_time_arg_val(23);
    constexpr uint32_t sem_wr_done = get_compile_time_arg_val(24);
    constexpr uint32_t help_from = get_compile_time_arg_val(25);          // first block index the reads are helped from
    constexpr bool coef_incremental = get_compile_time_arg_val(26) != 0;  // expand one term per event-loop pass
    constexpr auto f_args = TensorAccessorArgs<27>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    constexpr uint32_t row_tiles = n * col_tiles_per_row;
    constexpr bool reads_helped = x_help_lo < n || f_help;
    constexpr bool writes_helped = w_help_lo < n;
    constexpr uint32_t x_block_bytes = n * B * x_page;
    constexpr uint32_t f_block_bytes = B * f_page;

    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t post_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_unit = get_arg_val<uint32_t>(5);
    const uint32_t num_units = get_arg_val<uint32_t>(6);

    const auto f_acc = TensorAccessor(f_args, f_addr, f_page);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page);
    const auto o_acc = TensorAccessor(o_args, o_addr, x_page);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page);

    const uint32_t rd_go = get_semaphore(sem_rd_go);
    const uint32_t rd_done = get_semaphore(sem_rd_done);
    const uint32_t wr_go = get_semaphore(sem_wr_go);
    const uint32_t wr_done = get_semaphore(sem_wr_done);
    const uint8_t my_noc = noc_index;
    const uint8_t help_noc = help_alt ? 1 - noc_index : noc_index;

    struct Blk {
        uint32_t row, col_start, valid;
        bool first_of_segment;
    };
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
            return Blk{seg.row, seg.col0 + b * B, mhc_post::block_valid_col_tiles(seg.col_tiles, B, b), b == 0};
        }
    };

    // Reads of X streams [i_lo, i_hi) (+ F if with_f) of block p into windows at x_win / f_win, on `noc`.
    auto issue_reads =
        [&](const Blk& p, bool with_f, uint32_t f_win, uint32_t i_lo, uint32_t i_hi, uint32_t x_win, uint8_t noc) {
            if (with_f) {
                const uint32_t f_page0 = p.row * col_tiles_per_row + p.col_start;
                for (uint32_t c = 0; c < p.valid; ++c) {
                    data_read(f_acc.get_noc_addr(f_page0 + c, 0, noc), f_win + c * f_page, f_page, noc);
                }
            }
            for (uint32_t i = i_lo; i < i_hi; ++i) {
                const uint32_t x_page0 = p.row * row_tiles + i * col_tiles_per_row + p.col_start;
                const uint32_t slot0 = x_win + i * B * x_page;
                for (uint32_t c = 0; c < p.valid; ++c) {
                    data_read(x_acc.get_noc_addr(x_page0 + c, 0, noc), slot0 + c * x_page, x_page, noc);
                }
            }
        };
    auto issue_writes = [&](const Blk& p, uint32_t j_lo, uint32_t j_hi, uint32_t out_win, uint8_t noc) {
        for (uint32_t j = j_lo; j < j_hi; ++j) {
            const uint32_t page0 = p.row * row_tiles + j * col_tiles_per_row + p.col_start;
            const uint32_t slot0 = out_win + j * B * x_page;
            for (uint32_t c = 0; c < p.valid; ++c) {
                data_write(slot0 + c * x_page, o_acc.get_noc_addr(page0 + c, 0, noc), x_page, noc);
            }
        }
    };

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);

    if constexpr (role == 0) {
        // ================= primary reader (+ write helper) =================
        BlockIter rd(walker);
        BlockIter wh(walker);
        uint32_t r_next = 0;  // blocks pushed
        bool r_waiting = false;
        uint32_t wh_done = 0;
        const uint32_t out_base = get_read_ptr(cb_out);
        while (!rd.done() || r_waiting || (writes_helped && !wh.done())) {
            if (!r_waiting && !rd.done() && cb_pages_reservable_at_back(cb_f, B) &&
                cb_pages_reservable_at_back(cb_x, n * B)) {
                cb_reserve_back(cb_f, B);
                cb_reserve_back(cb_x, n * B);
                const Blk p = rd.next();
                const bool helped = reads_helped && r_next >= help_from;
                if (helped) {
                    sem_write(rd_go, r_next + 1);
                }
                {
                    MaybeDeviceZoneScope("dm_read_issue");
                    issue_reads(
                        p,
                        !(helped && f_help),
                        get_write_ptr(cb_f),
                        0,
                        helped ? x_help_lo : n,
                        get_write_ptr(cb_x),
                        my_noc);
                }
                {
                    MaybeDeviceZoneScope("dm_read_barrier");
                    noc_async_read_barrier(my_noc);
                }
                r_waiting = true;
            }
            if (r_waiting && (!reads_helped || r_next < help_from || sem_read(rd_done) >= r_next + 1)) {
                cb_push_back(cb_f, B);
                cb_push_back(cb_x, n * B);
                ++r_next;
                r_waiting = false;
            }
            if constexpr (writes_helped) {
                if (!wh.done() && sem_read(wr_go) >= wh_done + 1) {
                    MaybeDeviceZoneScope("dm_help_write");
                    const Blk p = wh.next();
                    issue_writes(p, w_help_lo, n, out_base + (wh_done % depth_out) * x_block_bytes, help_noc);
                    noc_async_write_barrier(help_noc);
                    ++wh_done;
                    sem_write(wr_done, wh_done);
                }
            }
        }
    } else {
        // ================= primary writer (+ read helper, coefficient expander) =================
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
        mhc_post::SegmentWalker ahead = walker;
        // Coefficient load as a resumable job: raw read + barrier at start, then one half-tile expansion per
        // event-loop pass (coef_incremental) so a pending help request waits at most one term, not a whole set.
        constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
        constexpr uint32_t num_raw_tiles = post_tiles_per_row + comb_tiles_per_row;
        bool job_active = false;
        uint32_t job_j = 0, job_t = 0, job_base = 0;
        uint32_t loads_pending = 0;  // look-ahead loads requested, not yet started
        auto job_start = [&]() {
            MaybeDeviceZoneScope("dm_coef_start");
            WAYPOINT("CFRR");
            while (!cb_pages_reservable_at_back(cb_coef_raw, num_raw_tiles)) {
            }
            WAYPOINT("CFRD");
            coefs.issue_raw_reads(ahead.next().row);
            noc_async_read_barrier(my_noc);
            cb_push_back(cb_coef_raw, num_raw_tiles);
            cb_wait_front(cb_coef_raw, num_raw_tiles);
            WAYPOINT("CFBW");
            cb_reserve_back(cb_coef_bcast, num_coef_tiles);
            WAYPOINT("CFBD");
            job_base = get_write_ptr(cb_coef_bcast);
            job_j = job_t = 0;
            job_active = true;
        };
        auto job_step = [&]() {
#ifndef MHC_SKIP_EXPAND
            const uint32_t t = job_t;
            const uint32_t raw_addr = t == 0 ? coefs.raw_post_addr : coefs.raw_comb_addr;
            const uint32_t raw_col = t == 0 ? job_j : (t - 1) * n + job_j;
            const uint32_t tile = job_j * coef_tiles_per_stream + mhc_post::coef_tile_in_stream(t);
            mhc_post::expand_half(raw_addr, raw_col, job_base + tile * coef_page, mhc_post::coef_half(t));
#endif
            if (++job_t > n) {
                cb_push_back(cb_coef_bcast, coef_tiles_per_stream);
                job_t = 0;
                if (++job_j == n) {
                    cb_pop_front(cb_coef_raw, num_raw_tiles);
                    job_active = false;
                }
            }
        };
        auto coef_progress = [&](bool finish) {
            if (!job_active && loads_pending > 0) {
                --loads_pending;
                job_start();
            }
            if (job_active) {
                if (finish) {
                    MaybeDeviceZoneScope("dm_coef_expand");
                    while (job_active) {
                        job_step();
                    }
                } else {
                    job_step();
                }
            }
        };
        loads_pending = 1;  // set 0 (walker is non-empty: the host gives every core >= 1 unit)
        coef_progress(!coef_incremental);
        BlockIter wr(walker);
        BlockIter rh(walker);
        for (uint32_t k = 0; k < help_from && !rh.done(); ++k) {
            rh.next();  // the reader reads these blocks alone
        }
        uint32_t w_next = 0;
        uint32_t w_state = 0;  // 0 idle, 1 own writes done / waiting for the helper
        Blk w_cur{0, 0, 0, false};
        uint32_t rh_done = 0;
        const uint32_t x_base = get_write_ptr(cb_x);
        const uint32_t f_base = get_write_ptr(cb_f);
        while (!wr.done() || w_state != 0 || (reads_helped && !rh.done()) || job_active || loads_pending) {
            if constexpr (reads_helped) {
                if (!rh.done() && sem_read(rd_go) >= rh_done + help_from + 1) {
                    {
                        MaybeDeviceZoneScope("dm_help_read");
                        const Blk p = rh.next();
                        const uint32_t k = (rh_done + help_from) % depth_in;
                        issue_reads(
                            p, f_help, f_base + k * f_block_bytes, x_help_lo, n, x_base + k * x_block_bytes, help_noc);
                        noc_async_read_barrier(help_noc);
                    }
                    ++rh_done;
                    sem_write(rd_done, rh_done + help_from);
                }
            }
            if (w_state == 0 && !wr.done() && cb_pages_available_at_front(cb_out, n * B)) {
                cb_wait_front(cb_out, n * B);
                w_cur = wr.next();
                if constexpr (writes_helped) {
                    sem_write(wr_go, w_next + 1);
                }
                {
                    MaybeDeviceZoneScope("dm_write_issue");
                    issue_writes(w_cur, 0, w_help_lo, get_read_ptr(cb_out), my_noc);
                }
                {
                    MaybeDeviceZoneScope("dm_write_barrier");
                    noc_async_write_barrier(my_noc);
                }
                w_state = 1;
            }
            if (w_state == 1 && (!writes_helped || sem_read(wr_done) >= w_next + 1)) {
                cb_pop_front(cb_out, n * B);
                ++w_next;
                w_state = 0;
                if (w_cur.first_of_segment && !ahead.done()) {
                    ++loads_pending;  // next segment's set, ahead of compute
                    coef_progress(!coef_incremental);
                }
            }
            if (job_active || loads_pending) {
                coef_progress(!coef_incremental);
            }
        }
    }
}

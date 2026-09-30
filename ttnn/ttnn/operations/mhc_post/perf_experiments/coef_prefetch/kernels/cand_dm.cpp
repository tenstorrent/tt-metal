// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// coef_prefetch candidate of mhc_post_dm.cpp (the op's DM kernel; see its header for roles / read help).
// Compile-time switches (defines, default = the op's behaviour):
//   COEF_EAGER (0/1) — start the next segment's coefficient set as soon as the previous job is finished and
//        cb_coef_bcast has room for a whole set (non-blocking cb_pages_reservable_at_back), instead of right
//        after writing the current segment's first output block.
//   COEF_INC (0/1)   — expand one term per event-loop pass also without read help (op: whole set at once).
//   COEF_SHARE (0/1) — NCRISC expands the odd-index terms in its reserve / barrier / wait-help shadow
//        (cand_coef_expand.hpp ReaderShare; BRISC stays the only CB-API producer of the coefficient CBs).
//   COEF_FAST (0/1)  — software-pipelined expand_half.
//   COEF_WHOLE (0/1) — expand the whole set in one go also WITH read help (op: one term per pass with help).
//   COEF_HELP_NB (0/1) — the writer's read-help F reads are polled for completion instead of a blocking barrier,
//        so output writes and coefficient steps proceed while the helped block is in flight.
// No raw LLK: dataflow_api only (noc_async_*, CB sync, L1 semaphores).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "cand_coef_expand.hpp"

#ifndef COEF_EAGER
#define COEF_EAGER 0
#endif
#ifndef COEF_INC
#define COEF_INC 0
#endif
#ifndef COEF_SHARE
#define COEF_SHARE 0
#endif
#ifndef COEF_FAST
#define COEF_FAST 0
#endif
#ifndef COEF_WHOLE
#define COEF_WHOLE 0
#endif
#ifndef COEF_HELP_NB
#define COEF_HELP_NB 0
#endif
#ifndef COEF_DEPTH_SETS
#define COEF_DEPTH_SETS 2
#endif

namespace {
FORCE_INLINE uint32_t sem_read(uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}
FORCE_INLINE void sem_write(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = v; }

struct Blk {
    uint32_t row, col_start, valid;
    bool first_of_segment;
};

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
        return Blk{seg.row, seg.col0 + b * B, mhc_post::block_valid_col_tiles(seg.col_tiles, B, b), b == 0};
    }
};

FORCE_INLINE bool reads_flushed(uint8_t noc) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        invalidate_l1_cache();
        return ncrisc_dynamic_noc_reads_flushed(noc);
    } else {
        return ncrisc_noc_reads_flushed(noc);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t role = get_compile_time_arg_val(3);
    constexpr bool read_help = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t help_from = get_compile_time_arg_val(5);
    constexpr uint32_t depth_in = get_compile_time_arg_val(6);
    constexpr uint32_t cb_f = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(10);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(11);
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(12);
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(13);
    constexpr uint32_t f_page = get_compile_time_arg_val(14);
    constexpr uint32_t x_page = get_compile_time_arg_val(15);
    constexpr uint32_t coef_page = get_compile_time_arg_val(16);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(17);
    constexpr uint32_t sem_rd_go = get_compile_time_arg_val(18);
    constexpr uint32_t sem_rd_done = get_compile_time_arg_val(19);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(20);
    constexpr uint32_t sem_exp_go = get_compile_time_arg_val(21);
    constexpr uint32_t sem_exp_done = get_compile_time_arg_val(22);
    constexpr auto f_args = TensorAccessorArgs<23>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * mhc_post::FACE_HW, "mhc_post: expansion assumes 32x32 tiles of 16x16 faces");

    constexpr bool EAGER = COEF_EAGER != 0;
    constexpr bool INC = (read_help && COEF_WHOLE == 0) || COEF_INC != 0;  // one term per event-loop pass
    constexpr bool SHARE = COEF_SHARE != 0;
    constexpr bool FAST = COEF_FAST != 0;
    constexpr bool HELP_NB = COEF_HELP_NB != 0;

    constexpr uint32_t row_tiles = n * col_tiles_per_row;
    constexpr uint32_t f_block_bytes = B * f_page;

    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t post_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_unit = get_arg_val<uint32_t>(5);
    const uint32_t num_units = get_arg_val<uint32_t>(6);

    const uint32_t rd_go = get_semaphore(sem_rd_go);
    const uint32_t rd_done = get_semaphore(sem_rd_done);
    const uint32_t exp_go = get_semaphore(sem_exp_go);
    const uint32_t exp_done = get_semaphore(sem_exp_done);

    const auto f_acc = TensorAccessor(f_args, f_addr, f_page);
    auto issue_f_reads = [&](const Blk& p, uint32_t f_win, uint8_t noc) {
        const uint32_t f_page0 = p.row * col_tiles_per_row + p.col_start;
        for (uint32_t c = 0; c < p.valid; ++c) {
            noc_async_read(f_acc.get_noc_addr(f_page0 + c, 0, noc), f_win + c * f_page, f_page, noc);
        }
    };

    const mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);

    if constexpr (role == 0) {
        // ================= reader (+ share of the coefficient expansion) =================
        const auto x_acc = TensorAccessor(x_args, x_addr, x_page);
        uint32_t total_sets = 0;
        for (mhc_post::SegmentWalker w = walker; !w.done(); w.next()) {
            ++total_sets;
        }
        mhc_post::ReaderShare<n, coef_tiles_per_stream, coef_page, post_tiles_per_row, COEF_DEPTH_SETS, FAST> share(
            exp_go, exp_done, get_write_ptr(cb_coef_raw), get_write_ptr(cb_coef_bcast), SHARE ? total_sets : 0);
        auto share_poll = [&]() {
            if constexpr (SHARE) {
                if (share.ready()) {
                    MaybeDeviceZoneScope("reader_coef_term");
                    share.work();
                }
            }
        };
        BlockIter<B> rd(walker);
        for (uint32_t k = 0; !rd.done(); ++k) {
            {
                MaybeDeviceZoneScope("reader_reserve");
                if constexpr (SHARE) {
                    while (!(cb_pages_reservable_at_back(cb_f, B) && cb_pages_reservable_at_back(cb_x, n * B))) {
                        share_poll();
                    }
                }
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
                if constexpr (SHARE) {
                    while (!reads_flushed(noc_index)) {
                        share_poll();
                    }
                }
                noc_async_read_barrier();
            }
            if (helped) {
                MaybeDeviceZoneScope("reader_wait_help");
                while (sem_read(rd_done) < k + 1) {
                    share_poll();
                }
            }
            cb_push_back(cb_f, B);
            cb_push_back(cb_x, n * B);
        }
        if constexpr (SHARE) {
            while (!share.finished()) {
                share_poll();
            }
        }
    } else {
        // ================= writer (+ coefficient expander, + read help) =================
        const auto o_acc = TensorAccessor(o_args, o_addr, x_page);
        const auto post_acc = TensorAccessor(post_args, post_addr, coef_page);
        const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page);
        constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
        mhc_post::CoefExpander<
            n,
            coef_tiles_per_stream,
            post_tiles_per_row,
            comb_tiles_per_row,
            coef_page,
            cb_coef_raw,
            cb_coef_bcast,
            SHARE,
            FAST,
            decltype(post_acc)>
            coefs(post_acc, comb_acc, exp_go, exp_done);
        const uint8_t help_noc = 1 - noc_index;

        mhc_post::SegmentWalker ahead = walker;
        uint32_t loads_pending = 1;  // op schedule (EAGER = 0)
        auto want_start = [&]() -> bool {
            if (coefs.active() || ahead.done()) {
                return false;
            }
            if constexpr (EAGER) {
                return cb_pages_reservable_at_back(cb_coef_bcast, num_coef_tiles);
            } else {
                return loads_pending > 0;
            }
        };
        auto coef_progress = [&]() {
            if (want_start()) {
                if constexpr (!EAGER) {
                    --loads_pending;
                }
                MaybeDeviceZoneScope("writer_coef_start");
                coefs.start(ahead.next().row);
            }
            if constexpr (INC) {
                if (coefs.active()) {
                    coefs.step();
                }
            } else if (coefs.active() && !coefs.own_done()) {
                MaybeDeviceZoneScope("writer_coef_expand");
                while (!coefs.own_done()) {
                    coefs.step();
                }
            } else if (coefs.active()) {
                coefs.try_push();  // SHARE: waiting on the reader's share
            }
        };
        coef_progress();

        BlockIter<B> wr(walker);
        BlockIter<B> rh(walker);
        for (uint32_t k = 0; k < help_from && !rh.done(); ++k) {
            rh.next();
        }
        uint32_t helped_k = help_from;
        const uint32_t f_base = get_write_ptr(cb_f);
        auto coef_pending = [&]() -> bool {
            if constexpr (EAGER) {
                return coefs.active() || !ahead.done();
            } else {
                return coefs.active() || loads_pending > 0;
            }
        };
        bool help_inflight = false;
        while (!wr.done() || (read_help && (!rh.done() || help_inflight)) || coef_pending()) {
            if constexpr (read_help) {
                if constexpr (HELP_NB) {
                    // Non-blocking help: issue, then poll the NoC0 read flush in later passes (in dynamic-NoC
                    // mode the flush covers both RISCs' reads on that NoC, so a blocking barrier here also waits
                    // for the reader's X reads — up to ~25 us of a stalled event loop, measured).
                    if (help_inflight) {
                        if (reads_flushed(help_noc)) {
                            help_inflight = false;
                            ++helped_k;
                            sem_write(rd_done, helped_k);
                        }
                    } else if (!rh.done() && sem_read(rd_go) >= helped_k + 1) {
                        MaybeDeviceZoneScope("writer_help_read");
                        issue_f_reads(rh.next(), f_base + (helped_k % depth_in) * f_block_bytes, help_noc);
                        help_inflight = true;
                    }
                } else if (!rh.done() && sem_read(rd_go) >= helped_k + 1) {
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
                if constexpr (!EAGER) {
                    if (p.first_of_segment && !ahead.done()) {
                        ++loads_pending;
                    }
                }
            }
            if (coef_pending()) {
                coef_progress();
            }
        }
    }
}

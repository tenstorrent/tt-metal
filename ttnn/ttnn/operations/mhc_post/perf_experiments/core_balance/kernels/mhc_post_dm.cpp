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
//
// core_balance (perf round 2 experiment): per-core finish times are equalized by run-time claimed TAIL QUEUES.
// Each core first walks its static prefix, then the reader claims chunks one at a time with a NoC atomic
// fetch-and-increment on a queue counter (a semaphore word in the queue's home core L1):
//   pool_mode 1: one global queue (the last units of the tensor) homed on the counter core;
//   pool_mode 2: one queue per core = the tail of its own uniform-split range, homed on that core; the owner
//                drains its own queue first (same token row -> same coefficient set), then steals from the other
//                queues in a fixed stride order.
// noc_fast_atomic_increment with a programmed return address is used because Blackhole NOC_AT_INS_INCR_GET
// returns the old value and dataflow_api's noc_semaphore_inc discards it. The claimed tag goes to compute
// (cb_meta_c, read_tile_value) and to the writer (cb_meta_w); 0 = no more work. The reader claims a chunk only
// after reserving its input window, i.e. when its input queue has drained to the block compute is working on,
// so a core claims in proportion to how fast it actually progresses.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"

namespace {
FORCE_INLINE uint32_t sem_read(uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}
FORCE_INLINE void sem_write(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = v; }

using Blk = mhc_post::RampBlk;
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
    constexpr uint32_t pool_mode = get_compile_time_arg_val(21);  // 0 static, 1 global queue, 2 per-core queues
    constexpr uint32_t chunk_cols = get_compile_time_arg_val(22);
    constexpr uint32_t cb_meta_c = get_compile_time_arg_val(23);
    constexpr uint32_t cb_meta_w = get_compile_time_arg_val(24);
    constexpr uint32_t sem_claim = get_compile_time_arg_val(25);
    constexpr uint32_t sem_ret = get_compile_time_arg_val(26);
    constexpr bool help_dynamic = get_compile_time_arg_val(27) != 0;  // the helper also reads F of claimed chunks
    constexpr uint32_t ramp0 = get_compile_time_arg_val(28);          // first block width (ramped walk; B = op's walk)
    constexpr uint32_t cb_scan = get_compile_time_arg_val(29);        // reader-private scratch: one 16 B slot per queue
    constexpr auto f_args = TensorAccessorArgs<30>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * mhc_post::FACE_HW, "mhc_post: expansion assumes 32x32 tiles of 16x16 faces");
    static_assert(chunk_cols <= B, "core_balance: a claimed chunk is one block");
    constexpr bool dynamic_pool = pool_mode != 0;
    constexpr bool help_dyn = read_help && help_dynamic;

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
    const mhc_post::PoolParams pool{
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10),
        get_arg_val<uint32_t>(11),
        col_tiles_per_row,
        chunk_cols};

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
        auto issue_x_reads = [&](const Blk& p) {
            const uint32_t x_base = get_write_ptr(cb_x);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t x_page0 = p.row * row_tiles + i * col_tiles_per_row + p.col_start;
                const uint32_t slot0 = x_base + i * B * x_page;
                for (uint32_t c = 0; c < p.valid; ++c) {
                    noc_async_read(x_acc.get_noc_addr(x_page0 + c), slot0 + c * x_page, x_page);
                }
            }
        };
        uint32_t k = 0;  // block counter over the whole sequence (static, then claimed)
        // One block into the (already reserved) windows: F by the helper when helped, one barrier, pushes.
        auto read_block = [&](const Blk& p, bool help_ok) {
            const bool helped = help_ok && read_help && k >= help_from;
            if (helped) {
                sem_write(rd_go, k + 1);
            }
            {
                MaybeDeviceZoneScope("reader_issue");
                if (!helped) {
                    issue_f_reads(p, get_write_ptr(cb_f), noc_index);
                }
                issue_x_reads(p);
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
            ++k;
        };
        auto reserve = [&]() {
            MaybeDeviceZoneScope("reader_reserve");  // back-pressure from compute
            cb_reserve_back(cb_f, B);
            cb_reserve_back(cb_x, n * B);
        };
        mhc_post::RampBlocks<B, ramp0> rd(walker);
        while (!rd.done()) {
            reserve();
            read_block(rd.next(), true);
        }
        if constexpr (dynamic_pool) {
            const uint32_t own = get_arg_val<uint32_t>(12);
            const uint32_t num_queues = get_arg_val<uint32_t>(13);
            const uint32_t stride = get_arg_val<uint32_t>(14) & 0xFFFF;
            const uint32_t steal_min = get_arg_val<uint32_t>(14) >> 16;  // chunks a victim must have left
            const uint32_t ret_addr = get_semaphore(sem_ret);
            const uint32_t claim_l1 = get_semaphore(sem_claim);
            volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ret_addr);
            auto publish = [&](uint32_t cb, uint32_t tag) {
                cb_reserve_back(cb, 1);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb)) = tag;
                cb_push_back(cb, 1);
            };
            // queue home word (host): chunks << 16 | noc x << 8 | noc y
            auto home = [&](uint32_t v) { return get_arg_val<uint32_t>(15 + (pool_mode == 1 ? 0 : v)); };
            auto claim = [&](uint32_t h) -> uint32_t {
                MaybeDeviceZoneScope("reader_claim");  // atomic fetch-and-increment round trip
                const uint64_t addr = get_noc_addr((h >> 8) & 0xFF, h & 0xFF, claim_l1, noc_index);
                *ret = 0xFFFFFFFFu;
                noc_fast_atomic_increment<noc_mode, true>(
                    noc_index, write_at_cmd_buf, addr, NOC_UNICAST_WRITE_VC, 1, 31, false, false, ret_addr);
                noc_async_atomic_barrier();
                uint32_t id;
                do {
                    invalidate_l1_cache();
                    id = *ret;
                } while (id == 0xFFFFFFFFu);
                return id;
            };
            auto drain = [&](uint32_t v) {  // claim chunks of queue v until it is empty
                const uint32_t h = home(v);
                const uint32_t nq = h >> 16;
                while (true) {
                    reserve();
                    const uint32_t id = claim(h);
                    if (id >= nq) {
                        return;  // queue drained: the reserved windows carry over to the next claim
                    }
                    const uint32_t tag = v * pool.maxq + id + 1;
                    publish(cb_meta_w, tag);
                    publish(cb_meta_c, tag);
                    const mhc_post::Segment sg = pool.chunk(tag);
                    read_block(Blk{sg.row, sg.col0, sg.col_tiles, true}, help_dyn);
                }
            };
            if constexpr (pool_mode == 1) {
                drain(0);
            } else {
                drain(own);
                // Steal: a NON-mutating scan of every queue counter (num_queues 4-byte NoC reads, one barrier) picks
                // the queue with the most chunks left (ties: stride order from this core, so thieves spread); one
                // chunk is stolen per scan, and only from a queue with >= steal_min chunks left (a queue whose owner
                // is about to take its last chunk is not worth a new coefficient set). Stops when no queue qualifies.
                const uint32_t scan_base = get_write_ptr(cb_scan);
                while (true) {
                    {
                        MaybeDeviceZoneScope("reader_scan");
                        for (uint32_t v = 0; v < num_queues; ++v) {
                            if (v == own) {
                                continue;
                            }
                            const uint32_t h = home(v);
                            noc_async_read(
                                get_noc_addr((h >> 8) & 0xFF, h & 0xFF, claim_l1, noc_index), scan_base + v * 16, 4);
                        }
                        noc_async_read_barrier();
                    }
                    uint32_t best = num_queues, best_left = steal_min - 1;
                    for (uint32_t i = 1, v = own; i < num_queues; ++i) {
                        v += stride;
                        if (v >= num_queues) {
                            v -= num_queues;
                        }
                        const uint32_t taken = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scan_base + v * 16);
                        const uint32_t nq = home(v) >> 16;
                        if (taken < nq && nq - taken > best_left) {
                            best_left = nq - taken;
                            best = v;
                        }
                    }
                    if (best == num_queues) {
                        break;
                    }
                    reserve();
                    const uint32_t h = home(best);
                    const uint32_t id = claim(h);
                    if (id < (h >> 16)) {
                        const uint32_t tag = best * pool.maxq + id + 1;
                        publish(cb_meta_w, tag);
                        publish(cb_meta_c, tag);
                        const mhc_post::Segment sg = pool.chunk(tag);
                        read_block(Blk{sg.row, sg.col0, sg.col_tiles, true}, help_dyn);
                    }
                }
            }
            publish(cb_meta_w, 0);
            publish(cb_meta_c, 0);
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
        constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
        const uint8_t help_noc = 1 - noc_index;  // the reader's NoC (dynamic-NoC mode when read_help)

        // Claimed chunks, in claim order: ring of segments filled from cb_meta_w. The reader runs at most
        // DEPTH_IN + DEPTH_OUT (+1) chunks ahead of the output walker, so RING entries never alias.
        constexpr uint32_t RING = 16;
        mhc_post::Segment dq[RING];
        uint32_t dq_n = 0;                // chunks received
        bool dyn_end = !dynamic_pool;     // terminator received
        uint32_t wd = 0;                  // next chunk to write out
        uint32_t wc = 0;                  // next chunk whose coefficient set is considered
        uint32_t hd = 0;                  // next chunk the helper considers
        uint32_t last_row = 0xFFFFFFFFu;  // row of the most recently loaded set

        mhc_post::SegmentWalker ahead = walker;
        uint32_t loads_pending = walker.done() ? 0 : 1;
        auto run_job = [&]() {
            if constexpr (read_help) {
                if (coefs.active()) {
                    coefs.step();
                }
            } else if (coefs.active()) {
                MaybeDeviceZoneScope("writer_coef_expand");
                while (coefs.active()) {
                    coefs.step();
                }
            }
        };
        auto coef_progress = [&]() {
            if (!coefs.active() && loads_pending > 0) {
                --loads_pending;
                MaybeDeviceZoneScope("writer_coef_start");
                last_row = ahead.next().row;
                coefs.start(last_row);
            }
            if constexpr (dynamic_pool) {
                // claimed chunks: a new set as soon as the chunk is known and a set slot is free (after every
                // static load); same row as the previous segment -> reuse (compute mirrors it).
                while (!coefs.active() && loads_pending == 0 && ahead.done() && wc < dq_n) {
                    const uint32_t row = dq[wc % RING].row;
                    if (row == last_row) {
                        ++wc;
                        continue;
                    }
                    if (!cb_pages_reservable_at_back(cb_coef_bcast, num_coef_tiles)) {
                        break;
                    }
                    MaybeDeviceZoneScope("writer_coef_start");
                    last_row = row;
                    ++wc;
                    coefs.start(row);
                }
            }
            run_job();
        };
        coef_progress();

        mhc_post::RampBlocks<B, ramp0> wr(walker);
        mhc_post::RampBlocks<B, ramp0> rh(walker);  // blocks the helper reads F for
        uint32_t skip_left = help_from;             // the reader reads the first help_from blocks alone
        while (skip_left > 0 && !rh.done()) {
            rh.next();
            --skip_left;
        }
        uint32_t helped_k = help_from;
        const uint32_t f_base = get_write_ptr(cb_f);
        auto write_block = [&](const Blk& p) {
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
        };
        auto help_pending = [&]() -> bool {
            if constexpr (!read_help) {
                return false;
            } else if constexpr (help_dyn) {
                return !rh.done() || !dyn_end || hd < dq_n;
            } else {
                return !rh.done();
            }
        };
        while (!wr.done() || !dyn_end || wd < dq_n || help_pending() || coefs.active() || loads_pending) {
            if constexpr (dynamic_pool) {
                while (!dyn_end && cb_pages_available_at_front(cb_meta_w, 1)) {
                    cb_wait_front(cb_meta_w, 1);
                    invalidate_l1_cache();
                    const uint32_t tag = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta_w));
                    cb_pop_front(cb_meta_w, 1);
                    if (tag == 0) {
                        dyn_end = true;
                    } else {
                        dq[dq_n % RING] = pool.chunk(tag);
                        ++dq_n;
                    }
                }
            }
            if constexpr (read_help) {
                if constexpr (help_dyn) {
                    while (skip_left > 0 && rh.done() && hd < dq_n) {  // claimed blocks below help_from
                        ++hd;
                        --skip_left;
                    }
                }
                if (sem_read(rd_go) >= helped_k + 1) {
                    bool have = false;
                    Blk p{0, 0, 0, false};
                    if (!rh.done()) {
                        p = rh.next();
                        have = true;
                    } else if constexpr (help_dyn) {
                        if (hd < dq_n) {
                            const mhc_post::Segment& sg = dq[hd % RING];
                            p = Blk{sg.row, sg.col0, sg.col_tiles, true};
                            ++hd;
                            have = true;
                        }
                    }
                    if (have) {
                        {
                            MaybeDeviceZoneScope("writer_help_read");
                            issue_f_reads(p, f_base + (helped_k % depth_in) * f_block_bytes, help_noc);
                            noc_async_read_barrier(help_noc);
                        }
                        ++helped_k;
                        sem_write(rd_done, helped_k);
                    }
                }
            }
            if (!wr.done()) {
                if (cb_pages_available_at_front(cb_out, n * B)) {
                    cb_wait_front(cb_out, n * B);
                    const Blk p = wr.next();
                    write_block(p);
                    if (p.first_of_segment && !ahead.done()) {
                        ++loads_pending;
                    }
                }
            } else if (wd < dq_n && cb_pages_available_at_front(cb_out, n * B)) {
                cb_wait_front(cb_out, n * B);
                const mhc_post::Segment& sg = dq[wd % RING];
                write_block(Blk{sg.row, sg.col0, sg.col_tiles, true});
                ++wd;
            }
            coef_progress();
        }
    }
}

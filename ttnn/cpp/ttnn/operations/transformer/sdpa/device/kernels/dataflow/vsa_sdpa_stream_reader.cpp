// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v3) reader. Cores are grouped per head; the group's first core is a
// fetch-only LEADER, the rest are WORKERS holding resident query rows.
//
// LEADER (is_leader=1): streams every real (count > 0) KV block of the head from DRAM exactly once
// per pass, in ascending id order, into its own stream-slot CBs (V here, K via the leader writer's
// kreq/kack service). After a block lands it appends a {block_id, slot} log entry into every
// worker's log ring and bumps their arrivals semaphore. Slot reuse is gated on a per-worker
// progress MAILBOX (each worker posts its count of consumed arrivals into its own word of the
// leader's ackbox; the leader gates on the MINIMUM). A pooled ack count cannot bound the slowest
// worker -- a fast worker's acks would let the leader overwrite slots and log entries a lagging
// worker has not pulled yet. A sentinel log entry (block_id = 0xFFFFFFFF) ends each pass.
//
// WORKER (is_leader=0): builds per-resident-row membership bitmaps for the pass, then consumes log
// entries in order. For a block some resident row lists, it pulls V from the leader's L1 slot
// (the K pull rides the local writer via kreq/kack), acks the leader, builds the ragged mask tile
// if needed, and emits ONE multi-row visit message to compute. Blocks no row lists are acked
// immediately. On the sentinel it emits per-row FLUSH messages carrying each row's final state
// parity. CB layouts are identical on every core, so a leader slot's L1 address equals the
// worker's own CB base plus the slot offset.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"
#include "dataflow_common.hpp"

constexpr uint32_t sentinel = 0xFFFFFFFFu;

constexpr uint32_t MSG_VISIT = 0;   // {type | n_blocks<<16, rowinfo, (slot | count<<8 | vmask<<15) x n}
constexpr uint32_t MSG_FLUSH = 1;   // {type, row_slot, parity}
constexpr uint32_t MSG_WINDOW = 2;  // {type, n_slots}: compute returns n_slots stream credits
// rowinfo word: row_slot | parity << 8 | is_first << 9
constexpr uint32_t ROW_PARITY = 1u << 8;
constexpr uint32_t ROW_IS_FIRST = 1u << 9;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);            // index row width (entries)
    constexpr uint32_t n_kv_blocks = get_compile_time_arg_val(1);  // T / block_size
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(2);    // S / 64 per head
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t R_MAX = get_compile_time_arg_val(4);
    constexpr uint32_t stream_depth = get_compile_time_arg_val(5);
    constexpr uint32_t log_depth = get_compile_time_arg_val(6);
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(7);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(8);
    constexpr uint32_t v_head_stride = get_compile_time_arg_val(9);  // tiles per head in v
    constexpr uint32_t idx_row_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t counts_row_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(13);

    constexpr uint32_t cb_k_stream = get_compile_time_arg_val(14);
    constexpr uint32_t cb_v_stream = get_compile_time_arg_val(15);
    constexpr uint32_t cb_idxrow = get_compile_time_arg_val(16);
    constexpr uint32_t cb_counts = get_compile_time_arg_val(17);
    constexpr uint32_t cb_bitmap = get_compile_time_arg_val(18);
    constexpr uint32_t cb_log = get_compile_time_arg_val(19);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(20);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(21);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(22);
    constexpr uint32_t cb_free = get_compile_time_arg_val(23);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(24);
    constexpr uint32_t cb_ackbox = get_compile_time_arg_val(25);  // per-worker progress words on the leader
    constexpr uint32_t sem_arrivals = get_compile_time_arg_val(26);

    constexpr auto v_args = TensorAccessorArgs<27, 0>();
    constexpr auto idx_args =
        TensorAccessorArgs<v_args.next_compile_time_args_offset(), v_args.next_common_runtime_args_offset()>();
    constexpr auto counts_args =
        TensorAccessorArgs<idx_args.next_compile_time_args_offset(), idx_args.next_common_runtime_args_offset()>();

    uint32_t argi = 0;
    const uint32_t v_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t idx_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t head = get_arg_val<uint32_t>(argi++);
    const uint32_t is_leader = get_arg_val<uint32_t>(argi++);
    const uint32_t n_passes = get_arg_val<uint32_t>(argi++);
    const uint32_t leader_x = get_arg_val<uint32_t>(argi++);
    const uint32_t leader_y = get_arg_val<uint32_t>(argi++);
    const uint32_t n_workers = get_arg_val<uint32_t>(argi++);
    const uint32_t worker_index = get_arg_val<uint32_t>(argi++);  // this worker's slot in the group
    const uint32_t row_start = get_arg_val<uint32_t>(argi++);     // worker: first resident row
    const uint32_t row_stride = get_arg_val<uint32_t>(argi++);    // worker: row stride
    const uint32_t row_count = get_arg_val<uint32_t>(argi++);     // worker: total rows

    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint32_t bitmap_words = (n_kv_blocks + 31) / 32;
    constexpr uint32_t log_entry_words = 4;  // {block_id, slot, pad, pad} (16 B, DRAM/NoC aligned)

    Noc noc;
    experimental::CB v_cb(cb_v_stream), idx_cb(cb_idxrow), counts_cb(cb_counts);
    experimental::CB ctrl_cb(cb_ctrl), kreq_cb(cb_kreq), kack_cb(cb_kack), free_cb(cb_free);
    Semaphore<> arrivals_sem(sem_arrivals);
    experimental::CB ackbox_cb(cb_ackbox);
    ackbox_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ackbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ackbox_cb.get_write_ptr());
    const uint32_t ackbox_l1 = ackbox_cb.get_write_ptr();
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);

    // counts row: resident for the whole kernel on both roles.
    counts_cb.reserve_back(1);
    noc.async_read(counts, counts_cb, counts_row_bytes, {.page_id = 0}, {.offset_bytes = 0});
    volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr());
    noc.async_read_barrier();

    experimental::CB log_cb(cb_log);
    log_cb.reserve_back(1);
    const uint32_t log_l1 = log_cb.get_write_ptr();

    if (is_leader) {
        // ---------------- LEADER ----------------
        // Multicast strips (height-1 rectangles covering the group's workers) follow in the
        // runtime args; log entries are published with one multicast write per strip.
        const uint32_t n_strips = get_arg_val<uint32_t>(argi++);
        uint32_t strip_sx[4], strip_sy[4], strip_ex[4], strip_ey[4], strip_n[4];
        for (uint32_t st = 0; st < n_strips; ++st) {
            strip_sx[st] = get_arg_val<uint32_t>(argi++);
            strip_sy[st] = get_arg_val<uint32_t>(argi++);
            strip_ex[st] = get_arg_val<uint32_t>(argi++);
            strip_ey[st] = get_arg_val<uint32_t>(argi++);
            strip_n[st] = get_arg_val<uint32_t>(argi++);
        }
        // Translated (virtual) coordinates are direction-agnostic on Blackhole: the per-NoC
        // translation tables absorb NOC_1's reversed orientation, so the rectangle is given in
        // ascending translated order on both NoCs (a raw-coordinate swap here breaks NOC_1).
        uint64_t strip_base[4];
        for (uint32_t st = 0; st < n_strips; ++st) {
            strip_base[st] = get_noc_multicast_addr(
                strip_sx[st], strip_sy[st], strip_ex[st], strip_ey[st], 0, noc.get_noc_id());
        }
        const uint32_t v_base = head * v_head_stride;
        volatile tt_l1_ptr uint32_t* log_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_l1);
        for (uint32_t w = 0; w < n_workers; ++w) {
            ackbox[w] = 0;
        }
        const auto wait_all_workers_at = [&](uint32_t target) {
            for (uint32_t w = 0; w < n_workers; ++w) {
                while (ackbox[w] < target) {
                    invalidate_l1_cache();
                }
            }
        };

        uint32_t arrival = 0;   // published arrivals, monotonic across passes (gates slot reuse)
        uint32_t fetched = 0;   // fetches issued; runs up to kFetchLag ahead of `arrival`
        uint32_t log_n = 0;     // log entries emitted (arrivals + sentinels)
        // Fetches are pipelined kFetchLag blocks deep: every tile read of block N is tagged with
        // trid (N % 8) + 1, so publishing N costs one per-block trid barrier (long since landed
        // with the pipeline full) instead of a full DRAM round trip. A shared per-tile trid ring
        // would cap the pipeline at ONE block in flight (8 tiles), throttling fetch throughput to
        // 16KB per DRAM latency.
        constexpr uint32_t kFetchLag = 4;  // blocks in flight (kFetchLag * 8 tiles <= 8 trids x reuse)
        static_assert(kFetchLag * 2 <= stream_depth, "prefetch must not outrun slot recycling");
        // Workers zero their log rings then bump this (host-zeroed) semaphore; publishing before
        // every ring is zeroed would race the zeroing.
        arrivals_sem.wait_min(n_workers);
        const auto publish = [&](uint32_t b, uint32_t slot) {
            // One multicast write per strip. The entry's seq word (log_n + 1, nonzero, monotonic)
            // is what workers spin on; the 16-byte entry rides one flit, so a visible seq implies
            // the whole entry is visible. No barrier, no semaphore: posted writes from one source
            // to one destination on one NoC stay ordered, and the V-pull that follows the seq is
            // ordered behind the DRAM fetch by the lag-1 trid recycling.
            const uint32_t entry_off = (log_n % log_depth) * log_entry_words * 4;
            noc_async_writes_flushed(noc.get_noc_id());  // ring slot's previous mcast left the source
            volatile tt_l1_ptr uint32_t* entry =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_l1 + entry_off);
            entry[0] = b;
            entry[1] = slot;
            entry[2] = log_n + 1;
            for (uint32_t st = 0; st < n_strips; ++st) {
                noc_async_write_multicast(
                    log_l1 + entry_off, strip_base[st] + log_l1 + entry_off, log_entry_words * 4, strip_n[st],
                    false, noc.get_noc_id());
            }
            ++log_n;
        };
        uint32_t pend_b[kFetchLag], pend_slot[kFetchLag];
        const auto issue_fetch = [&](uint32_t b) {
            const uint32_t slot = fetched % stream_depth;
            if (fetched >= stream_depth) {
                // Every worker must have consumed the arrival that last used this slot.
                wait_all_workers_at(fetched - stream_depth + 1);
            }
            // K rides the leader writer (other NoC); V here, tagged with the block's trid.
#if !(defined(VSA_PROBE) && VSA_PROBE == 7)
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = b;
                rq[1] = slot;
            }
            kreq_cb.push_back(1);
            experimental::set_read_trid(noc, (fetched % 8) + 1);
            const uint32_t v_tile0 = v_base + b * v_tiles_per_block;
            for (uint32_t i = 0; i < v_tiles_per_block; ++i) {
                noc.async_read(
                    v, v_cb, v_tile_bytes, {.page_id = v_tile0 + i},
                    {.offset_bytes = (slot * v_tiles_per_block + i) * v_tile_bytes});
            }
            experimental::set_read_trid(noc, 0);
#endif
            pend_b[fetched % kFetchLag] = b;
            pend_slot[fetched % kFetchLag] = slot;
            ++fetched;
        };
        const auto publish_oldest = [&]() {
#if !(defined(VSA_PROBE) && VSA_PROBE == 7)
            experimental::async_read_barrier_with_trid(noc, (arrival % 8) + 1);  // its V landed
            kack_cb.wait_front(1);  // its K landed (writer acks per block, same pipelining)
            kack_cb.pop_front(1);
#endif
            publish(pend_b[arrival % kFetchLag], pend_slot[arrival % kFetchLag]);
            ++arrival;
        };
        for (uint32_t pass = 0; pass < n_passes; ++pass) {
            for (uint32_t b = 0; b < n_kv_blocks; ++b) {
                if (counts_ptr[b] == 0) {
                    continue;  // pad block: never listed
                }
                if (fetched - arrival == kFetchLag) {
                    publish_oldest();
                }
                issue_fetch(b);
            }
            // Sentinel kreq FIRST (the writer acks its pending blocks on seeing it), then drain
            // the prefetch queue, then the sentinel log entry (no slot, no arrival).
#if !(defined(VSA_PROBE) && VSA_PROBE == 7)
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = sentinel;
                rq[1] = 0;
            }
            kreq_cb.push_back(1);
#endif
            while (arrival < fetched) {
                publish_oldest();
            }
            publish(sentinel, 0);
        }
        // Drain: every arrival consumed before the program ends (so no worker still reads our
        // L1), and no arrivals-sem atomic may still be in flight when the next program resets it.
        wait_all_workers_at(arrival);
        noc_async_atomic_barrier(noc.get_noc_id());
        return;
    }

    // ---------------- WORKER ----------------
    idx_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());
    experimental::CB bitmap_cb(cb_bitmap);
    bitmap_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* bitmaps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bitmap_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* log_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_l1);
    // Zero the log ring's seq words (the leader's multicast publishes are spun on, and stale L1
    // could alias a valid seq), then signal READY on the leader's host-zeroed arrivals semaphore.
    for (uint32_t e = 0; e < log_depth; ++e) {
        log_ptr[e * log_entry_words + 2] = 0;
    }
    arrivals_sem.up(noc, leader_x, leader_y, 1);

    // Per-row running state parity (bit) and first-visit flag, tracked across a pass, plus the
    // window-pending block list per row: the blocks pulled within one window are freed together
    // after the window's batched visits, so a row folds every block it lists in the window into
    // ONE visit (real top-k sets cluster in runs of adjacent ids, so multi-block batches are
    // common). The slot space is split into two window halves, double-buffered: the reader pulls
    // window N+1 into the other half while compute chews window N, and waits for a half's credits
    // only when coming back around to it.
    uint32_t row_parity_bits = 0;
    uint32_t row_seen_bits = 0;
    constexpr uint32_t half_slots = stream_depth / 2;
    uint32_t pending[32][half_slots > 0 ? half_slots : 1];
    uint32_t n_pending[32];
    uint32_t half_outstanding[2] = {0, 0};  // credits still owed by compute for each half

    // Progress mailbox: this worker's consumed-arrival count, staged locally (own ackbox word) and
    // posted to the leader's ackbox word for this worker. Monotonic single-writer, so unbarriered
    // posted writes are safe; one barrier before exit flushes the tail. The local staging word sits
    // at the SAME offset as the remote target: NoC L1->L1 transfers require src and dst to share
    // their 16-byte phase.
    uint32_t consumed = 0;
    uint32_t posted = 0;
    const uint32_t my_box_local = ackbox_l1 + worker_index * 4;
    const uint64_t my_box_remote = get_noc_addr(leader_x, leader_y, ackbox_l1 + worker_index * 4, noc.get_noc_id());
    const auto post_progress_now = [&]() {
        if (posted == consumed) {
            return;
        }
        posted = consumed;
        ackbox[worker_index] = consumed;
        noc_async_write(my_box_local, my_box_remote, 4, noc.get_noc_id());
    };
    // Progress may only be posted for arrivals whose pulls have LANDED (the leader recycles its
    // slot on the strength of the post, and an in-flight read would then fetch the wrong block),
    // so pulls advance `consumed` but the post itself happens at window flush, after the drains.
    // Skip-only stretches with no pending pulls may post immediately.
    uint32_t window_pulls_pending = 0;
    const auto note_consumed = [&]() {
        ++consumed;
        if (window_pulls_pending == 0 && consumed - posted >= 4) {
            post_progress_now();
        }
    };

    // Slot credits flow per window: compute returns a window's slots after consuming its visits.
    uint32_t log_n = 0;
    // Window/credit state persists ACROSS passes: the final window of a pass leaves its slot
    // credits outstanding, and resetting the bookkeeping per pass would leak them until free_cb
    // fills and compute wedges pushing credits (observed as a multi-pass-only hang).
    uint32_t half = 0;          // which slot half the open window fills
    uint32_t window_slots = 0;  // pulled blocks in the open window
    uint32_t cur_pass_rows = 0;
        // Emit one batched visit per pending row, then a WINDOW message; compute returns the
        // window's slot credits when it has consumed every visit. The window's V pulls are still
        // in flight on the trid ring and its K pulls on the writer: drain both exactly once here,
        // then post the progress that lets the leader recycle its slots.
    sparse_sdpa_msa::TridRing vring{noc};
        const auto flush_window = [&]() {
            if (window_slots == 0) {
                post_progress_now();
                return;
            }
#if !(defined(VSA_PROBE) && VSA_PROBE == 3)
            vring.drain();  // this window's V tiles are local
            // window-end marker: the writer barriers its K pulls and acks once
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = 0xFFFFFFFFu;
                rq[1] = 0;
            }
            kreq_cb.push_back(1);
            kack_cb.wait_front(1);
            kack_cb.pop_front(1);
#endif
            window_pulls_pending = 0;
            post_progress_now();
            for (uint32_t r = 0; r < cur_pass_rows; ++r) {
                if (n_pending[r] == 0) {
                    continue;
                }
                const uint32_t rbit = 1u << r;
                uint32_t info = r;
                if (!(row_seen_bits & rbit)) {
                    info |= ROW_IS_FIRST;
                    row_seen_bits |= rbit;
                } else {
                    row_parity_bits ^= rbit;
                    if (row_parity_bits & rbit) {
                        info |= ROW_PARITY;
                    }
                }
                ctrl_cb.reserve_back(1);
                {
                    volatile tt_l1_ptr uint32_t* cp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                    cp[0] = MSG_VISIT | (n_pending[r] << 16);
                    cp[1] = info;
                    for (uint32_t j = 0; j < n_pending[r]; ++j) {
                        cp[2 + j] = pending[r][j];
                    }
                }
                ctrl_cb.push_back(1);
                n_pending[r] = 0;
            }
            ctrl_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* cp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = MSG_WINDOW;
                cp[1] = window_slots;
            }
            ctrl_cb.push_back(1);
            half_outstanding[half] = window_slots;
            window_slots = 0;
            half ^= 1;
        };


    for (uint32_t pass = 0; pass < n_passes; ++pass) {
        const uint32_t pass_row_base = pass * R_MAX;
        const uint32_t pass_rows =
            (row_count > pass_row_base) ? ((row_count - pass_row_base < R_MAX) ? row_count - pass_row_base : R_MAX)
                                        : 0;
        // Membership bitmaps for this pass's rows.
        for (uint32_t r = 0; r < pass_rows; ++r) {
            volatile tt_l1_ptr uint32_t* bm = bitmaps + r * bitmap_words;
            for (uint32_t wd = 0; wd < bitmap_words; ++wd) {
                bm[wd] = 0;
            }
            // chunk-cyclic placement: 4-row chunks dealt round-robin (row_stride = workers * 4)
            const uint32_t ri = pass_row_base + r;
            const uint32_t q_tile = row_start + (ri >> 2) * row_stride + (ri & 3);
            noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = head * n_q_tiles + q_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            for (uint32_t e = 0; e < W; ++e) {
                const uint32_t b = idx_ptr[e];
                if (b == sentinel) {
                    break;
                }
                ASSERT(b < n_kv_blocks);
                bm[b >> 5] |= (1u << (b & 31));
            }
        }
        row_parity_bits = 0;
        row_seen_bits = 0;
        for (uint32_t r = 0; r < pass_rows; ++r) {
            n_pending[r] = 0;
        }
        cur_pass_rows = pass_rows;

        while (true) {
            // Spin on the next entry's seq word. Before blocking, any progress still unposted
            // (lazy batching, or pulls of a partial window not yet drained) must be flushed FIRST
            // -- the leader's next publish can be gated on this worker's posted count (slot
            // recycling), so spinning while owing progress deadlocks the group.
            const uint32_t entry_off = (log_n % log_depth) * log_entry_words;
            invalidate_l1_cache();
            if (log_ptr[entry_off + 2] != log_n + 1) {
                if (consumed != posted) {
                    flush_window();
                }
                do {
                    invalidate_l1_cache();
                } while (log_ptr[entry_off + 2] != log_n + 1);
            }
            const uint32_t b = log_ptr[entry_off + 0];
            const uint32_t leader_slot = log_ptr[entry_off + 1];
            ++log_n;
            if (b == sentinel) {
                flush_window();
                break;
            }

            uint32_t listing[32];
            uint32_t n_listing = 0;
#if !(defined(VSA_PROBE) && VSA_PROBE == 5)
            const uint32_t wd = b >> 5;
            const uint32_t bit = 1u << (b & 31);
            for (uint32_t r = 0; r < pass_rows; ++r) {
                if (bitmaps[r * bitmap_words + wd] & bit) {
                    listing[n_listing++] = r;
                }
            }
#endif
#if defined(VSA_PROBE) && VSA_PROBE == 6
            n_listing = 0;  // probe 6: bitmap loop runs, visits suppressed
#endif
            if (n_listing == 0) {
                note_consumed();
                if (window_pulls_pending > 0 && consumed - posted >= 8) {
                    flush_window();  // a long skip-run must not starve the leader behind a partial window
                }
                continue;
            }

            // Claim a slot in the open window half, flushing first if the half is full and
            // reclaiming the half's previous credits before its first reuse.
            if (window_slots == half_slots) {
                flush_window();
            }
            if (window_slots == 0 && half_outstanding[half] > 0) {
                free_cb.wait_front(half_outstanding[half]);
                free_cb.pop_front(half_outstanding[half]);
                half_outstanding[half] = 0;
            }
            const uint32_t slot = half * half_slots + window_slots;

            // K rides the local writer (other NoC): send it the leader slot to pull from. Both
            // pulls are issue-only; the single window-end drain/kack covers them, and progress
            // posts lazily -- the leader slot is only reused stream_depth arrivals later, and the
            // pull ISSUE (source read) happens now, so a deferred post cannot race the reuse.
#if !(defined(VSA_PROBE) && VSA_PROBE == 3)
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = leader_slot;
                rq[1] = slot;
            }
            kreq_cb.push_back(1);
            {
                const uint32_t v_l1_base = v_cb.get_write_ptr();
                const uint64_t src = get_noc_addr(
                    leader_x, leader_y, v_l1_base + leader_slot * v_tiles_per_block * v_tile_bytes, noc.get_noc_id());
                vring.read_addr(src, v_l1_base + slot * v_tiles_per_block * v_tile_bytes,
                                v_tiles_per_block * v_tile_bytes);
            }
            ++window_pulls_pending;
#endif
            note_consumed();

            const uint32_t count = counts_ptr[b];
            const bool ragged = count < block_size;
            const bool needs_vmask = ragged && (count % keys_per_tile) != 0;
            if (needs_vmask) {
                // Slot-indexed RAM tile (freed with the window): visits reference the window's
                // blocks out of pull order, so a FIFO cannot serve them.
                constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
                fill_vertical_tile_bf16<mask_tile_bytes>(noc, cb_vmask, slot, count % keys_per_tile);
            }

            const uint32_t entry = slot | (count << 8) | (needs_vmask ? (1u << 15) : 0);
            for (uint32_t i = 0; i < n_listing; ++i) {
                pending[listing[i]][n_pending[listing[i]]++] = entry;
            }
            ++window_slots;
        }

        // FLUSH each pass row with its final state parity.
        for (uint32_t r = 0; r < pass_rows; ++r) {
#if !(defined(VSA_PROBE) && (VSA_PROBE == 5 || VSA_PROBE == 6))
            ASSERT(row_seen_bits & (1u << r));  // every row lists at least one block
#endif
            ctrl_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* cp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = MSG_FLUSH;
                cp[1] = r;
                cp[2] = (row_parity_bits >> r) & 1u;
            }
            ctrl_cb.push_back(1);
        }
    }

    // Flush the tail progress posts (and any stray atomics) before this program ends; in-flight
    // writes would otherwise land on the next program's memory (watcher-detected race otherwise).
    noc.async_write_barrier();
    noc_async_atomic_barrier(noc.get_noc_id());
}

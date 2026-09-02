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
#if defined(VSA_PROBE) && VSA_PROBE == 9
#include "api/debug/dprint.h"
#define VSA_TICK() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#endif
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

#ifdef VSA_IS_LEADER
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

        // ---- Leader-as-worker: this core's compute is otherwise idle, and every block's K/V is
        // already resident in its own stream slots -- so the leader carries resident rows too.
        // The machinery mirrors a worker's window engine minus the pulls and the log-ring spin:
        // arrivals are consumed inline at publish time. `own_commit` is the prefix of arrivals
        // whose slots the local compute no longer needs; the slot-reuse gate takes the MIN of the
        // workers' posted counts and own_commit.
        idx_cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());
        experimental::CB bitmap_cb(cb_bitmap);
        bitmap_cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* bitmaps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bitmap_cb.get_write_ptr());
        uint32_t own_row_parity = 0, own_row_seen = 0;
        constexpr uint32_t kOwnWin = stream_depth / 2;
        uint32_t own_pending[32][kOwnWin > 0 ? kOwnWin : 1];
        uint32_t own_np[32];
        uint32_t own_commit = 0, own_consumed = 0;
        bool own_busy[stream_depth] = {};
        uint32_t own_warr[kOwnWin > 0 ? kOwnWin : 1];
        uint32_t own_wslots = 0;
        constexpr uint32_t kOwnFifo = 16;
        uint32_t own_fifo_arr[kOwnFifo][kOwnWin > 0 ? kOwnWin : 1];
        uint32_t own_fifo_n[kOwnFifo];
        uint32_t own_head = 0, own_tail = 0;
        uint32_t own_pass_rows = 0;
        const auto own_advance = [&]() {
            while (own_commit < own_consumed && !own_busy[own_commit % stream_depth]) {
                ++own_commit;
            }
        };
        const auto own_poll_credits = [&]() {
            while (own_head != own_tail && cb_pages_available_at_front(cb_free, own_fifo_n[own_head % kOwnFifo])) {
                const uint32_t h = own_head % kOwnFifo;
                free_cb.wait_front(own_fifo_n[h]);
                free_cb.pop_front(own_fifo_n[h]);
                for (uint32_t j = 0; j < own_fifo_n[h]; ++j) {
                    own_busy[own_fifo_arr[h][j] % stream_depth] = false;
                }
                ++own_head;
                own_advance();
            }
        };
        const auto own_close_window = [&]() {
            if (own_wslots == 0) {
                return;
            }
            for (uint32_t r = 0; r < own_pass_rows; ++r) {
                if (own_np[r] == 0) {
                    continue;
                }
                const uint32_t rbit = 1u << r;
                uint32_t info = r;
                if (!(own_row_seen & rbit)) {
                    info |= ROW_IS_FIRST;
                    own_row_seen |= rbit;
                } else {
                    own_row_parity ^= rbit;
                    if (own_row_parity & rbit) {
                        info |= ROW_PARITY;
                    }
                }
                ctrl_cb.reserve_back(1);
                {
                    volatile tt_l1_ptr uint32_t* cp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                    cp[0] = MSG_VISIT | (own_np[r] << 16);
                    cp[1] = info;
                    for (uint32_t j = 0; j < own_np[r]; ++j) {
                        cp[2 + j] = own_pending[r][j];
                    }
                }
                ctrl_cb.push_back(1);
                own_np[r] = 0;
            }
            ctrl_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = MSG_WINDOW;
                cp[1] = own_wslots;
            }
            ctrl_cb.push_back(1);
            const uint32_t t = own_tail % kOwnFifo;
            own_fifo_n[t] = own_wslots;
            for (uint32_t j = 0; j < own_wslots; ++j) {
                own_fifo_arr[t][j] = own_warr[j];
            }
            ++own_tail;
            own_wslots = 0;
        };
        const auto own_consume = [&](uint32_t b, uint32_t slot) {
            uint32_t listing[32];
            uint32_t nl = 0;
            const uint32_t wd = b >> 5;
            const uint32_t bit = 1u << (b & 31);
            for (uint32_t r = 0; r < own_pass_rows; ++r) {
                if (bitmaps[r * bitmap_words + wd] & bit) {
                    listing[nl++] = r;
                }
            }
            if (nl == 0) {
                ++own_consumed;
                own_advance();
                return;
            }
            const uint32_t count = counts_ptr[b];
            const bool ragged = count < block_size;
            const bool needs_vmask = ragged && (count % keys_per_tile) != 0;
            if (needs_vmask) {
                constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
                fill_vertical_tile_bf16<mask_tile_bytes>(noc, cb_vmask, slot, count % keys_per_tile);
            }
            const uint32_t entry = slot | (count << 8) | (needs_vmask ? (1u << 15) : 0);
            for (uint32_t j = 0; j < nl; ++j) {
                own_pending[listing[j]][own_np[listing[j]]++] = entry;
            }
            own_busy[slot] = true;
            own_warr[own_wslots++] = own_consumed;
            ++own_consumed;
            if (own_wslots == kOwnWin) {
                own_close_window();
                own_poll_credits();
            }
        };
        // Gate helper: workers' posted counts AND the local compute's committed prefix. Compute
        // defers a chunk's credits with its deferred PV until the NEXT chunk -- which will never
        // come while we are gated -- so a zero-slot WINDOW message nudges it to drain.
        const auto own_wait_at = [&](uint32_t target) {
            if (own_commit >= target) {
                return;
            }
            own_close_window();  // our own open window can be what pins the commit
            own_poll_credits();
            if (own_commit >= target) {
                return;
            }
            ctrl_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = MSG_WINDOW;
                cp[1] = 0;
            }
            ctrl_cb.push_back(1);
            while (own_commit < target) {
                own_poll_credits();
            }
        };

        uint32_t arrival = 0;   // published arrivals, monotonic across passes (gates slot reuse)
        uint32_t fetched = 0;   // fetches issued; runs up to kFetchLag ahead of `arrival`
        uint32_t log_n = 0;     // log entries emitted (arrivals + sentinels)
        // Fetches are pipelined kFetchLag blocks deep: every tile read of block N is tagged with
        // trid (N % 8) + 1, so publishing N costs one per-block trid barrier (long since landed
        // with the pipeline full) instead of a full DRAM round trip. Blocks are pumped in PAIRS:
        // one worker gate check, one two-block kreq page, and one two-entry log multicast per
        // pair -- the leader's serial per-arrival cost is the measured protocol floor.
        constexpr uint32_t kFetchLag = 4;  // blocks in flight (kFetchLag * 8 tiles <= 8 trids x reuse)
        constexpr uint32_t kNoBlock = 0xFFFFFFFEu;
        static_assert(kFetchLag * 2 <= stream_depth, "prefetch must not outrun slot recycling");
        // Workers zero their log rings then bump this (host-zeroed) semaphore; publishing before
        // every ring is zeroed would race the zeroing.
        arrivals_sem.wait_min(n_workers);
        // Stage `k` entries (each 16B: {b, slot, seq, pad}) into the ring and multicast them in
        // one write per strip (two per strip on a ring wrap). A worker's per-entry seq spin is
        // safe: a NoC write's bytes land in ascending address order, so entry k's seq is visible
        // only after entry k is complete and entry k-1 fully landed.
        const auto publish_run = [&](const uint32_t* bs, const uint32_t* slots, uint32_t k) {
            noc_async_writes_flushed(noc.get_noc_id());  // ring slots' previous mcasts left the source
            const uint32_t e0 = log_n % log_depth;
            for (uint32_t j = 0; j < k; ++j) {
                volatile tt_l1_ptr uint32_t* entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    log_l1 + ((log_n + j) % log_depth) * log_entry_words * 4);
                entry[0] = bs[j];
                entry[1] = slots[j];
                entry[2] = log_n + j + 1;
            }
            const uint32_t run1 = (e0 + k <= log_depth) ? k : (log_depth - e0);
            for (uint32_t st = 0; st < n_strips; ++st) {
                noc_async_write_multicast(
                    log_l1 + e0 * log_entry_words * 4, strip_base[st] + log_l1 + e0 * log_entry_words * 4,
                    run1 * log_entry_words * 4, strip_n[st], false, noc.get_noc_id());
                if (run1 < k) {
                    noc_async_write_multicast(
                        log_l1, strip_base[st] + log_l1, (k - run1) * log_entry_words * 4, strip_n[st], false,
                        noc.get_noc_id());
                }
            }
            log_n += k;
        };
        uint32_t pend_b[kFetchLag], pend_slot[kFetchLag];
        // Publish up to a pair of pending fetched blocks (their V trid barriers + K acks first).
        const auto publish_pending = [&](uint32_t k) {
            uint32_t bs[2], slots[2];
            for (uint32_t j = 0; j < k; ++j) {
                experimental::async_read_barrier_with_trid(noc, ((arrival + j) % 8) + 1);  // V landed
                kack_cb.wait_front(1);  // K landed (writer acks per block, same pipelining)
                kack_cb.pop_front(1);
                bs[j] = pend_b[(arrival + j) % kFetchLag];
                slots[j] = pend_slot[(arrival + j) % kFetchLag];
            }
            publish_run(bs, slots, k);
            if (row_count > 0) {
                for (uint32_t j = 0; j < k; ++j) {
                    own_consume(bs[j], slots[j]);
                }
                own_poll_credits();
            }
            arrival += k;
        };
        // Fetch a pair (or single tail) of blocks: ONE gate check, ONE kreq page, per-block trids.
        const auto issue_pair = [&](const uint32_t* bs, uint32_t k) {
            if (fetched + k > stream_depth) {
                // Every consumer (workers AND the local compute) must be done with these slots.
                wait_all_workers_at(fetched + k - stream_depth);
                if (row_count > 0) {
                    own_wait_at(fetched + k - stream_depth);
                }
            }
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = bs[0];
                rq[1] = fetched % stream_depth;
                rq[2] = (k > 1) ? bs[1] : kNoBlock;
                rq[3] = (fetched + 1) % stream_depth;
            }
            kreq_cb.push_back(1);
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t slot = fetched % stream_depth;
                experimental::set_read_trid(noc, (fetched % 8) + 1);
                const uint32_t v_tile0 = v_base + bs[j] * v_tiles_per_block;
                for (uint32_t i = 0; i < v_tiles_per_block; ++i) {
                    noc.async_read(
                        v, v_cb, v_tile_bytes, {.page_id = v_tile0 + i},
                        {.offset_bytes = (slot * v_tiles_per_block + i) * v_tile_bytes});
                }
                experimental::set_read_trid(noc, 0);
                pend_b[fetched % kFetchLag] = bs[j];
                pend_slot[fetched % kFetchLag] = slot;
                ++fetched;
            }
        };
        for (uint32_t pass = 0; pass < n_passes; ++pass) {
            const uint32_t pass_row_base = pass * R_MAX;
            own_pass_rows = (row_count > pass_row_base)
                                ? ((row_count - pass_row_base < R_MAX) ? row_count - pass_row_base : R_MAX)
                                : 0;
            for (uint32_t r = 0; r < own_pass_rows; ++r) {
                volatile tt_l1_ptr uint32_t* bm = bitmaps + r * bitmap_words;
                for (uint32_t wd = 0; wd < bitmap_words; ++wd) {
                    bm[wd] = 0;
                }
                const uint32_t ri = pass_row_base + r;
                const uint32_t q_tile = row_start + (ri >> VSA_ROW_CHUNK_LOG2) * row_stride +
                                        (ri & ((1u << VSA_ROW_CHUNK_LOG2) - 1));
                noc.async_read(
                    idx, idx_cb, idx_row_bytes, {.page_id = head * n_q_tiles + q_tile}, {.offset_bytes = 0});
                noc.async_read_barrier();
                for (uint32_t e = 0; e < W; ++e) {
                    const uint32_t bb = idx_ptr[e];
                    if (bb == sentinel) {
                        break;
                    }
                    bm[bb >> 5] |= (1u << (bb & 31));
                }
            }
            own_row_parity = 0;
            own_row_seen = 0;
            for (uint32_t r = 0; r < own_pass_rows; ++r) {
                own_np[r] = 0;
            }
            uint32_t pair[2];
            uint32_t np = 0;
            for (uint32_t b = 0; b < n_kv_blocks; ++b) {
                if (counts_ptr[b] == 0) {
                    continue;  // pad block: never listed
                }
                pair[np++] = b;
                if (np < 2) {
                    continue;
                }
                if (fetched - arrival + 2 > kFetchLag) {
                    publish_pending(fetched - arrival + 2 - kFetchLag);
                }
                issue_pair(pair, 2);
                np = 0;
            }
            if (np == 1) {
                if (fetched - arrival + 1 > kFetchLag) {
                    publish_pending(fetched - arrival + 1 - kFetchLag);
                }
                issue_pair(pair, 1);
            }
            // Sentinel kreq FIRST (the writer acks its pending blocks on seeing it), then drain
            // the prefetch queue, then the sentinel log entry (no slot, no arrival).
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = sentinel;
                rq[1] = 0;
                rq[2] = kNoBlock;
                rq[3] = 0;
            }
            kreq_cb.push_back(1);
            while (arrival < fetched) {
                publish_pending((fetched - arrival >= 2) ? 2 : 1);
            }
            {
                const uint32_t sb = sentinel, ss = 0;
                publish_run(&sb, &ss, 1);
            }
            // Own pass end: close the open window and FLUSH the rows FIRST (the FLUSH handler
            // drains compute's deferred PV, releasing the last window's stashed credits), then
            // reclaim the FIFO.
            if (row_count > 0) {
                own_close_window();
                for (uint32_t r = 0; r < own_pass_rows; ++r) {
                    ctrl_cb.reserve_back(1);
                    {
                        volatile tt_l1_ptr uint32_t* cp =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                        cp[0] = MSG_FLUSH;
                        cp[1] = r;
                        cp[2] = (own_row_parity >> r) & 1u;
                    }
                    ctrl_cb.push_back(1);
                }
                while (own_head != own_tail) {
                    own_poll_credits();
                }
                own_advance();
            }
        }
        // Drain: every arrival consumed before the program ends (so no worker still reads our
        // L1), and no arrivals-sem atomic may still be in flight when the next program resets it.
        wait_all_workers_at(arrival);
        noc_async_atomic_barrier(noc.get_noc_id());
        return;
    }
#else

    // ---------------- WORKER ----------------
    if (worker_index >= n_workers) {
        return;  // idle spare core (outside the active group): it must NOT signal READY -- the
                 // leader counts n_active incs, and an idle core's early inc would let publishing
                 // start before an active worker finished zeroing its log ring (wiping its early
                 // entries). An ACTIVE worker with zero rows still runs: it consumes arrivals as
                 // skips and posts the progress the leader's slot gate counts.
    }
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
    // per-half window-pending block lists: blocks pulled within one window are freed together
    // after the window's batched visits, so a row folds every block it lists in the window into
    // ONE visit. The slot space is split into two window halves, double-buffered.
    //
    // Windows are emitted LAZILY: closing a window sends the writer its K marker and leaves the
    // window pending; its visits are emitted only once a NON-BLOCKING trid check says the half's
    // V pulls landed and the writer's (equally lazy) kack arrived. The reader keeps consuming
    // arrivals into the other half meanwhile -- the blocking window drain this replaces was 60%
    // of the reader's wall time (measured), stalled on leader-NIU congestion.
    uint32_t row_parity_bits = 0;
    uint32_t row_seen_bits = 0;
    constexpr uint32_t half_slots = stream_depth / 2;
    uint32_t pending[2][32][half_slots > 0 ? half_slots : 1];
    uint32_t n_pending[2][32];
    uint32_t half_outstanding[2] = {0, 0};  // credits still owed by compute for each half

    // Progress mailbox: this worker's consumed-arrival count, staged locally (own ackbox word) and
    // posted to the leader's ackbox word for this worker (same word offset both sides: L1-to-L1
    // NoC transfers must share their 16-byte phase). Posts are bounded by `post_limit`, the index
    // of the oldest listed arrival whose pull has NOT been confirmed landed -- the leader recycles
    // its slot on the strength of the post, and an in-flight read would then fetch the wrong block.
    uint32_t consumed = 0;
    uint32_t posted = 0;
    uint32_t post_limit = 0xFFFFFFFFu;
    const uint32_t my_box_local = ackbox_l1 + worker_index * 4;
    const uint64_t my_box_remote = get_noc_addr(leader_x, leader_y, ackbox_l1 + worker_index * 4, noc.get_noc_id());
    const auto post_progress_now = [&]() {
        const uint32_t target = (consumed < post_limit) ? consumed : post_limit;
        if (posted == target) {
            return;
        }
        posted = target;
        ackbox[worker_index] = target;
        noc_async_write(my_box_local, my_box_remote, 4, noc.get_noc_id());
    };

    // Per-half V-pull trid groups (half h -> trids 4h+1 .. 4h+4): a half's landing is checked
    // with four non-blocking outstanding-count reads instead of a blocking drain.
    const uint32_t v_l1_base = v_cb.get_write_ptr();
    const auto issue_pull = [&](uint32_t half, uint32_t idx, uint32_t leader_slot, uint32_t slot) {
        const uint32_t trid = half * 4 + 1 + (idx & 3);
        if (idx >= 4) {
            experimental::async_read_barrier_with_trid(noc, trid);  // reuse within this window
        }
        experimental::set_read_trid(noc, trid);
        noc_async_read(
            get_noc_addr(leader_x, leader_y, v_l1_base + leader_slot * v_tiles_per_block * v_tile_bytes,
                         noc.get_noc_id()),
            v_l1_base + slot * v_tiles_per_block * v_tile_bytes, v_tiles_per_block * v_tile_bytes,
            noc.get_noc_id());
        experimental::set_read_trid(noc, 0);
    };
    const auto half_landed = [&](uint32_t half) {
        for (uint32_t t = 0; t < 4; ++t) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc.get_noc_id(), half * 4 + 1 + t)) {
                return false;
            }
        }
        return true;
    };

    // Pending (closed, not yet emitted) windows, oldest first; at most both halves.
    struct PendingWin {
        uint32_t half;
        uint32_t n_slots;
        uint32_t first_listed;  // arrival index of its first listed block (post_limit source)
    };
    PendingWin pendq[2];
    uint32_t pend_head = 0, pend_tail = 0;
    uint32_t half = 0;              // the half the OPEN window fills
    uint32_t window_slots = 0;      // pulled blocks in the open window
    uint32_t window_first_listed = 0xFFFFFFFFu;
    uint32_t cur_pass_rows = 0;

    // Close the open window: K marker to the writer (it acks lazily, in order), queue for
    // emission. No waiting of any kind here.
    const auto close_window = [&]() {
        if (window_slots == 0) {
            return;
        }
        kreq_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* rq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
            rq[0] = 0xFFFFFFFFu;
            rq[1] = half;
        }
        kreq_cb.push_back(1);
        pendq[pend_tail & 1] = {half, window_slots, window_first_listed};
        ++pend_tail;
        window_slots = 0;
        window_first_listed = 0xFFFFFFFFu;
        half ^= 1;
    };

    // Emit the oldest pending window if its V pulls landed and its kack arrived. Returns true
    // if it emitted (so callers can re-poll).
    const auto try_emit = [&]() {
        if (pend_head == pend_tail) {
            return false;
        }
        const PendingWin& w = pendq[pend_head & 1];
        if (!half_landed(w.half) || !cb_pages_available_at_front(cb_kack, 1)) {
            return false;
        }
        kack_cb.wait_front(1);
        kack_cb.pop_front(1);
        for (uint32_t r = 0; r < cur_pass_rows; ++r) {
            if (n_pending[w.half][r] == 0) {
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
                cp[0] = MSG_VISIT | (n_pending[w.half][r] << 16);
                cp[1] = info;
                for (uint32_t j = 0; j < n_pending[w.half][r]; ++j) {
                    cp[2 + j] = pending[w.half][r][j];
                }
            }
            ctrl_cb.push_back(1);
            n_pending[w.half][r] = 0;
        }
        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = MSG_WINDOW;
            cp[1] = w.n_slots;
        }
        ctrl_cb.push_back(1);
        half_outstanding[w.half] = w.n_slots;
        ++pend_head;
        // The safe-post bound moves to the next un-landed window's first listed arrival.
        if (pend_head != pend_tail) {
            post_limit = pendq[pend_head & 1].first_listed;
        } else if (window_first_listed != 0xFFFFFFFFu) {
            post_limit = window_first_listed;
        } else {
            post_limit = 0xFFFFFFFFu;
        }
        post_progress_now();
        return true;
    };

    uint32_t log_n = 0;
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
            const uint32_t q_tile = row_start + (ri >> VSA_ROW_CHUNK_LOG2) * row_stride + (ri & ((1u << VSA_ROW_CHUNK_LOG2) - 1));
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
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t r = 0; r < pass_rows; ++r) {
                n_pending[h][r] = 0;
            }
        }
        cur_pass_rows = pass_rows;

        while (true) {
            // Poll pending emissions every iteration; when about to block on the next arrival,
            // close the open window first (the leader's next publish can be gated on our posted
            // progress, which is bounded by un-emitted windows) and keep polling in the spin.
            try_emit();
            const uint32_t entry_off = (log_n % log_depth) * log_entry_words;
            invalidate_l1_cache();
            if (log_ptr[entry_off + 2] != log_n + 1) {
                close_window();
                do {
                    try_emit();
                    post_progress_now();
                    invalidate_l1_cache();
                } while (log_ptr[entry_off + 2] != log_n + 1);
            }
            const uint32_t b = log_ptr[entry_off + 0];
            const uint32_t leader_slot = log_ptr[entry_off + 1];
            ++log_n;
            if (b == sentinel) {
                close_window();
                break;
            }

            uint32_t listing[32];
            uint32_t n_listing = 0;
            const uint32_t wd = b >> 5;
            const uint32_t bit = 1u << (b & 31);
            for (uint32_t r = 0; r < pass_rows; ++r) {
                if (bitmaps[r * bitmap_words + wd] & bit) {
                    listing[n_listing++] = r;
                }
            }
            if (n_listing == 0) {
                ++consumed;
                if (consumed - posted >= 4) {
                    post_progress_now();
                }
                continue;
            }

            // Claim a slot in the open window half, closing first if the half is full and
            // reclaiming the half's previous credits before its first reuse.
            if (window_slots == half_slots) {
                close_window();
            }
            if (window_slots == 0) {
                // A still-pending window on this half owns its slots AND its pending visit
                // lists: it must emit before the half refills, even when the half has never
                // been emitted before (half_outstanding == 0 -- guard on the QUEUE, not credits).
                while (pend_head != pend_tail && pendq[pend_head & 1].half == half) {
                    try_emit();
                    post_progress_now();
                }
                if (half_outstanding[half] > 0) {
                    while (!cb_pages_available_at_front(cb_free, half_outstanding[half])) {
                        try_emit();
                        post_progress_now();
                    }
                    free_cb.wait_front(half_outstanding[half]);
                    free_cb.pop_front(half_outstanding[half]);
                    half_outstanding[half] = 0;
                }
            }
            const uint32_t slot = half * half_slots + window_slots;

            // K rides the local writer (other NoC): send it the leader slot to pull from, tagged
            // with the open half so its lazy per-half ack can barrier only this window's trids.
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = leader_slot;
                rq[1] = slot;
                rq[2] = half;
            }
            kreq_cb.push_back(1);
            issue_pull(half, window_slots, leader_slot, slot);
            if (window_first_listed == 0xFFFFFFFFu) {
                window_first_listed = consumed;
                if (pend_head == pend_tail) {
                    post_limit = consumed;  // first unconfirmed pull: posts stop here until landed
                }
            }
            ++consumed;
            if (consumed - posted >= 4) {
                post_progress_now();
            }

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
                pending[half][listing[i]][n_pending[half][listing[i]]++] = entry;
            }
            ++window_slots;
        }

        // Drain every pending window (the pass's FLUSH messages must follow all its visits).
        while (pend_head != pend_tail) {
            try_emit();
        }
        post_progress_now();

        // FLUSH each pass row with its final state parity.
        for (uint32_t r = 0; r < pass_rows; ++r) {
            ASSERT(row_seen_bits & (1u << r));  // every row lists at least one block
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
#endif  // VSA_IS_LEADER
}

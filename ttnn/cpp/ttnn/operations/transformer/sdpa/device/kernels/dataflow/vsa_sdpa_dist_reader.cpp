// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa distributed-window (v18) reader. Every core of a head group is a PEER with two duties:
//
//  OWNER  -- the KV block sequence is walked in group windows of G = n_peers * kOwnSlots blocks; in
//            window w peer p fetches blocks w*G + p*kOwnSlots + j (j < kOwnSlots) from DRAM into its
//            OWNED slots [0, kOwnSlots) (V here, K via the writer), builds their ragged masks, and
//            multicasts ready[p] = w+1 to the group. Block -> (owner, slot) is a pure function of the
//            block id, so no per-block publish exists. A peer refetches for window w+1 only after
//            every peer posted done(w) (the whole group has finished gathering from window w).
//  CONSUMER -- for each resident row, the row's listed blocks inside the window are gathered into
//            ONE visit: blocks this peer owns are referenced in place, the others are pulled from
//            the owner's L1 into GATHER slots [kOwnSlots, stream_depth). A row with ~11% density
//            lists ~5 of a 48-block window, so visits carry ~5 blocks instead of the streaming
//            kernel's ~1.3: the per-visit softmax bookkeeping is amortised ~4x. Visits are emitted
//            lazily (trid landed + writer kack), several per window message, and gather slots come
//            back as compute credits in order.
//
// Windows are a function of the inputs only (no timing-driven partition): the kernel is
// deterministic. Deadlock freedom: done(w) posts need ready(w) from every owner; owners publish
// ready(w) before consuming w; refetch of w+1 waits only on done(w).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"
#include "dataflow_common.hpp"

#if defined(VSA_PROBE) && VSA_PROBE == 9
#define VSA_TICK() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#define VSA_LAP(acc) lap(acc)
#else
#define VSA_LAP(acc) ((void)0)
#endif

__attribute__((noinline)) void vsa_trap_dist(uint32_t) {
    for (;;) {
        invalidate_l1_cache();
    }
}

constexpr uint32_t MSG_VISIT = 0;   // {type | n_blocks<<16, rowinfo, (slot | count<<8 | vmask<<15) x n}
constexpr uint32_t MSG_FLUSH = 1;   // {type, row_slot, parity}
constexpr uint32_t MSG_WINDOW = 2;  // {type, n_slots}: compute returns n_slots gather credits
constexpr uint32_t ROW_PARITY = 1u << 8;
constexpr uint32_t ROW_IS_FIRST = 1u << 9;

// kreq page (16 B) to the writer: word0 = kind | group << 8, word1/2 per kind
constexpr uint32_t KREQ_FETCH = 1;  // word1 = block id, word2 = dst slot          (owned slice, K from DRAM)
constexpr uint32_t KREQ_PULL = 2;   // word1 = x | y<<8 | src_slot<<16, word2 = dst slot (K from a peer's L1)
constexpr uint32_t KREQ_MARK = 3;   // group's transfers complete -> one kack once they landed
constexpr uint32_t KREQ_END = 4;    // nothing more: the writer exits once its outputs drained
constexpr uint32_t GROUP_OWN = 15;  // kreq/kack tag of the owned-slice fetch (pull messages use their ring index)

constexpr uint32_t kMaxPeers = 16;
constexpr uint32_t kReadyMagic = 0x52454459u;
constexpr uint32_t kAckboxReady = 16;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);            // index row width (entries)
    constexpr uint32_t n_kv_blocks = get_compile_time_arg_val(1);  // T / block_size
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(2);    // S / 64 per head
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t R_MAX = get_compile_time_arg_val(4);
    constexpr uint32_t stream_depth = get_compile_time_arg_val(5);
    constexpr uint32_t log_depth = get_compile_time_arg_val(6);  // unused (kept for a shared arg layout)
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
    constexpr uint32_t cb_log = get_compile_time_arg_val(19);  // used as the READY board (n_peers words)
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(20);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(21);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(22);
    constexpr uint32_t cb_free = get_compile_time_arg_val(23);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(24);
    constexpr uint32_t cb_ackbox = get_compile_time_arg_val(25);  // [0,16) done posts, [16,32) READY flags
    (void)get_compile_time_arg_val(26);                           // sem_arrivals (unused)
    (void)log_depth;

    constexpr auto v_args = TensorAccessorArgs<27, 0>();
    constexpr auto idx_args =
        TensorAccessorArgs<v_args.next_compile_time_args_offset(), v_args.next_common_runtime_args_offset()>();
    constexpr auto counts_args =
        TensorAccessorArgs<idx_args.next_compile_time_args_offset(), idx_args.next_common_runtime_args_offset()>();

#ifndef VSA_SLICE
#define VSA_SLICE 6
#endif
    constexpr uint32_t kOwnSlots = VSA_SLICE;            // owned slice per peer per window
    constexpr uint32_t kOwnBufs = 2;                     // slices double-buffered: window w+1 prefetched during w
    constexpr uint32_t kOwnBase = kOwnSlots * kOwnBufs;  // first gather slot
    constexpr uint32_t kGatherSlots = stream_depth - kOwnBase;  // pulled blocks in flight
    constexpr uint32_t kVisitMax = 6;                           // entries per visit (<= compute's stream_depth/2)
    constexpr uint32_t kMsgPulls = 2;                           // pulled blocks per message: gather ring / kMsgPulls
                                                                // messages can be in flight (pipelining vs compute)
    static_assert(kGatherSlots >= kMsgPulls, "distributed kernel needs >= 6 gather slots");
    static_assert(kVisitMax <= stream_depth / 2, "visit width exceeds the compute kernel's cap");
    constexpr uint32_t kMaxVisitsPerMsg = 8;     // <= compute's visit buffer (16)
    constexpr uint32_t kPendMax = 8;             // messages issued and not yet emitted
    constexpr uint32_t kOwnTrid = kPendMax + 1;  // own-slice V reads; messages use 1..kPendMax
    static_assert(kOwnTrid <= 15, "trid budget");
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint32_t bitmap_words = (n_kv_blocks + 31) / 32;
    constexpr uint32_t sentinel = 0xFFFFFFFFu;

    uint32_t argi = 0;
    const uint32_t v_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t idx_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t head = get_arg_val<uint32_t>(argi++);
    const uint32_t my_peer = get_arg_val<uint32_t>(argi++);
    const uint32_t n_peers = get_arg_val<uint32_t>(argi++);
    const uint32_t n_passes = get_arg_val<uint32_t>(argi++);
    const uint32_t row_count = get_arg_val<uint32_t>(argi++);
    // raw-selection inputs (see VsaSdpaParams)
    const uint32_t list_len_arg = get_arg_val<uint32_t>(argi++);
    const uint32_t dense_mask_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t cb_dense = get_arg_val<uint32_t>(argi++);
    const uint32_t dense_bytes = get_arg_val<uint32_t>(argi++);
    const uint32_t coarse_shift = get_arg_val<uint32_t>(argi++);
    const uint32_t coarse_pad = get_arg_val<uint32_t>(argi++);
    const uint32_t n_exempt = get_arg_val<uint32_t>(argi++);
    const uint32_t exempt_argi = argi;
    argi += n_exempt;
    // peers: NoC coords (x, y) per peer
    const uint32_t peer_argi = argi;
    argi += 2 * n_peers;
    // rows: pass_rows[n_passes], then q tiles pass-major (row_count of them)
    const uint32_t pass_rows_argi = argi;
    argi += n_passes;
    const uint32_t rows_argi = argi;
    argi += row_count;

    const auto peer_x = [&](uint32_t p) { return get_arg_val<uint32_t>(peer_argi + 2 * p); };
    const auto peer_y = [&](uint32_t p) { return get_arg_val<uint32_t>(peer_argi + 2 * p + 1); };
    const auto pass_rows_of = [&](uint32_t pass) { return get_arg_val<uint32_t>(pass_rows_argi + pass); };
    const auto q_tile_of = [&](uint32_t ri) { return get_arg_val<uint32_t>(rows_argi + ri); };
    const auto exempt_id = [&](uint32_t i) { return get_arg_val<uint32_t>(exempt_argi + i); };
    const uint32_t list_len = (list_len_arg == 0) ? W : list_len_arg;
    const auto real_block = [&](uint32_t b) -> uint32_t {
        return coarse_shift ? b - (b >> coarse_shift) * coarse_pad : b;
    };
    if (my_peer >= n_peers) {
        return;  // surplus core of an oversubscribed group: no slice, no rows
    }
    if (n_peers > kMaxPeers || R_MAX > 32) {
        vsa_trap_dist(0);
    }

    Noc noc;
    experimental::CB v_cb(cb_v_stream), idx_cb(cb_idxrow), counts_cb(cb_counts);
    experimental::CB ctrl_cb(cb_ctrl), kreq_cb(cb_kreq), kack_cb(cb_kack), free_cb(cb_free);
    experimental::CB ackbox_cb(cb_ackbox), ready_cb(cb_log);
    ackbox_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ackbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ackbox_cb.get_write_ptr());
    const uint32_t ackbox_l1 = ackbox_cb.get_write_ptr();
    ready_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_cb.get_write_ptr());
    const uint32_t ready_l1 = ready_cb.get_write_ptr();
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);

    // counts row: resident for the whole kernel
    counts_cb.reserve_back(1);
    noc.async_read(counts, counts_cb, counts_row_bytes, {.page_id = 0}, {.offset_bytes = 0});
    volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr());
    noc.async_read_barrier();
    invalidate_l1_cache();
    // dense-row bitmask (bit q_tile)
    volatile tt_l1_ptr uint32_t* dense_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_dense));
    if (dense_mask_addr != 0) {
        const auto dense_acc = TensorAccessor(counts_args, dense_mask_addr, dense_bytes);
        noc_async_read(dense_acc.get_noc_addr(0), get_write_ptr(cb_dense), dense_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    const auto row_is_dense = [&](uint32_t q_tile) -> bool {
        return dense_mask_addr != 0 && ((dense_words[q_tile >> 5] >> (q_tile & 31)) & 1u);
    };
    idx_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());
    experimental::CB bitmap_cb(cb_bitmap);
    bitmap_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* bitmaps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bitmap_cb.get_write_ptr());

    // ---- group start handshake: every peer zeroes its boards, posts READY to peer 0; peer 0
    // multicasts GO once all are in. No ready/done word can then land before its board is zeroed.
    for (uint32_t i = 0; i < 2 * kMaxPeers; ++i) {
        ackbox[i] = 0;
    }
    for (uint32_t i = 0; i < kMaxPeers + 1; ++i) {
        ready[i] = 0;
    }
    const uint32_t go_word = kMaxPeers;  // ready[kMaxPeers] is the GO flag
    // Publish ONE word of our board to the same word of every peer's board: unicast per peer (a few
    // 4 B writes per window). Multicast strips were tried first; a peer alone on the next grid row
    // forms a 1x1 strip that never received the others' flags (group hang), and only our own word
    // may be written anyway (a wider write would clobber the peers' words with stale copies).
    const auto mcast_word = [&](uint32_t word_index) {
        const uint32_t off = word_index * 4;
        for (uint32_t p = 0; p < n_peers; ++p) {
            if (p == my_peer) {
                continue;
            }
            noc_async_write(
                ready_l1 + off,
                get_noc_addr(peer_x(p), peer_y(p), ready_l1 + off, noc.get_noc_id()),
                4,
                noc.get_noc_id());
        }
    };
    {
        // READY -> peer 0's flag word for me; peer 0 waits for all, then GO
        const uint32_t ready_local = ackbox_l1 + (kAckboxReady + my_peer) * 4;
        const uint64_t ready_remote =
            get_noc_addr(peer_x(0), peer_y(0), ackbox_l1 + (kAckboxReady + my_peer) * 4, noc.get_noc_id());
        ackbox[kAckboxReady + my_peer] = kReadyMagic;
        if (my_peer == 0) {
            for (uint32_t p = 1; p < n_peers; ++p) {
                while (ackbox[kAckboxReady + p] != kReadyMagic) {
                    invalidate_l1_cache();
                }
            }
            ready[go_word] = 1;
            mcast_word(go_word);
        } else {
            do {
                noc_async_write(ready_local, ready_remote, 4, noc.get_noc_id());
                noc_async_write_barrier();
                invalidate_l1_cache();
            } while (ready[go_word] == 0);
        }
    }

    // ---- geometry of the group window ----
    const uint32_t G = n_peers * kOwnSlots;  // blocks per group window
    const uint32_t n_windows = (n_kv_blocks + G - 1) / G;
    const uint32_t v_l1_base = v_cb.get_write_ptr();
    const uint32_t v_block_bytes = v_tiles_per_block * v_tile_bytes;
    const uint32_t k_l1_base = get_write_ptr(cb_k_stream);  // K pulls ride this NoC too (no writer round trip)
    const uint32_t k_block_bytes = k_tiles_per_block * k_tile_bytes;
    const uint32_t v_base = head * v_head_stride;

#if defined(VSA_PROBE) && VSA_PROBE == 9
    // per-phase wall-clock accounting (probe 9), printed by two peers of head 0
    uint32_t t_last = VSA_TICK(), t_begin = t_last;
    uint32_t t_scan = 0, t_alloc = 0, t_issue = 0, t_emit = 0, t_done = 0, t_fetch = 0, t_ready = 0, t_flush = 0;
    uint32_t n_msgs = 0, n_pulls = 0, n_nudges = 0, n_visits = 0;
    uint32_t t_land_sum = 0, n_wait_v = 0, n_wait_k = 0, n_alloc_pend = 0, n_alloc_empty = 0;
    const auto lap = [&](uint32_t& acc) {
        const uint32_t now = VSA_TICK();
        acc += now - t_last;
        t_last = now;
    };
#endif
    // ---- done posts: my window-progress word on every peer (including me) ----
    const uint32_t my_done_local = ackbox_l1 + my_peer * 4;
    const auto post_done = [&](uint32_t value) {
        ackbox[my_peer] = value;
        for (uint32_t p = 0; p < n_peers; ++p) {
            if (p == my_peer) {
                continue;
            }
            noc_async_write(
                my_done_local,
                get_noc_addr(peer_x(p), peer_y(p), ackbox_l1 + my_peer * 4, noc.get_noc_id()),
                4,
                noc.get_noc_id());
        }
    };
    const auto all_done_at_least = [&](uint32_t value) -> bool {
        invalidate_l1_cache();
        for (uint32_t p = 0; p < n_peers; ++p) {
            if (ackbox[p] < value) {
                return false;
            }
        }
        return true;
    };

    // ---- kreq helpers ----
    const auto kreq = [&](uint32_t w0, uint32_t w1, uint32_t w2) {
        kreq_cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* rq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
        rq[0] = w0;
        rq[1] = w1;
        rq[2] = w2;
        rq[3] = 0;
        kreq_cb.push_back(1);
    };
    // Every read of a message (its K and V pulls) carries the message's trid; a message has landed
    // when that trid is flushed. The owned-slice V reads use kOwnTrid.
    const auto trid_flushed = [&](uint32_t trid) -> bool {
        return ncrisc_noc_read_with_transaction_id_flushed(noc.get_noc_id(), trid);
    };
    const auto tagged_read = [&](uint64_t src, uint32_t dst, uint32_t bytes, uint32_t trid) {
        experimental::set_read_trid(noc, trid);
        noc_async_read(src, dst, bytes, noc.get_noc_id());
        experimental::set_read_trid(noc, 0);
    };
    const auto build_vmask = [&](uint32_t slot, uint32_t count) -> bool {
        const bool needs = (count < block_size) && (count % keys_per_tile) != 0;
        if (needs) {
            constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
            fill_vertical_tile_bf16<mask_tile_bytes>(noc, cb_vmask, slot, count % keys_per_tile);
        }
        return needs;
    };

    // ---- gather-slot ring and lazy visit emission ----
    // Visits are collected into a pending message (<= kMaxVisitsPerMsg visits, <= kGatherSlots pulled
    // blocks) and emitted once its pulls landed and the writer acked its K pulls. Two pending
    // messages may be open at once (trid groups 0/1). Gather slots are allocated round-robin and
    // credited back by compute in message order, so a count suffices.
    struct PendMsg {
        uint32_t trid;
        uint32_t t_close;  // probe 9: landing latency
        uint32_t n_visits;
        uint32_t n_pulled;
        uint32_t vis_row[kMaxVisitsPerMsg];
        uint32_t vis_n[kMaxVisitsPerMsg];
        uint32_t vis_entries[kMaxVisitsPerMsg][kVisitMax];
    };
    // pending ring lives in L1 behind the ready board (cb_log is sized for it in distributed mode)
    PendMsg* pendq = reinterpret_cast<PendMsg*>(ready_l1 + 128);
    uint32_t pend_head = 0, pend_tail = 0;
    uint32_t gather_next = 0;   // next gather slot (ring index)
    uint32_t slots_in_use = 0;  // gather slots allocated and not yet credited back
    // Emitted messages awaiting their credits, in order: compute returns n_pulled + 1 pages per
    // message (one phantom so all-owned messages are accounted too).
    constexpr uint32_t kEmitRing = 64;
    uint32_t emitted_pulled[kEmitRing];
    uint32_t emit_head = 0, emit_tail = 0;
    uint32_t row_parity_bits = 0, row_seen_bits = 0;
    bool nudged = false;  // a NUDGE is outstanding for the credits compute holds in its deferred PV

    const auto reclaim_credits = [&]() {
        while (emit_head != emit_tail) {
            const uint32_t n = emitted_pulled[emit_head % kEmitRing] + 1;
            if (!cb_pages_available_at_front(cb_free, n)) {
                return;
            }
            free_cb.wait_front(n);
            free_cb.pop_front(n);
            slots_in_use -= n - 1;
            ++emit_head;
        }
    };
    const auto ctrl_reserve = [&]() {
        while (!cb_pages_reservable_at_back(cb_ctrl, 1)) {
            reclaim_credits();
        }
        ctrl_cb.reserve_back(1);
    };
    // kack pages carry their tag: the owned-slice ack (GROUP_OWN; its prefetch is in flight across a
    // window) and the pull-message acks (issued and acked in message order) share the FIFO
    uint32_t pull_acks = 0;
    bool own_acked = false;
    const auto poll_kacks = [&]() {
        while (cb_pages_available_at_front(cb_kack, 1)) {
            kack_cb.wait_front(1);
            volatile tt_l1_ptr uint32_t* kp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kack_cb.get_read_ptr());
            invalidate_l1_cache();
            const uint32_t g = kp[0];
            kack_cb.pop_front(1);
            if (g == GROUP_OWN) {
                own_acked = true;
            } else {
                ++pull_acks;
            }
        }
    };
    const auto try_emit = [&]() -> bool {
        if (pend_head == pend_tail) {
            return false;
        }
        PendMsg& m = pendq[pend_head % kPendMax];
        if (m.n_pulled > 0) {
            poll_kacks();
            if (pull_acks == 0 || !trid_flushed(m.trid)) {
#if defined(VSA_PROBE) && VSA_PROBE == 9
                if (!trid_flushed(m.trid)) {
                    ++n_wait_v;  // V (this NoC) still in flight
                } else {
                    ++n_wait_k;  // V landed, K ack pending
                }
#endif
                return false;
            }
            --pull_acks;
        }
#if defined(VSA_PROBE) && VSA_PROBE == 9
        t_land_sum += VSA_TICK() - m.t_close;
#endif
        for (uint32_t i = 0; i < m.n_visits; ++i) {
            const uint32_t r = m.vis_row[i];
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
            ctrl_reserve();
            {
                volatile tt_l1_ptr uint32_t* cp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = MSG_VISIT | (m.vis_n[i] << 16);
                cp[1] = info;
                for (uint32_t j = 0; j < m.vis_n[i]; ++j) {
                    cp[2 + j] = m.vis_entries[i][j];
                }
            }
            ctrl_cb.push_back(1);
        }
        // credits: the pulled slots plus one phantom so every message is accounted (all-owned visits
        // pull nothing); reclaim_credits subtracts the phantom
        ctrl_reserve();
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = MSG_WINDOW;
            cp[1] = m.n_pulled + 1;
        }
        ctrl_cb.push_back(1);
        while (emit_tail - emit_head == kEmitRing) {
            reclaim_credits();
        }
        emitted_pulled[emit_tail % kEmitRing] = m.n_pulled;
        ++emit_tail;
        ++pend_head;
        nudged = false;
#if defined(VSA_PROBE) && VSA_PROBE == 9
        ++n_msgs;
        n_visits += m.n_visits;
#endif
        return true;
    };
    // an open (being filled) message
    PendMsg* open = nullptr;
    const auto open_msg = [&]() {
        while (pend_tail - pend_head == kPendMax) {  // ring full: wait for the oldest to emit
            reclaim_credits();
            try_emit();
        }
        open = &pendq[pend_tail % kPendMax];
        open->trid = 1 + (pend_tail % kPendMax);  // its slot's trid: free once the previous tenant emitted
        open->n_visits = 0;
        open->n_pulled = 0;
    };
    const auto close_msg = [&]() {
        if (open == nullptr || open->n_visits == 0) {
            open = nullptr;
            return;
        }
        if (open->n_pulled > 0) {
            kreq(KREQ_MARK | ((pend_tail % kPendMax) << 8), 0, 0);  // writer: ack once this message's K landed
        }
#if defined(VSA_PROBE) && VSA_PROBE == 9
        open->t_close = VSA_TICK();
#endif
        ++pend_tail;
        open = nullptr;
    };
    // One step of "make progress on credits": reclaim, emit, and when nothing is left to emit but
    // credits are outstanding, NUDGE once. Compute defers the last chunk's PV (and the window's
    // credits with it) to overlap the next chunk's QK; only a 0-slot window releases them.
    const auto credit_step = [&]() {
        reclaim_credits();
        if (try_emit()) {
            return;
        }
        if (pend_head == pend_tail && emit_head != emit_tail && !nudged) {
            ctrl_reserve();
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = MSG_WINDOW;
            cp[1] = 0;
            ctrl_cb.push_back(1);
            nudged = true;
#if defined(VSA_PROBE) && VSA_PROBE == 9
            ++n_nudges;
#endif
        }
    };
    // allocate one gather slot (blocks until compute credits one back); returns the slot index
    const auto alloc_gather = [&]() -> uint32_t {
        VSA_LAP(t_issue);
#if defined(VSA_PROBE) && VSA_PROBE == 9
        if (slots_in_use >= kGatherSlots) {
            if (pend_head == pend_tail) {
                ++n_alloc_empty;
            } else {
                ++n_alloc_pend;
            }
        }
#endif
        while (slots_in_use >= kGatherSlots) {
            credit_step();
        }
        VSA_LAP(t_alloc);
#if defined(VSA_PROBE) && VSA_PROBE == 9
        ++n_pulls;
#endif
        const uint32_t slot = kOwnBase + (gather_next % kGatherSlots);
        ++gather_next;
        ++slots_in_use;
        return slot;
    };

    // ---- owned-slice prefetch (double-buffered) ----
    // Window gw lives in own buffer gw & 1. fetch_slice(gw) issues the V reads and the K kreqs (no wait);
    // slice_ready(gw) waits for them and publishes ready[me] = gw + 1. The buffer of window gw is
    // refilled with gw + 2 only after every peer posted done(gw) (value gw + 1).
    const uint32_t total_windows = n_passes * n_windows;
    uint32_t own_count[kOwnBufs][kOwnSlots];
    uint32_t own_mask[kOwnBufs][kOwnSlots];
    uint32_t own_fetched[kOwnBufs] = {0, 0};
    // Owned slots are referenced IN PLACE by this core's visits (no credits), so a buffer may be
    // refilled only once every message emitted up to the end of its previous window came back.
    uint32_t win_emit_end[kOwnBufs] = {0, 0};
    const auto fetch_slice = [&](uint32_t gw) {
        const uint32_t buf = gw & 1;
        const uint32_t w_first = (gw % n_windows) * G;
        uint32_t n_fetch = 0;
        for (uint32_t j = 0; j < kOwnSlots; ++j) {
            const uint32_t b = w_first + my_peer * kOwnSlots + j;
            own_count[buf][j] = 0;
            own_mask[buf][j] = 0;
            if (b >= n_kv_blocks) {
                continue;
            }
            const uint32_t slot = buf * kOwnSlots + j;
            own_count[buf][j] = counts_ptr[b];
            const uint32_t v_tile0 = v_base + b * v_tiles_per_block;
#if defined(VSA_PROBE) && VSA_PROBE == 4
            (void)v_tile0;  // probe 4: no DRAM slice fetch (window protocol only)
#else
            for (uint32_t i = 0; i < v_tiles_per_block; ++i) {
                tagged_read(
                    v.get_noc_addr(v_tile0 + i),
                    v_l1_base + slot * v_block_bytes + i * v_tile_bytes,
                    v_tile_bytes,
                    kOwnTrid);
            }
            kreq(KREQ_FETCH | (GROUP_OWN << 8), b, slot);
#endif
            own_mask[buf][j] = build_vmask(slot, own_count[buf][j]) ? 1u : 0u;
            ++n_fetch;
        }
        if (n_fetch > 0) {
            kreq(KREQ_MARK | (GROUP_OWN << 8), 0, 0);
            own_acked = false;
        }
        own_fetched[buf] = n_fetch;
    };
    const auto slice_ready = [&](uint32_t gw) {
        if (own_fetched[gw & 1] > 0) {
            while (!trid_flushed(kOwnTrid)) {
                reclaim_credits();
                try_emit();
            }
            while (!own_acked) {
                poll_kacks();
                reclaim_credits();
                try_emit();
            }
        }
        ready[my_peer] = gw + 1;
        mcast_word(my_peer);
    };
    if (total_windows > 0) {
        fetch_slice(0);
        slice_ready(0);
        VSA_LAP(t_fetch);
    }

    // ---- passes ----
    uint32_t row_base = 0;
    for (uint32_t pass = 0; pass < n_passes; ++pass) {
        const uint32_t pass_rows = pass_rows_of(pass);
        row_seen_bits = 0;
        row_parity_bits = 0;
        // membership bitmaps for this pass's rows
        for (uint32_t r = 0; r < pass_rows; ++r) {
            volatile tt_l1_ptr uint32_t* bm = bitmaps + r * bitmap_words;
            for (uint32_t wd = 0; wd < bitmap_words; ++wd) {
                bm[wd] = 0;
            }
            const uint32_t q_tile = q_tile_of(row_base + r);
            if (row_is_dense(q_tile)) {
                for (uint32_t b = 0; b < n_kv_blocks; ++b) {
                    if (counts_ptr[b] != 0) {
                        bm[b >> 5] |= (1u << (b & 31));
                    }
                }
                continue;
            }
            noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = head * n_q_tiles + q_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t e = 0; e < list_len; ++e) {
                const uint32_t raw = idx_ptr[e];
                if (raw == sentinel) {
                    break;
                }
                const uint32_t b = real_block(raw);
                if (b >= n_kv_blocks) {
                    vsa_trap_dist(1);
                }
                bm[b >> 5] |= (1u << (b & 31));
            }
            for (uint32_t i = 0; i < n_exempt; ++i) {
                const uint32_t eb = exempt_id(i);
                bm[eb >> 5] |= (1u << (eb & 31));
            }
        }
        VSA_LAP(t_scan);

        for (uint32_t w = 0; w < n_windows; ++w) {
            const uint32_t gw = pass * n_windows + w;  // global window counter across passes
            const uint32_t buf = gw & 1;
            // ---- OWNER: prefetch the next window into the other buffer once its previous tenant
            // (window gw-1) has been consumed by every peer ----
            if (gw + 1 < total_windows) {
                if (gw > 0) {
                    while (!all_done_at_least(gw)) {
                        credit_step();
                    }
                    while (emit_head < win_emit_end[(gw + 1) & 1]) {  // my compute done with window gw-1
                        credit_step();
                    }
                }
                VSA_LAP(t_done);
                fetch_slice(gw + 1);
                VSA_LAP(t_fetch);
            }
            // ---- CONSUMER: this window's slices are ready on every owner ----
            for (uint32_t p = 0; p < n_peers; ++p) {
                while (ready[p] < gw + 1) {
                    reclaim_credits();
                    try_emit();
                    invalidate_l1_cache();
                }
            }
            VSA_LAP(t_ready);
            const uint32_t w_first = w * G;
            const uint32_t w_end = (w_first + G < n_kv_blocks) ? w_first + G : n_kv_blocks;
            for (uint32_t r = 0; r < pass_rows; ++r) {
                const volatile tt_l1_ptr uint32_t* bm = bitmaps + r * bitmap_words;
                // The row's listed real blocks inside the window, in block order. Visits are fixed
                // chunks of <= kVisitMax of them: a function of the row's list alone (never split at a
                // message boundary), so results do not depend on which rows share the core.
                uint32_t blist[kMaxPeers * kOwnSlots];
                uint32_t n_listed = 0;
                for (uint32_t b = w_first; b < w_end; ++b) {
                    if (((bm[b >> 5] >> (b & 31)) & 1u) && counts_ptr[b] != 0) {
                        blist[n_listed++] = b;
                    }
                }
                VSA_LAP(t_scan);
                for (uint32_t c = 0; c < n_listed; c += kVisitMax) {
                    const uint32_t n_ent = (n_listed - c < kVisitMax) ? n_listed - c : kVisitMax;
                    uint32_t n_pulls = 0;
                    for (uint32_t j = 0; j < n_ent; ++j) {
                        if ((blist[c + j] - w_first) / kOwnSlots != my_peer) {
                            ++n_pulls;
                        }
                    }
                    // The visit joins the open message only if it fits whole; else a fresh one. A row's
                    // later visits (c > 0) always start a new message: the compute's online-softmax
                    // engine processes a message chunk with ONE visit per row (the running max/corr of
                    // two same-row visits in one chunk would read each other's half-written state).
                    if (c > 0 || open == nullptr || open->n_visits == kMaxVisitsPerMsg ||
                        open->n_pulled + n_pulls > kMsgPulls) {
                        close_msg();
                        open_msg();
                    }
                    PendMsg& m = *open;
                    uint32_t pulled = 0;
                    for (uint32_t j = 0; j < n_ent; ++j) {
                        const uint32_t b = blist[c + j];
                        const uint32_t i_in_w = b - w_first;
                        const uint32_t owner = i_in_w / kOwnSlots;
                        const uint32_t oslot = buf * kOwnSlots + i_in_w % kOwnSlots;  // slot on the owner
                        const uint32_t count = counts_ptr[b];
                        uint32_t entry;
                        if (owner == my_peer) {
                            entry = oslot | (count << 8) | (own_mask[buf][i_in_w % kOwnSlots] << 15);
                        } else {
                            const uint32_t slot = alloc_gather();
#if defined(VSA_PROBE) && (VSA_PROBE == 3 || VSA_PROBE == 4)
                            // probe 3/4: protocol only -- no peer-L1 pulls (garbage output)
#else
                            // V on this NoC, K via the writer's (the two NoC rings run opposite ways: one
                            // ring carrying both measured 3x slower from congestion)
                            tagged_read(
                                get_noc_addr(
                                    peer_x(owner), peer_y(owner), v_l1_base + oslot * v_block_bytes, noc.get_noc_id()),
                                v_l1_base + slot * v_block_bytes,
                                v_block_bytes,
                                m.trid);
                            kreq(
                                KREQ_PULL | ((pend_tail % kPendMax) << 8),
                                peer_x(owner) | (peer_y(owner) << 8) | (oslot << 16),
                                slot);
#endif
                            const uint32_t vm = build_vmask(slot, count) ? 1u : 0u;
                            entry = slot | (count << 8) | (vm << 15);
                            ++pulled;
                        }
                        m.vis_entries[m.n_visits][j] = entry;
                    }
                    m.vis_row[m.n_visits] = r;
                    m.vis_n[m.n_visits] = n_ent;
                    ++m.n_visits;
                    m.n_pulled += pulled;
                }
            }
            close_msg();
            // everything gathered from this window has been ISSUED; the owners may not overwrite
            // until the pulls LANDED -- so emit (which waits for landing) before posting done.
            VSA_LAP(t_issue);
            while (pend_head != pend_tail) {
                reclaim_credits();
                try_emit();
            }
            VSA_LAP(t_emit);
            win_emit_end[buf] = emit_tail;
            post_done(gw + 1);
            // publish the prefetched next window (its reads have long landed)
            if (gw + 1 < total_windows) {
                slice_ready(gw + 1);
                VSA_LAP(t_fetch);
            }
        }

        // pass end: every message emitted; FLUSH each row with its final parity
        while (pend_head != pend_tail) {
            reclaim_credits();
            try_emit();
        }
        for (uint32_t r = 0; r < pass_rows; ++r) {
            ctrl_reserve();
            {
                volatile tt_l1_ptr uint32_t* cp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = MSG_FLUSH;
                cp[1] = r;
                cp[2] = (row_parity_bits >> r) & 1u;
            }
            ctrl_cb.push_back(1);
        }
        row_base += pass_rows;
        VSA_LAP(t_flush);
    }
    // the owned slots are still read by peers until they posted done(last); wait for that before
    // this kernel (and its L1) can go away, then flush our own NoC traffic
    while (!all_done_at_least(n_passes * n_windows)) {
        reclaim_credits();
        invalidate_l1_cache();
    }
    while (emit_head != emit_tail) {
        reclaim_credits();
    }
    kreq(KREQ_END, 0, 0);
#if defined(VSA_PROBE) && VSA_PROBE == 9
    VSA_LAP(t_flush);
    if (head == 0 && (my_peer == 0 || my_peer == 3)) {
        DPRINT(
            "VSA_RD p{} r{} tot {} scan {} alloc {} iss {} emit {} done {} fetch {} rdy {} fl {} msg {} vis {} pull {} "
            "nud {}\n",
            my_peer,
            row_count,
            t_last - t_begin,
            t_scan,
            t_alloc,
            t_issue,
            t_emit,
            t_done,
            t_fetch,
            t_ready,
            t_flush,
            n_msgs,
            n_visits,
            n_pulls,
            n_nudges);
        (void)t_land_sum;
        (void)n_wait_k;
        (void)n_wait_v;
        (void)n_alloc_pend;
        (void)n_alloc_empty;
    }
#endif
    noc.async_write_barrier();
    noc_async_atomic_barrier(noc.get_noc_id());
}

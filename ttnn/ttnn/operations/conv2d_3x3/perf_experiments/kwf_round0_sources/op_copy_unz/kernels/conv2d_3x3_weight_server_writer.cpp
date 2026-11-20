// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server, writer half (BRISC / NoC1). Per round: wait until every receiver of the rectangle can take
// it (a monotonic ack counter: one ack per receiver per round that reuses a cb_weights slot), then multicast the chunks
// the server reader hands over into the same slot of every receiver, and free the slot once the multicast has read it.
// The first weight_depth rounds wait for nothing: a program's go reaches a core only after every core finished the
// previous program and every core's semaphores and runtime args were written (fast dispatch: dispatch_s sends the go
// after the program's config writes and the previous program's completion count; slow dispatch: every core is
// configured, then launched), so a receiver's slots and flags are ready before this kernel starts, whether or not the
// receiver has started. The old kernel-start ack (one NoC atomic per receiver) arrived ~10 ns apart at this core:
// 99 receivers (res2, center_head.1) took ~0.8-1.0 us, which with the DRAM read gated round 0. Every chunk that has
// landed goes out in one linked chain, one path set-up per chain (NoC estimator, Blackhole: 16 linked 8 KB multicasts
// to 64 cores at 43 B/cycle, unlinked at 15; one chain per chunk measured ~0.3 us per 8.7 KB chunk on res3.x). The
// chain is closed by a 4 B write to the receivers' flag of this slot when the round is complete (the flag is set) or
// when the next chunk has not landed yet (a write of zero: the flag stays INVALID until this round's end, the receiver
// cleared it before its ack let this round start), so no path stays reserved across a wait. One multicast into the
// rectangle at a time on one NoC: concurrent multicasts from several senders into one rectangle measured 1.6x to 3.7x
// slower.
//
// Dual-source rectangles (kw-fold, descriptor _weight_servers): a second spare core, the partner, multicasts part of
// every chunk on NoC0 while this server multicasts the rest on NoC1 (args part_lo / part_hi: slot tiles of each
// chunk; the partner runs these two kernels with their NoCs swapped: reads on NoC1, multicast on NoC0, no bias). The
// two NoCs are separate links into the same receivers. Measured on res3.x (zoned, 77-core rectangle): one server's
// 52 KB round 0 multicast ran at ~34 GB/s beside the fill (~35 GB/s with the reads stubbed), the receivers saw round 0
// at ~3.2 us and compute waited ~1.4 us for its three kernel rows; stubbing the multicast saved ~1 us, the reads ~0.25.
// With a partner each NoC carries half: round 0 at ~2.3 us, compute waits ~0.7 us. Every flag word carries one byte
// lane per server (arg lane: 1 here, byte 0; 2 on the partner, byte 1), and each server writes its byte alone (a 1-byte
// multicast from its local flag word holding that lane's byte), so a receiver sees 0x0101 once both parts of a round
// (or of round 0's first kernel rows) landed, then clears the whole word as before. No extra semaphore: the 4-byte
// word the dispatcher zeroes at launch holds both lanes. Lane 0 (one server) keeps the 4-byte VALID writes. Slot reuse:
// the receivers ack this server only; it forwards each round's clearance to the partner (arg fwd: its NoC (x | y << 8)
// + 1), whose ack semaphore counts the forwarded rounds (SERVER_FWD_ACKS). res3.x (unzoned, medians of 3) 9.82 ->
// 8.94 us.
//
// With a bias the server also reads it once (NoC1, at kernel start) and multicasts it into every receiver's cb_bias
// (same index and size, so the same L1 address) before round 0, then sets the receivers' bias flag (the semaphore
// after the slot flags). HEIGHT: every receiver takes all of C_out. ~100 receivers reading the same bias pages from
// DRAM themselves landed it at ~7 us, behind kw-fold's ~6 us bias decode in front of the final pass. A round 0 of at
// most one NoC burst gates compute's first matmul on a short multicast, so the bias goes out after it instead (res2
// unzoned, leave-one-out: 10.87 us with the bias first, 10.27 us after); larger rounds keep it first (res3.x: a later
// bias pushed the reader's ~1.3 us-per-tile decode past the final pass, +1.9%).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#ifdef KRS_R0_SELF_READ
#include "conv2d_3x3_gather_dataflow.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t ack_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t flag_sem_id = get_compile_time_arg_val(3);  // one per weight slot, then the bias flag
    constexpr uint32_t weight_depth = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);
    constexpr bool has_bias = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t cb_bias = get_compile_time_arg_val(7);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(8);
    constexpr auto bias_args = TensorAccessorArgs<9>();
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;
    constexpr uint32_t bias_flag_sem_id = flag_sem_id + weight_depth;
#ifdef KWF_ROUND0_STREAM
    // Round 0 streams (kw-fold, conv2d_3x3_kw_fold.hpp): chunk j < num_chunks - 1 has its own flag (after the bias
    // flag), set VALID in the chain right after the multicast that completes chunks 0 .. j, so a receiver pushes and
    // compute multiplies each landed kernel row of round 0 before the round is complete. Written once, never reused.
    constexpr uint32_t early_flag_sem_id = bias_flag_sem_id + 1;
    constexpr bool stream_round0 = true;
#else
    constexpr uint32_t early_flag_sem_id = 0;
    constexpr bool stream_round0 = false;
#endif

    const uint32_t num_rounds = get_arg_val<uint32_t>(0);
    const uint32_t num_receivers = get_arg_val<uint32_t>(1);
    // The rectangle in NoC1 multicast order: start (high corner) x, y, end (low corner) x, y.
    const uint32_t sx = get_arg_val<uint32_t>(2);
    const uint32_t sy = get_arg_val<uint32_t>(3);
    const uint32_t ex = get_arg_val<uint32_t>(4);
    const uint32_t ey = get_arg_val<uint32_t>(5);
    // Dual-source rectangles (see the file comment): this server's slot tiles [part_lo, part_hi) of every chunk, its
    // flag byte lane (0: one server, whole VALID words), the partner to forward slot-reuse clearances to (0: none).
    const uint32_t part_lo = get_arg_val<uint32_t>(7);
    const uint32_t part_hi = get_arg_val<uint32_t>(8);
    const uint32_t lane = get_arg_val<uint32_t>(9);
    const uint32_t fwd = get_arg_val<uint32_t>(10);
    // KRS: round 0's split of every chunk (byte c: the first server's slot tiles of chunk c) and this server's role
    // (0: tiles [0, split_c), 1: [split_c, chunk)); an empty part multicasts nothing (only the flag lane is written).
    const uint32_t r0_split = get_arg_val<uint32_t>(11);
    const uint32_t r0_role = get_arg_val<uint32_t>(12);
    const uint32_t lane_off = lane == 2 ? 1 : 0;
    const uint64_t fwd_ack_noc =
        fwd != 0 ? get_noc_addr((fwd - 1) & 0xFF, ((fwd - 1) >> 8) & 0xFF, get_semaphore(ack_sem_id)) : 0;
    // A slot / early flag write: the whole VALID (or zero) word with one server, this server's byte lane with two.
    auto set_flag = [&](uint32_t src, uint64_t dst, uint32_t num_dests, bool linked) {
        if (lane == 0) {
            noc_semaphore_set_multicast(src, dst, num_dests, linked);
        } else {
            noc_async_write_multicast(src + lane_off, dst + lane_off, 1, num_dests, linked);
        }
    };
    const uint32_t tile_bytes = get_tile_size(cb_weights);
#ifdef KRS_R0_SELF_READ
    // KRS: this (partner) writer reads its part of round 0 itself, on its own NoC, at kernel start (kw-fold round 0:
    // column kw = 1, slot (kh, c, n) = prepared tile ((kh * 3 + 1) C_in + c, n), page runs as the server reader's);
    // the reader hands round 0's slot over unread. Chunk c's reads carry transaction id 1 + c.
    constexpr uint32_t sr_bk = KRS_SR_BK;
    constexpr uint32_t sr_rw = KRS_SR_RW;
    constexpr uint32_t sr_tn = KRS_SR_TN;
    constexpr uint32_t sr_shard_rows = KRS_SR_SROWS;
    constexpr uint32_t sr_shard_cols = KRS_SR_SCOLS;
    constexpr auto weight_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    const bool self_read = get_arg_val<uint32_t>(14) != 0;
    if (self_read) {
        const auto weights = TensorAccessor(weight_args, get_arg_val<uint32_t>(13), tile_bytes);
        conv2d_3x3::TaggedReads<false> sr_reads;
        const uint32_t l1 = get_read_ptr(cb_weights);
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t sc = (get_arg_val<uint32_t>(11) >> (8 * c)) & 0xFF;
            uint32_t t = c * chunk_tiles + (get_arg_val<uint32_t>(12) == 0 ? 0 : sc);
            const uint32_t end = c * chunk_tiles + (get_arg_val<uint32_t>(12) == 0 ? sc : chunk_tiles);
            sr_reads.begin(1 + c);
            while (t < end) {
                const uint32_t n = t % sr_rw;
                const uint32_t kc = t / sr_rw;
                const uint32_t kh = kc / sr_bk;
                const uint32_t page = ((kh * 3 + 1) * sr_bk + (kc - kh * sr_bk)) * sr_tn + n;
                uint32_t len = 1;
                if constexpr (sr_shard_rows > 0) {
                    const uint32_t row = page / sr_tn;
                    const uint32_t col = page - row * sr_tn;
                    len = sr_shard_cols - col % sr_shard_cols;
                    if (len > sr_rw - n) {
                        len = sr_rw - n;
                    }
                    if (sr_rw == sr_shard_cols && (col - n) % sr_shard_cols == 0 && n + len == sr_rw) {
                        const uint32_t in_shard = sr_shard_rows - 1 - row % sr_shard_rows;
                        const uint32_t follow = sr_bk - 1 - kc % sr_bk;
                        len += (in_shard < follow ? in_shard : follow) * sr_rw;
                    }
                    if (len > end - t) {
                        len = end - t;
                    }
                }
                sr_reads.read(weights.get_noc_addr(page), l1 + t * tile_bytes, len * tile_bytes);
                t += len;
            }
        }
        dataflow_kernel_lib::set_read_trid(sr_reads.noc, 0);
    }
    auto r0_self_landed = [&](uint32_t c) -> bool {
        return !self_read || ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + c);
    };
#endif
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_sem_id));
    // The round whose ack wait the bias multicast follows. Round 1 only with two or more slots: its ack wait then needs
    // no ack, while a non-kw-fold receiver blocks on the bias right after round 0 (with one slot, round 1 would wait
    // for that receiver's reuse ack).
    const uint32_t bias_round =
        round_tiles * tile_bytes <= NOC_MAX_BURST_SIZE && num_rounds > 1 && weight_depth >= 2 ? 1 : 0;

    [[maybe_unused]] uint32_t bias_l1 = 0;
    if constexpr (has_bias) {
        const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
        const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(6), bias_tile_bytes);
        bias_l1 = get_write_ptr(cb_bias);
        for (uint32_t n = 0; n < bias_tiles; ++n) {
            noc_async_read_page(n, bias, bias_l1 + n * bias_tile_bytes);
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(bias_flag_sem_id)) = VALID;
    }
    // The multicast source of every slot's flag.
    for (uint32_t slot = 0; slot < weight_depth; ++slot) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flag_sem_id + slot)) = VALID << (8 * lane_off);
    }

    for (uint32_t r = 0; r < num_rounds; ++r) {
        {
#if defined(SERVER_FWD_ACKS)
            // The partner: the first server's forwarded clearances, one per round that reuses a slot.
            noc_semaphore_wait_min(ack, r + 1 > weight_depth ? r + 1 - weight_depth : 0);
#elif defined(SERVER_SLOT_ACKS)
            // One counter per slot (the receivers' SERVER_SLOT_ACKS; slot 0's is the ack semaphore), so a receiver's
            // acks for later slots never stand in for another receiver's ack of this one.
            const uint32_t slot = r % weight_depth;
            const uint32_t uses = r / weight_depth;
            noc_semaphore_wait_min(
                slot == 0 ? ack
                          : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SERVER_SLOT_ACKS + slot - 1)),
                uses * num_receivers);
#else
            const uint32_t acks = r + 1 > weight_depth ? r + 1 - weight_depth : 0;
            noc_semaphore_wait_min(ack, acks * num_receivers);
#endif
            if (fwd_ack_noc != 0 && r >= weight_depth) {
                noc_semaphore_inc(fwd_ack_noc, 1);
            }
        }
        if constexpr (has_bias) {
            if (r == bias_round) {
                noc_async_read_barrier();
                const uint32_t bias_bytes = bias_tiles * get_tile_size(cb_bias);
                noc_async_write_multicast(
                    bias_l1, get_noc_multicast_addr(sx, sy, ex, ey, bias_l1), bias_bytes, num_receivers, true);
                const uint32_t bias_flag = get_semaphore(bias_flag_sem_id);
                noc_semaphore_set_multicast(
                    bias_flag, get_noc_multicast_addr(sx, sy, ex, ey, bias_flag), num_receivers, false);
            }
        }
        const uint32_t flag = get_semaphore(flag_sem_id + r % weight_depth);
        const uint64_t flag_mcast = get_noc_multicast_addr(sx, sy, ex, ey, flag);
        const uint32_t base = get_read_ptr(cb_weights);
        auto landed = [&](uint32_t c) -> bool {
#ifdef KRS_R0_SELF_READ
            if (r == 0 && !r0_self_landed(c)) {
                return false;
            }
#endif
            return cb_pages_available_at_front(cb_weights, (c + 1) * chunk_tiles);
        };
        for (uint32_t c = 0; c < num_chunks;) {
            {
                cb_wait_front(cb_weights, (c + 1) * chunk_tiles);
#ifdef KRS_R0_SELF_READ
                if (r == 0) {
                    while (!r0_self_landed(c)) {
                    }
                }
#endif
            }
            uint32_t n = 1;
            while (c + n < num_chunks && landed(c + n)) {
                ++n;
            }
#ifdef KRS_NO_R0_MCAST
            if (r == 0) {
                // Ablation: round 0's data is not multicast (only its flags).
            } else
#endif
                if (lane == 0) {
                const uint32_t src = base + c * chunk_tiles * tile_bytes;
                noc_async_write_multicast(
                    src,
                    get_noc_multicast_addr(sx, sy, ex, ey, src),
                    n * chunk_tiles * tile_bytes,
                    num_receivers,
                    true);
            } else {
                // This server's part of each landed chunk, one multicast per chunk in the same linked chain.
                for (uint32_t i = c; i < c + n; ++i) {
                    uint32_t lo = part_lo;
                    uint32_t hi = part_hi;
                    if (r == 0) {
                        const uint32_t sc = (r0_split >> (8 * i)) & 0xFF;
                        lo = r0_role == 0 ? 0 : sc;
                        hi = r0_role == 0 ? sc : chunk_tiles;
                    }
                    if (hi > lo) {
                        const uint32_t src = base + (i * chunk_tiles + lo) * tile_bytes;
                        noc_async_write_multicast(
                            src,
                            get_noc_multicast_addr(sx, sy, ex, ey, src),
                            (hi - lo) * tile_bytes,
                            num_receivers,
                            true);
                    }
                }
            }
            c += n;
            if (stream_round0 && r == 0 && c < num_chunks) {
                // Chunks 0 .. c - 1 landed: their early flag (the VALID word of this slot's flag as the source).
                const uint32_t early = get_semaphore(early_flag_sem_id + c - 1);
                set_flag(flag, get_noc_multicast_addr(sx, sy, ex, ey, early), num_receivers, landed(c));
                continue;
            }
            // Keep the chain open only into a chunk that already landed.
            if (c == num_chunks || !landed(c)) {
                set_flag(c == num_chunks ? flag : MEM_ZEROS_BASE, flag_mcast, num_receivers, false);
            }
        }
        {
            // The reader refills this slot weight_depth rounds on: the multicast must have read it.
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_weights, round_tiles);
    }
    noc_async_write_barrier();
}

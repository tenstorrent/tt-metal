// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server, writer half (BRISC / NoC1). Per round: wait until every receiver of the rectangle can take
// it (a monotonic ack counter: one ack per receiver at its kernel start, then one per round that reuses a cb_weights
// slot), then multicast each chunk the server reader hands over into the same slot of every receiver, and free the
// slot once the multicast has read it. Each chunk's linked multicast is closed by a 4 B write of zero to the receivers'
// flag of this slot (INVALID there until this round's end: the receiver cleared it before its ack let this round
// start); the last chunk's closing write sets the flag. One multicast into the rectangle at a time: concurrent
// multicasts from several senders into one rectangle measured 1.6x to 3.7x slower.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t ack_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t flag_sem_id = get_compile_time_arg_val(3);  // one per weight slot
    constexpr uint32_t weight_depth = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t num_rounds = get_arg_val<uint32_t>(0);
    const uint32_t num_receivers = get_arg_val<uint32_t>(1);
    // The rectangle in NoC1 multicast order: start (high corner) x, y, end (low corner) x, y.
    const uint32_t sx = get_arg_val<uint32_t>(2);
    const uint32_t sy = get_arg_val<uint32_t>(3);
    const uint32_t ex = get_arg_val<uint32_t>(4);
    const uint32_t ey = get_arg_val<uint32_t>(5);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_sem_id));
#if SSL_SPLIT
    // server_stream_latency: this RISC reads the first SSL_SPLIT chunks of every round itself (NoC1, one trid per
    // chunk) while the server reader reads the rest, so two RISCs issue the round's one-tile DRAM reads.
    constexpr auto weight_args = TensorAccessorArgs<6>();
    constexpr uint32_t bias_ct = weight_args.next_compile_time_args_offset();
    constexpr uint32_t split = SSL_SPLIT;
    static_assert(split >= 1 && split <= num_chunks && split <= 8);
    constexpr uint32_t block_k_tiles = SSL_BK;
    constexpr uint32_t block_n_tiles = SSL_BN;
    constexpr uint32_t num_k_blocks = SSL_NKB;
    const auto weights = TensorAccessor(weight_args, get_arg_val<uint32_t>(7), tile_bytes);
    constexpr uint32_t nbanks = weight_args.is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    uint64_t bank_base[nbanks];
    for (uint32_t b = 0; b < nbanks; ++b) {
        bank_base[b] = weights.get_noc_addr(b);
    }
    const uint32_t bank_stride = static_cast<uint32_t>(weights.get_noc_addr(nbanks) - bank_base[0]);
    const uint32_t ring_base = get_read_ptr(cb_weights);
    // Round r's own chunk c reads under trid 1 + c + 8 (r % 2): the next round's chunks go out while this round's land.
    auto own_trid = [&](uint32_t r, uint32_t c) -> uint32_t { return 1 + c + 8 * (r & 1); };
    auto own_issue_chunk = [&](uint32_t r, uint32_t c) {
        MaybeDeviceZoneScope("server_own_read");
        const uint32_t kb = r % num_k_blocks;
        const uint32_t l1 = ring_base + (r % weight_depth) * round_tiles * tile_bytes;
        noc_async_read_set_trid(own_trid(r, c));
        for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
            uint32_t p;
            if constexpr (SSL_KWF) {
                constexpr uint32_t n_tiles = block_n_tiles / 3;
                const uint32_t n = t % n_tiles;
                const uint32_t kc = t / n_tiles;
                const uint32_t kh = kc / block_k_tiles;
                const uint32_t kw = kb == 0 ? 1 : (kb == 1 ? 0 : 2);  // kw_fold.hpp KW_ROUND_ORDER
                p = ((kh * 3 + kw) * block_k_tiles + (kc - kh * block_k_tiles)) * n_tiles + n;
            } else {
                p = kb * round_tiles + t;
            }
            noc_async_read<NOC_MAX_BURST_SIZE>(
                bank_base[p % nbanks] + (p / nbanks) * bank_stride, l1 + t * tile_bytes, tile_bytes);
        }
        noc_async_read_set_trid(0);
    };
    // Own chunks issued so far: rounds [0, own_round) complete, own_chunk of round own_round. The caller's limit keeps
    // them at most one round ahead, in a slot whose multicast was already flushed.
    uint32_t own_round = 0, own_chunk = 0;
    auto own_issue_next = [&](uint32_t limit) -> bool {
        if (own_round >= num_rounds || own_round >= limit) {
            return false;
        }
        own_issue_chunk(own_round, own_chunk);
        if (++own_chunk == split) {
            own_chunk = 0;
            ++own_round;
        }
        return true;
    };
#else
    constexpr uint32_t bias_ct = 6;
#endif
#ifdef SSL_BIAS_MCAST
    // server_stream_latency: the server reads the bias once (NoC1, at kernel start) and multicasts it into every
    // receiver's cb_bias (same index and size: same L1 address), then sets the receivers' bias flag.
    constexpr auto bias_args = TensorAccessorArgs<bias_ct>();
    constexpr uint32_t TRID_BIAS = 15;
    const uint32_t bias_tile_bytes = get_tile_size(SSL_CB_BIAS);
    const uint32_t bias_l1 = get_write_ptr(SSL_CB_BIAS);
    {
        const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(6), bias_tile_bytes);
        noc_async_read_set_trid(TRID_BIAS);
        for (uint32_t n = 0; n < SSL_BIAS_TILES; ++n) {
            noc_async_read(bias.get_noc_addr(n), bias_l1 + n * bias_tile_bytes, bias_tile_bytes);
        }
        noc_async_read_set_trid(0);
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SSL_BIAS_FLAG)) = VALID;
    auto send_bias = [&]() {
        MaybeDeviceZoneScope("server_send_bias");
        noc_async_read_barrier_with_trid(TRID_BIAS);
        noc_async_write_multicast(
            bias_l1,
            get_noc_multicast_addr(sx, sy, ex, ey, bias_l1),
            SSL_BIAS_TILES * bias_tile_bytes,
            num_receivers,
            true);
        const uint32_t bias_flag = get_semaphore(SSL_BIAS_FLAG);
        noc_semaphore_set_multicast(bias_flag, get_noc_multicast_addr(sx, sy, ex, ey, bias_flag), num_receivers, false);
    };
#endif
#if SSL_SPLIT
    while (own_issue_next(1)) {
    }
#endif
    // The multicast source of every slot's flag.
    for (uint32_t slot = 0; slot < weight_depth; ++slot) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flag_sem_id + slot)) = VALID;
    }

    for (uint32_t r = 0; r < num_rounds; ++r) {
        {
            MaybeDeviceZoneScope("server_wait_ack");
            const uint32_t acks = 1 + (r + 1 > weight_depth ? r + 1 - weight_depth : 0);
#if SSL_SPLIT
            while (true) {
                invalidate_l1_cache();
                if (*ack >= acks * num_receivers) {
                    break;
                }
                own_issue_next(r + (weight_depth >= 2 ? 2 : 1));
            }
#else
            noc_semaphore_wait_min(ack, acks * num_receivers);
#endif
        }
        const uint32_t flag = get_semaphore(flag_sem_id + r % weight_depth);
        const uint64_t flag_mcast = get_noc_multicast_addr(sx, sy, ex, ey, flag);
        const uint32_t base = get_read_ptr(cb_weights);
#if defined(SSL_BIAS_MCAST) && SSL_BIAS_MCAST == 2
        if (r == 0) {
            send_bias();
        }
#endif
#ifndef SSL_LINK
        for (uint32_t c = 0; c < num_chunks; ++c) {
            {
                MaybeDeviceZoneScope("server_wait_read");
                cb_wait_front(cb_weights, (c + 1) * chunk_tiles);
            }
            MaybeDeviceZoneScope("server_send");
            const uint32_t src = base + c * chunk_tiles * tile_bytes;
            noc_async_write_multicast(
                src, get_noc_multicast_addr(sx, sy, ex, ey, src), chunk_tiles * tile_bytes, num_receivers, true);
            noc_semaphore_set_multicast(c + 1 == num_chunks ? flag : MEM_ZEROS_BASE, flag_mcast, num_receivers, false);
        }
#else
        // server_stream_latency: every landed chunk of the round goes out as one linked multicast; the chain is closed
        // (by the flag write) only when the round is complete or the next chunk has not landed yet, so no path stays
        // reserved across a wait.
#if SSL_SPLIT
        auto landed = [&](uint32_t c) -> bool {
            if (c < split) {
                return ncrisc_noc_read_with_transaction_id_flushed(noc_index, own_trid(r, c));
            }
            return cb_pages_available_at_front(cb_weights, (c + 1) * chunk_tiles);
        };
#else
        auto landed = [&](uint32_t c) -> bool {
            return cb_pages_available_at_front(cb_weights, (c + 1) * chunk_tiles);
        };
#endif
        for (uint32_t c = 0; c < num_chunks;) {
            {
                MaybeDeviceZoneScope("server_wait_read");
                while (!landed(c)) {
#if SSL_SPLIT
                    own_issue_next(r + (weight_depth >= 2 ? 2 : 1));  // while waiting: next round's own chunks
#endif
                }
            }
            MaybeDeviceZoneScope("server_send");
#if SSL_SPLIT
            // One linked write per landed chunk; after each, one own chunk read of the next round goes out on the
            // read command buffer while the write drains.
            do {
                const uint32_t src = base + c * chunk_tiles * tile_bytes;
                noc_async_write_multicast(
                    src, get_noc_multicast_addr(sx, sy, ex, ey, src), chunk_tiles * tile_bytes, num_receivers, true);
                ++c;
                own_issue_next(r + (weight_depth >= 2 ? 2 : 1));
            } while (c < num_chunks && landed(c));
#else
            uint32_t n = 1;
            while (c + n < num_chunks && landed(c + n)) {
                ++n;
            }
            const uint32_t src = base + c * chunk_tiles * tile_bytes;
            noc_async_write_multicast(
                src, get_noc_multicast_addr(sx, sy, ex, ey, src), n * chunk_tiles * tile_bytes, num_receivers, true);
            c += n;
#endif
            if (c == num_chunks || !landed(c)) {
                noc_semaphore_set_multicast(c == num_chunks ? flag : MEM_ZEROS_BASE, flag_mcast, num_receivers, false);
            }
        }
#endif
#if defined(SSL_BIAS_MCAST) && SSL_BIAS_MCAST == 1
        if (r == 0) {
            send_bias();
        }
#endif
        {
            // The reader refills this slot weight_depth rounds on: the multicast must have read it.
            MaybeDeviceZoneScope("server_flush");
            noc_async_writes_flushed();
        }
        cb_wait_front(cb_weights, round_tiles);
        cb_pop_front(cb_weights, round_tiles);
#if SSL_SPLIT
        // Round r is flushed: the next round's own chunks are due now if the waits did not issue them.
        while (own_issue_next(r + 2)) {
        }
#endif
    }
    noc_async_write_barrier();
}

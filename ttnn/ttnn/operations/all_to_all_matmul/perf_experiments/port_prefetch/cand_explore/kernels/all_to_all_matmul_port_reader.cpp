// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul port reader (RISCV_1 / NoC0), one per (direction, link) port core. Never waits on the fabric.
//
// load_entry: entry (j, t) = destination chip t's quarter of source-local head j, all batches, all K, in the
// destination slot's tile order (b, mt, k): per batch the Mq_t * Kt input pages starting at
// ((b * Hl + j) * St + t * Mq_t) * Kt are consecutive. One CB page = one fabric packet (packet_tiles tiles). Entries
// are walked in groups of consecutive entries to one destination chip, cut into packets and batches by
// all_to_all_matmul_stream.hpp (the sender walks the same way); the tiles of a group are read in order, so a packed
// packet that straddles two entries simply holds the end of one and the start of the next.
// Regime R1' (Mq % 32 != 0, compile-time `retile`): per image the quarter's source tile rows are staged in
// cb_retile_stage (one DRAM barrier), then each destination tile is re-tiled into the packet page by core-local copies
// (all_to_all_matmul_retile.hpp); the batch's single barrier before the push covers them.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_to_all_matmul_retile.hpp"
#include "all_to_all_matmul_stream.hpp"

#if defined(ABLATE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_port_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t packet_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t batch_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t B = get_compile_time_arg_val(4);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(5);
    constexpr uint32_t Kt = get_compile_time_arg_val(6);
    constexpr uint32_t St = get_compile_time_arg_val(7);
    constexpr uint32_t Hl = get_compile_time_arg_val(8);
    constexpr uint32_t Mq = get_compile_time_arg_val(9);  // rows per device
    constexpr uint32_t cb_retile_stage = get_compile_time_arg_val(10);
    constexpr bool retile = (Mq % 32) != 0;
    constexpr bool pack = get_compile_time_arg_val(11) != 0;
    // port_prefetch: cb_port_chunks holds ring_pages = depth x batch_chunks pages (depth >= 2). The reads are issued in
    // units (a batch, or one packet when per_packet), each unit's reads tagged with its own NoC transaction id; up to
    // `inflight` units are outstanding at once and the oldest is pushed as soon as its own reads land, so the next
    // units' reads stay queued at DRAM meanwhile (no stop-and-go at a batch barrier). inflight = 1, batch units = the
    // one-barrier-per-batch walk. A batch (the sender's flush unit) still never crosses a batch_chunks window.
    constexpr uint32_t ring_pages = get_compile_time_arg_val(12);
    constexpr uint32_t inflight = get_compile_time_arg_val(13) & 0xFF;
    constexpr bool per_packet = (get_compile_time_arg_val(13) >> 8) != 0;
    static_assert(inflight >= 1 && inflight <= 15, "port reader: 1..15 units in flight (trids 1..15)");
    static_assert(ring_pages % batch_chunks == 0 && ring_pages >= 2 * batch_chunks, "port ring: whole batches, >= 2");
    constexpr auto in_args = TensorAccessorArgs<14>();
    constexpr uint32_t quarter_tiles = Mq_t * Kt;
    constexpr uint32_t slot_tiles = B * quarter_tiles;
    using Walk = a2amm_stream::Walk<slot_tiles, packet_tiles, pack>;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_entries = get_arg_val<uint32_t>(1);
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    uint32_t stage = 0;
    if constexpr (retile) {
        stage = get_write_ptr(cb_retile_stage);  // scratch (never pushed)
    }

    A2AMM_ZONE("prd_all");
    // tile cursor over the entries: entry ce = (j, t), image b, tile r within the image's quarter
    uint32_t ce = 0, j = 0, b = 0, r = 0, base = 0;
    a2amm_retile::Quarter qr{};
    auto open_entry = [&](uint32_t e) {
        const uint32_t word = get_arg_val<uint32_t>(2 + e);
        j = word & 0xFFFF;
        qr = a2amm_retile::quarter_of<Mq>(word >> 16);  // aligned: t0 = t * Mq_t, shift 0
        b = 0;
        r = 0;
        base = (j * St + qr.t0) * Kt;
    };
    uint32_t pos = 0;                                          // packets reserved (read issued) so far
    const uint32_t ring_base = get_write_ptr(cb_port_chunks);  // nothing pushed yet: the ring's first page
    // batches whose reads are outstanding, oldest first: pages and trid
    uint32_t pend_n[inflight], pend_t[inflight];
    uint32_t pend_head = 0, pend_cnt = 0, pend_pages = 0, next_trid = 0;
    auto retire_oldest = [&]() {
        const uint32_t n = pend_n[pend_head], t = pend_t[pend_head];
        {
            A2AMM_ZONE("prd_barrier");  // the oldest batch's DRAM reads
            if constexpr (inflight > 1) {
                noc_async_read_barrier_with_trid(t);
            } else {
                noc_async_read_barrier();
            }
        }
        cb_push_back(cb_port_chunks, n);
        pend_head = (pend_head + 1) % inflight;
        --pend_cnt;
        pend_pages -= n;
    };
    for (uint32_t g0 = 0; g0 < n_entries;) {
        // group: entries [g0, g1) to one destination chip
        const uint32_t t_g = get_arg_val<uint32_t>(2 + g0) >> 16;
        uint32_t g1 = g0 + 1;
        while (g1 < n_entries && (get_arg_val<uint32_t>(2 + g1) >> 16) == t_g) {
            ++g1;
        }
        const uint32_t ng = g1 - g0;
        const uint32_t packets = Walk::group_packets(ng);
        Walk w;
        ce = g0;
        open_entry(ce);
        for (uint32_t c = 0; c < packets;) {
            const uint32_t room = batch_chunks - pos % batch_chunks;
            const uint32_t need = (packets - c) < room ? (packets - c) : room;
            auto open_unit = [&](uint32_t n) {
                // a full ring retires (pushes) the outstanding units first: the sender may be waiting on them
                while (pend_cnt == inflight ||
                       (pend_cnt > 0 && !cb_pages_reservable_at_back(cb_port_chunks, pend_pages + n))) {
                    retire_oldest();
                }
                {
                    A2AMM_ZONE("prd_resv");                           // reader ahead of the sender by the whole ring
                    cb_reserve_back(cb_port_chunks, pend_pages + n);  // every unpushed unit + this one
                }
                if constexpr (inflight > 1) {
                    noc_async_read_set_trid(1 + next_trid);
                }
            };
            auto close_unit = [&](uint32_t n) {
                const uint32_t k = (pend_head + pend_cnt) % inflight;
                pend_n[k] = n;
                pend_t[k] = 1 + next_trid;
                next_trid = (next_trid + 1) % inflight;
                ++pend_cnt;
                pend_pages += n;
            };
            uint32_t dst = ring_base + (pos % ring_pages) * packet_bytes;
            if constexpr (!per_packet) {
                open_unit(need);
            }
            for (uint32_t i = 0; i < need; ++i) {
                if constexpr (per_packet) {
                    open_unit(1);
                }
                uint32_t n0, n1;
                w.next(ng, n0, n1);
                for (uint32_t tt = 0; tt < n0 + n1; ++tt) {
#ifndef ABLATE_NOPORTREAD
                    if constexpr (retile) {
                        if (r == 0) {
                            a2amm_retile::stage_rows<tile_bytes, Kt>(in, base, qr.nsrc, stage);
                        }
                        a2amm_retile::retile_tile<tile_bytes, Kt, Mq>(stage, dst, qr.shift, r / Kt, r % Kt);
                    } else {
                        noc_async_read(in.get_noc_addr(base + r), dst, tile_bytes);
                    }
#endif
                    dst += tile_bytes;
                    if (++r == quarter_tiles) {
                        r = 0;
                        if (++b == B) {
                            if (++ce < g1) {
                                open_entry(ce);
                            }
                        } else {
                            base = ((b * Hl + j) * St + qr.t0) * Kt;
                        }
                    }
                }
                w.advance(n0 + n1);
                dst += (packet_tiles - (n0 + n1)) * tile_bytes;  // a short packet still owns a whole page
                if constexpr (per_packet) {
                    close_unit(1);
                }
            }
            if constexpr (!per_packet) {
                close_unit(need);
            }
            pos += need;
            c += need;
        }
        g0 = g1;
    }
    while (pend_cnt > 0) {
        retire_oldest();
    }
    if constexpr (inflight > 1) {
        noc_async_read_set_trid(0);  // leave the read command buffer untagged
    }
}

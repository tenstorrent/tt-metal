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
// (all_to_all_matmul_retile.hpp), tagged with the packet's transaction id: its barrier before the push covers them.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_to_all_matmul_retile.hpp"
#include "all_to_all_matmul_stream.hpp"
#include "ttnn/cpp/ttnn/operations/all_to_all_matmul/device/kernels/all_to_all_matmul_args.hpp"

using namespace all_to_all_matmul;

#if defined(ABLATE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_port_chunks = get_compile_time_arg_val(port_reader_ct::CB_PORT_CHUNKS);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(port_reader_ct::TILE_BYTES);
    constexpr uint32_t packet_tiles = get_compile_time_arg_val(port_reader_ct::PACKET_TILES);
    constexpr uint32_t batch_chunks = get_compile_time_arg_val(port_reader_ct::BATCH_CHUNKS);
    constexpr uint32_t B = get_compile_time_arg_val(port_reader_ct::B);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(port_reader_ct::MQ_T);
    constexpr uint32_t Kt = get_compile_time_arg_val(port_reader_ct::KT);
    constexpr uint32_t St = get_compile_time_arg_val(port_reader_ct::ST);
    constexpr uint32_t Hl = get_compile_time_arg_val(port_reader_ct::HL);
    constexpr uint32_t Mq = get_compile_time_arg_val(port_reader_ct::MQ);  // rows per device
    constexpr uint32_t cb_retile_stage = get_compile_time_arg_val(port_reader_ct::CB_RETILE_STAGE);
    constexpr bool retile = (Mq % 32) != 0;
    constexpr bool pack = get_compile_time_arg_val(port_reader_ct::PACK) != 0;
    // Reads in flight: one packet = one read unit with its own NoC transaction id (1..window); up to `window` packets
    // have their DRAM reads outstanding and the oldest is pushed as soon as its own reads land, so the reads stay
    // queued at DRAM back to back instead of draining at a per-batch barrier (port_prefetch, Perf 2).
    constexpr uint32_t window = get_compile_time_arg_val(port_reader_ct::WINDOW);
    static_assert(window >= 1 && window <= 15, "port reader: 1..15 packets in flight (trids 1..15)");
    constexpr uint32_t ring_pages = 2 * batch_chunks;  // cb_port_chunks
    constexpr auto in_args = TensorAccessorArgs<port_reader_ct::COUNT>();
    constexpr uint32_t quarter_tiles = Mq_t * Kt;
    constexpr uint32_t slot_tiles = B * quarter_tiles;
    using Walk = a2amm_stream::Walk<slot_tiles, packet_tiles, pack>;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(port_reader_common::IN_ADDR);
    const uint32_t n_entries = get_arg_val<uint32_t>(port_reader_rt::N_ENTRIES);
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
        const uint32_t word = get_arg_val<uint32_t>(port_reader_rt::ENTRIES_BASE + e);
        j = word & port_reader_rt::ENTRY_HEAD_MASK;
        qr = a2amm_retile::quarter_of<Mq>(word >> port_reader_rt::ENTRY_DEST_SHIFT);  // aligned: t0 = t * Mq_t, shift 0
        b = 0;
        r = 0;
        base = (j * St + qr.t0) * Kt;
    };
    uint32_t pos = 0;                                          // packets whose reads are issued
    const uint32_t ring_base = get_write_ptr(cb_port_chunks);  // nothing pushed yet: the ring's first page
    // packets whose reads are outstanding (pushed oldest first): trid slot of the oldest, count
    uint32_t oldest = 0, pend = 0;
    auto retire_oldest = [&]() {
        {
            A2AMM_ZONE("prd_barrier");  // the oldest packet's DRAM reads
            noc_async_read_barrier_with_trid(1 + oldest);
        }
        cb_push_back(cb_port_chunks, 1);
        oldest = (oldest + 1) % window;
        --pend;
    };
    for (uint32_t g0 = 0; g0 < n_entries;) {
        // group: entries [g0, g1) to one destination chip
        const uint32_t t_g =
            get_arg_val<uint32_t>(port_reader_rt::ENTRIES_BASE + g0) >> port_reader_rt::ENTRY_DEST_SHIFT;
        uint32_t g1 = g0 + 1;
        while (g1 < n_entries &&
               (get_arg_val<uint32_t>(port_reader_rt::ENTRIES_BASE + g1) >> port_reader_rt::ENTRY_DEST_SHIFT) == t_g) {
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
            uint32_t dst = ring_base + (pos % ring_pages) * packet_bytes;  // a batch never wraps the ring
            for (uint32_t i = 0; i < need; ++i) {
                // the window is full, or the ring is (the sender may be waiting on an outstanding packet): push first
                while (pend == window || (pend > 0 && !cb_pages_reservable_at_back(cb_port_chunks, pend + 1))) {
                    retire_oldest();
                }
                {
                    A2AMM_ZONE("prd_resv");                     // reader a whole ring ahead of the sender
                    cb_reserve_back(cb_port_chunks, pend + 1);  // every outstanding packet + this one
                }
                noc_async_read_set_trid(1 + (oldest + pend) % window);
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
                ++pend;
            }
            pos += need;
            c += need;
        }
        g0 = g1;
    }
    while (pend > 0) {
        retire_oldest();
    }
    noc_async_read_set_trid(0);  // leave the read command buffer untagged
}

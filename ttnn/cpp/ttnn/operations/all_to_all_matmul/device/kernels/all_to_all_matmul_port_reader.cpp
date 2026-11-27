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
// Input tiles are addressed with the strides of all_to_all_matmul_args.hpp (plain: the pages above).
//
// q-stem mode (RT > 0), after the entries: the rope entries (the same (j, t), grouped by destination chip). A group
// is one stream of its entries' rope tiles (b, mt, rt) cut into packets of rope_packet_tiles (the sender scatters
// each tile to its own rope_seq page), batched like the entries.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_to_all_matmul_retile.hpp"
#include "all_to_all_matmul_stream.hpp"
#include "ttnn/cpp/ttnn/operations/all_to_all_matmul/device/kernels/all_to_all_matmul_args.hpp"

using namespace all_to_all_matmul;

#if defined(ABLATE_ZONES)
#include "api/debug/kernel_profiler.hpp"
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
    [[maybe_unused]] constexpr uint32_t St = get_compile_time_arg_val(port_reader_ct::ST);
    [[maybe_unused]] constexpr uint32_t Hl = get_compile_time_arg_val(port_reader_ct::HL);
    constexpr uint32_t Mq = get_compile_time_arg_val(port_reader_ct::MQ);  // rows per device
    constexpr uint32_t cb_retile_stage = get_compile_time_arg_val(port_reader_ct::CB_RETILE_STAGE);
    constexpr bool retile = (Mq % 32) != 0;
    constexpr bool pack = get_compile_time_arg_val(port_reader_ct::PACK) != 0;
    // Reads in flight: one packet = one read unit with its own NoC transaction id (1..window); up to `window` packets
    // have their DRAM reads outstanding and the oldest is pushed as soon as its own reads land, so the reads stay
    // queued at DRAM back to back instead of draining at a per-batch barrier (port_prefetch, Perf 2).
    constexpr uint32_t window = get_compile_time_arg_val(port_reader_ct::WINDOW);
    constexpr uint32_t in_batch_stride = get_compile_time_arg_val(port_reader_ct::IN_BATCH_STRIDE);
    constexpr uint32_t in_head_stride = get_compile_time_arg_val(port_reader_ct::IN_HEAD_STRIDE);
    constexpr uint32_t in_row_stride = get_compile_time_arg_val(port_reader_ct::IN_ROW_STRIDE);
    constexpr uint32_t Rt = get_compile_time_arg_val(port_reader_ct::RT);  // q-stem: rope tiles per row; 0 = off
    constexpr uint32_t RtD = Rt > 0 ? Rt : 1;  // divisor (the rope code is dead when Rt = 0)
    constexpr uint32_t rope_packet_tiles = get_compile_time_arg_val(port_reader_ct::ROPE_PACKET_TILES);
    constexpr uint32_t RptD = rope_packet_tiles > 0 ? rope_packet_tiles : 1;
    static_assert(Rt == 0 || (rope_packet_tiles >= 1 && rope_packet_tiles <= packet_tiles), "rope packet tiles");
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
    // tile cursor over the entries: entry ce = (j, t), image b, tile r within the image's quarter (tile rk of the
    // quarter's tile row whose page offset from base is roff)
    uint32_t ce = 0, j = 0, b = 0, r = 0, base = 0, rk = 0, roff = 0;
    a2amm_retile::Quarter qr{};
    auto open_entry = [&](uint32_t e) {
        const uint32_t word = get_arg_val<uint32_t>(port_reader_rt::ENTRIES_BASE + e);
        j = word & port_reader_rt::ENTRY_HEAD_MASK;
        qr = a2amm_retile::quarter_of<Mq>(word >> port_reader_rt::ENTRY_DEST_SHIFT);  // aligned: t0 = t * Mq_t, shift 0
        b = 0;
        r = 0;
        rk = 0;
        roff = 0;
        base = j * in_head_stride + qr.t0 * in_row_stride;
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
                        noc_async_read(in.get_noc_addr(base + roff + rk), dst, tile_bytes);
                    }
#endif
                    dst += tile_bytes;
                    if (++rk == Kt) {
                        rk = 0;
                        roff += in_row_stride;
                    }
                    if (++r == quarter_tiles) {
                        r = 0;
                        rk = 0;
                        roff = 0;
                        if (++b == B) {
                            if (++ce < g1) {
                                open_entry(ce);
                            }
                        } else {
                            base = b * in_batch_stride + j * in_head_stride + qr.t0 * in_row_stride;
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
    if constexpr (Rt > 0) {
        constexpr uint32_t rope_entry_tiles = B * Mq_t * Rt;
        const uint32_t rope_words = port_reader_rt::ENTRIES_BASE + n_entries + 1;
        const uint32_t n_rope = get_arg_val<uint32_t>(rope_words - 1);
        for (uint32_t g0 = 0; g0 < n_rope;) {
            const uint32_t t_g = get_arg_val<uint32_t>(rope_words + g0) >> port_reader_rt::ENTRY_DEST_SHIFT;
            uint32_t g1 = g0 + 1;
            while (g1 < n_rope && (get_arg_val<uint32_t>(rope_words + g1) >> port_reader_rt::ENTRY_DEST_SHIFT) == t_g) {
                ++g1;
            }
            uint32_t left = (g1 - g0) * rope_entry_tiles;
            const uint32_t packets = (left + RptD - 1) / RptD;
            // tile cursor: rope entry e, tile i_e = (b, mt, rt) of it; ebase = its head's first quarter row
            uint32_t e = g0, i_e = 0, ebase = 0;
            auto open_rope = [&](uint32_t idx) {
                const uint32_t word = get_arg_val<uint32_t>(rope_words + idx);
                ebase = (word & port_reader_rt::ENTRY_HEAD_MASK) * in_head_stride +
                        (word >> port_reader_rt::ENTRY_DEST_SHIFT) * Mq_t * in_row_stride + Kt;
            };
            open_rope(e);
            for (uint32_t c = 0; c < packets;) {
                const uint32_t room = batch_chunks - pos % batch_chunks;
                const uint32_t need = (packets - c) < room ? (packets - c) : room;
                uint32_t dst = ring_base + (pos % ring_pages) * packet_bytes;
                for (uint32_t i = 0; i < need; ++i) {
                    while (pend == window || (pend > 0 && !cb_pages_reservable_at_back(cb_port_chunks, pend + 1))) {
                        retire_oldest();
                    }
                    cb_reserve_back(cb_port_chunks, pend + 1);
                    noc_async_read_set_trid(1 + (oldest + pend) % window);
                    const uint32_t n = left < rope_packet_tiles ? left : rope_packet_tiles;
                    for (uint32_t tt = 0; tt < n; ++tt) {
                        const uint32_t rt = i_e % RtD;
                        const uint32_t mt = (i_e / RtD) % Mq_t;
                        const uint32_t bb = i_e / (RtD * Mq_t);
                        noc_async_read(
                            in.get_noc_addr(bb * in_batch_stride + ebase + mt * in_row_stride + rt), dst, tile_bytes);
                        dst += tile_bytes;
                        if (++i_e == rope_entry_tiles) {
                            i_e = 0;
                            if (++e < g1) {
                                open_rope(e);
                            }
                        }
                    }
                    left -= n;
                    dst += (packet_tiles - n) * tile_bytes;
                    ++pend;
                }
                pos += need;
                c += need;
            }
            g0 = g1;
        }
    }
    while (pend > 0) {
        retire_oldest();
    }
    noc_async_read_set_trid(0);  // leave the read command buffer untagged
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul port reader (RISCV_1 / NoC0), one per (direction, link) port core. Never waits on the fabric.
//
// load_entry: entry (j, t) = destination chip t's quarter of source-local head j, all batches, all K, in the
// destination slot's tile order (b, mt, k): per batch the Mq_t * Kt input pages starting at
// ((b * Hl + j) * St + t * Mq_t) * Kt are consecutive. One CB page = one fabric packet (packet_tiles tiles; the
// entry's last packet may be short). A batch is up to PORT_BATCH_CHUNKS packets, ends at a group boundary of the
// 2 x PORT_BATCH_CHUNKS CB (a reservation never wraps) and at the end of an entry; the sender mirrors this rule.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

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
    constexpr auto in_args = TensorAccessorArgs<9>();
    constexpr uint32_t quarter_tiles = Mq_t * Kt;
    constexpr uint32_t slot_tiles = B * quarter_tiles;
    constexpr uint32_t entry_chunks = (slot_tiles + packet_tiles - 1) / packet_tiles;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_entries = get_arg_val<uint32_t>(1);
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);

    A2AMM_ZONE("prd_all");
    uint32_t pos = 0;  // packets pushed so far
    for (uint32_t e = 0; e < n_entries; ++e) {
        const uint32_t word = get_arg_val<uint32_t>(2 + e);
        const uint32_t j = word & 0xFFFF;
        const uint32_t t = word >> 16;
        uint32_t b = 0, r = 0;  // tile cursor within the entry
        uint32_t base = (j * St + t * Mq_t) * Kt;
        uint32_t tile = 0;
        for (uint32_t c = 0; c < entry_chunks;) {
            const uint32_t room = batch_chunks - pos % batch_chunks;
            const uint32_t need = (entry_chunks - c) < room ? (entry_chunks - c) : room;
            cb_reserve_back(cb_port_chunks, need);
            uint32_t wp = get_write_ptr(cb_port_chunks);
            for (uint32_t i = 0; i < need; ++i) {
                const uint32_t left = slot_tiles - tile;
                const uint32_t n = left < packet_tiles ? left : packet_tiles;
                uint32_t dst = wp + i * packet_bytes;
                for (uint32_t tt = 0; tt < n; ++tt) {
#ifndef ABLATE_NOPORTREAD
                    noc_async_read(in.get_noc_addr(base + r), dst, tile_bytes);
#endif
                    dst += tile_bytes;
                    if (++r == quarter_tiles) {
                        r = 0;
                        ++b;
                        base = ((b * Hl + j) * St + t * Mq_t) * Kt;
                    }
                }
                tile += n;
            }
            noc_async_read_barrier();
            cb_push_back(cb_port_chunks, need);
            pos += need;
            c += need;
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — port-core BRISC: send_block over Fabric (raw worker API; the kernel-lib CCL sender
// programs 1-D hop routes only, and this op runs under the FABRIC_2D family, where routes are set by destination
// fabric node).
//
// Every segment of cb_xport_sum (relay sums, or a line end's own partials) is one packet into the downstream chip's
// relay_scratch page (slot * segs_per_block + seg). Increment streams (fused write + atomic inc on every inc_every-th
// segment of a stream and on its last): relay blocks count on the downstream port's arrival counter; the last block
// (the downstream chip's own) on the arrival counter (A forward / B backward) of the downstream final core that owns
// the segment (alternating halves h = (seg / L) & 1).
// Ready fence (as fabric_reduce_scatter): first tell the peer port that writes into this chip that it may, then
// wait for my own peer's ready before the first data packet. At the end: wait for everything upstream sends me,
// then re-arm my arrival counter.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(0);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);  // packets in flight before a flush (header ring)
    constexpr uint32_t inc_every = get_compile_time_arg_val(3);
    constexpr uint32_t cap_segs = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr auto scr_args = TensorAccessorArgs<6>();
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t num_headers = group;

    size_t arg = 0;
    const uint32_t scr_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_block = get_arg_val<uint32_t>(arg++);
    const uint32_t segs_per_row = get_arg_val<uint32_t>(arg++);
    const uint32_t blk_n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t first_seg = get_arg_val<uint32_t>(arg++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(arg++);  // L
    const uint32_t full = get_arg_val<uint32_t>(arg++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(arg++);    // A: my counter, and the downstream port's
    const uint32_t final_sem_addr = get_arg_val<uint32_t>(arg++);  // A or B on the downstream final cores
    const uint32_t expect_in = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t send_ready = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_x = get_arg_val<uint32_t>(arg++);
    const uint32_t ready_y = get_arg_val<uint32_t>(arg++);
    const uint32_t peer_x = get_arg_val<uint32_t>(arg++);
    const uint32_t peer_y = get_arg_val<uint32_t>(arg++);
    const uint32_t final_x0 = get_arg_val<uint32_t>(arg++);
    const uint32_t final_y0 = get_arg_val<uint32_t>(arg++);
    const uint32_t final_x1 = get_arg_val<uint32_t>(arg++);
    const uint32_t final_y1 = get_arg_val<uint32_t>(arg++);
    const uint32_t full_f0 = get_arg_val<uint32_t>(arg++);
    const uint32_t full_f1 = get_arg_val<uint32_t>(arg++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);  // entries: landing slot at the receiver
    const uint32_t entries_idx = arg;
    arg += num_blocks;
    const auto scr = TensorAccessor(scr_args, scr_addr, seg_bytes);

    if (num_blocks > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_blocks > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t relay_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        const uint64_t final_noc0 = get_noc_addr(final_x0, final_y0, final_sem_addr);
        const uint64_t final_noc1 = get_noc_addr(final_x1, final_y1, final_sem_addr);
        const uint32_t relay_total = num_blocks > 0 ? (num_blocks - 1) * full : 0;
        uint32_t sent_relay = 0, sent_f0 = 0, sent_f1 = 0;
        uint32_t h = 0, unflushed = 0, rpos = 0;
        for (uint32_t k = 0; k < num_blocks; ++k) {
            const uint32_t base = get_arg_val<uint32_t>(entries_idx + k) * segs_per_block;
            const bool fin = k + 1 == num_blocks;
            uint32_t i = 0;
            for (uint32_t seg = first_seg; seg < segs_per_block; seg += seg_stride, ++i) {
                const uint32_t row = seg / segs_per_row;
                const uint32_t c0 = (seg - row * segs_per_row) * seg_tiles;
                const uint32_t valid = blk_n_tiles - c0 < seg_tiles ? blk_n_tiles - c0 : seg_tiles;
                const uint32_t bytes = valid * tile_bytes;
                cb_wait_front(cb_xport_sum, seg_tiles * (unflushed + 1));
                const uint32_t src = get_read_ptr(cb_xport_sum) + unflushed * seg_bytes;
                const uint64_t dst = scr.get_noc_addr(base + seg, 0, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                bool inc;
                uint64_t ctr;
                if (fin) {
                    const bool h1 = (i & 1) != 0;  // final core half: seg = first + i * L, owner (l, i & 1)
                    uint32_t& sent = h1 ? sent_f1 : sent_f0;
                    ++sent;
                    inc = (sent % inc_every) == 0 || sent == (h1 ? full_f1 : full_f0);
                    ctr = h1 ? final_noc1 : final_noc0;
                } else {
                    ++sent_relay;
                    inc = (sent_relay % inc_every) == 0 || sent_relay == relay_total;
                    ctr = relay_noc;
                }
                if (inc) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                ++unflushed;
                const bool last = fin && seg + seg_stride >= segs_per_block;
                if (unflushed == group || rpos + unflushed == cap_segs || last) {
                    noc_async_writes_flushed();  // sources + headers of the in-flight packets have left L1
                    cb_pop_front(cb_xport_sum, seg_tiles * unflushed);
                    rpos += unflushed;
                    if (rpos == cap_segs) {
                        rpos = 0;
                    }
                    unflushed = 0;
                }
            }
        }
        conn.close();
    }
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}

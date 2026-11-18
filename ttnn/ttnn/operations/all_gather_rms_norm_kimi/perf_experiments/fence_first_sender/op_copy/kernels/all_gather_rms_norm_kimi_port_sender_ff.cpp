// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port sender (RISCV_0, NoC1), one per (link, direction). Adapted from examples/fabric_all_gather's sender:
//   fence      one header-only atomic inc of the ready counter of the peer port that writes into this chip, then wait
//              for my own ready >= 1 and consume it (the fence between back-to-back calls)       [TRANSPORT]
//   bulk       every chunk of my stream into the SAME `landing` offset of the SAME logical compute core on the peer;
//              every INC_EVERY-th chunk and each entry's last chunk carry a fused increment of the peer receiver's
//              arrival counter (the receiver turns it into per-core landing notifications)       [TRANSPORT]
//   stats      (link 0) whenever the hub has handed me a slot (stats_tx), send it into the peer hub's stats scratch,
//              the last packet fused with an increment of the peer hub's stats_fwd / stats_bwd. Polled while waiting
//              for bulk chunks and at every batch flush, so the stats never wait behind the bulk. While a slot is still
//              owed, at most STATS_BULK_WINDOW bulk packets stay queued in the router channel.  [OVERLAP]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "all_gather_rms_norm_kimi_perf.hpp"

using namespace tt::tt_fabric;

constexpr uint32_t cb = get_compile_time_arg_val(0);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t batch = get_compile_time_arg_val(3);
constexpr uint32_t inc_every = get_compile_time_arg_val(4);
constexpr uint32_t sem_stats_tx = get_compile_time_arg_val(7);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(8);
constexpr uint32_t max_payload = get_compile_time_arg_val(9);
constexpr uint32_t stats_window = get_compile_time_arg_val(10);  // bulk packets queued while stats are owed; 0 = off
constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count (ROUTING_MODE from the build).
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

// Copy a routed header word by word: the 2D route (fabric_set_unicast_route: table decode + byte-wise volatile writes
// of the 35-byte route buffer) measured ~0.35 us per header on BH; a 96-byte word copy is a few dozen cycles.
inline void copy_header(volatile tt_l1_ptr PACKET_HEADER_TYPE* dst, volatile tt_l1_ptr PACKET_HEADER_TYPE* src) {
    static_assert(sizeof(PACKET_HEADER_TYPE) % 4 == 0);
    auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < sizeof(PACKET_HEADER_TYPE) / 4; ++i) {
        d[i] = s[i];
    }
}

void kernel_main() {
    size_t a = 0;
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(a++);
    const uint32_t active = get_arg_val<uint32_t>(a++);
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const uint32_t n_blocks = get_arg_val<uint32_t>(a++);
    const uint32_t n_stats = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_xy = get_arg_val<uint32_t>(a++);  // the peer's port that writes into this chip
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t recv_xy = get_arg_val<uint32_t>(a++);         // the peer's receiver of my stream
    const uint32_t stats_sem_addr = get_arg_val<uint32_t>(a++);  // the peer hub's stats_fwd / stats_bwd
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t stats_idx = a;  // n_stats slots
    a += n_stats;
    const uint32_t blocks_idx = a;  // per block: core xy, n_chunks | entry_end << 17
    a += 2 * n_blocks;
    if (!active) {
        return;
    }

    // Fence first: the ready inc is the only thing the peer waits on, so it goes out before any other setup.
    //   open_start (connection reads in flight) -> route the ONE control header in their shadow -> open_finish ->
    //   send the ready inc -> set up the bulk headers (copies of the routed one) while the peer's ready is in flight
    //   -> wait for my own ready. The fence's guarantee is unchanged: no bulk write before the peer's ready.
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ctl_hdr;
    {
        MaybeDeviceZoneScope("sender_open");
        conn.open_start();
        ctl_hdr = PacketHeaderPool::allocate_header();
        route_one_hop(ctl_hdr, dst_chip_id, dst_mesh_id);
        conn.open_finish();
    }

    // ---- fence ----
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[batch];
    {
        MaybeDeviceZoneScope("sender_fence");
        ctl_hdr->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_xy >> 8, ready_xy & 0xFF, ready_addr), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(ctl_hdr), sizeof(PACKET_HEADER_TYPE));
        // in the shadow of the peer's ready: the bulk headers (only their route matters; every send rewrites the
        // command fields), copied from the routed control header (flushed above, so it is no longer read by the NoC)
        for (uint32_t h = 0; h < batch; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            copy_header(hdrs[h], ctl_hdr);
        }
        {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
    }  // sender_fence

    // ---- stats: send every slot the hub has handed over so far ----
    volatile tt_l1_ptr uint32_t* stats_tx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_tx));
    const uint64_t stats_sem_noc = get_noc_addr(hub_xy >> 8, hub_xy & 0xFF, stats_sem_addr);
    uint32_t stats_sent = 0;
    auto try_stats = [&]() {
        while (stats_sent < n_stats && *stats_tx > stats_sent) {
            const uint32_t q = get_arg_val<uint32_t>(stats_idx + stats_sent);
            const uint32_t off = q * slot_bytes;
            for (uint32_t s = 0; s < slot_bytes; s += max_payload) {
                const uint32_t bytes = (slot_bytes - s) < max_payload ? (slot_bytes - s) : max_payload;
                const uint64_t dst = get_noc_addr(hub_xy >> 8, hub_xy & 0xFF, stats_addr + off + s);
                if (s + bytes == slot_bytes) {
                    ctl_hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, stats_sem_noc, 1, true}, bytes);
                } else {
                    ctl_hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(stats_addr + off + s, bytes, reinterpret_cast<uint32_t>(ctl_hdr));
                noc_async_writes_flushed();  // ctl_hdr is reused right away
            }
            ++stats_sent;
        }
    };

    // ---- bulk ----
    const uint64_t arrival_noc = get_noc_addr(recv_xy >> 8, recv_xy & 0xFF, arrival_addr);
    uint32_t h = 0, unflushed = 0, sent = 0;
    {
        MaybeDeviceZoneScope("sender_bulk");
        for (uint32_t i = 0; i < n_blocks; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(blocks_idx + 2 * i);
            const uint32_t info = get_arg_val<uint32_t>(blocks_idx + 2 * i + 1);
            const uint32_t n = info & 0xFFFF;
            const bool entry_end = (info >> 17) & 1;
            for (uint32_t k = 0; k < n; ++k) {
                while (!cb_pages_available_at_front(cb, chunk_tiles * (unflushed + 1))) {
                    try_stats();
                }
                if constexpr (stats_window > 0) {
                    // a stats slot still owed (the relay) would queue behind every bulk packet in the channel: keep at
                    // most stats_window of them queued until it has gone out
                    while (stats_sent < n_stats &&
                           conn.num_buffers_per_channel - conn.get_num_free_write_slots() >= stats_window) {
                        try_stats();
                    }
                }
                const uint32_t src = get_read_ptr(cb) + unflushed * chunk_bytes;
                const uint64_t dst = get_noc_addr(xy >> 8, xy & 0xFF, landing_addr + k * chunk_bytes);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == batch) ? 0 : h + 1;
                ++sent;
                if ((sent % inc_every) == 0 || (entry_end && k + 1 == n)) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, chunk_bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, chunk_bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, chunk_bytes, reinterpret_cast<uint32_t>(hdr));
                --chunks_left;
                if (++unflushed == batch || chunks_left == 0) {
                    noc_async_writes_flushed();  // sources + headers of the in-flight chunks have left L1
                    cb_pop_front(cb, chunk_tiles * unflushed);
                    unflushed = 0;
                    try_stats();
                }
            }
        }
    }  // sender_bulk
    // ---- the stats that became ready only after the bulk ----
    {
        MaybeDeviceZoneScope("sender_tail_stats");
        while (stats_sent < n_stats) {
            try_stats();
        }
    }
    conn.close();
}

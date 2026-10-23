// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm port core (one per (direction, link); RISCV_0 / NoC1) — op_design.md -> exchange_wave.
// Moves bytes only: the stats all-gather over Fabric. The host (_exchange_plan) picks the transport per group:
//   Linear (and Ring with DIRECT_RING_SENDS off): one-hop store-and-forward relay — send item 0 is the own segment,
//                  item i >= 1 relays upstream's entry i-1 once it landed here (AG schedule).
//   Ring (DIRECT_RING_SENDS, lamp L6): direct multi-hop unicast — every item is the own segment, addressed straight to
//                  one peer by destination node (route by node on 2D fabrics, by hop count on 1D), first hop on this
//                  port's connection (the host matched it with ttnn.get_eth_forwarding_direction). No relay turnaround.
//
// landing[w][q] (this core's shard of the persistent landing scratch; the same address on every chip) holds slot q of
// wave w: device q's per-row partials of this link's rows, 128 B per tile-row. Every slot that lands on this port is
// one "entry" i with its OWN arrival counter (a global semaphore, arrival_addr[i]) counting waves: entries from
// different sources may arrive in any order (direct sends), entries from one source arrive in wave order.
//   ready fence    once per call: a header-only atomic inc of the ready counter of every port that writes into this
//                  chip through this port's direction (this chip finished its previous call, its landing is free);
//                  a sender waits for one ready per destination chip before its first send
//   sends (w, k)   item k of wave w once its source is in L1: own -> every root of the link published wave w
//                  (SEM_OWN_STATS >= (w+1) * n_roots); relay of entry j -> arrival[j] >= w + 1. Each item goes out as
//                  ceil(bytes / payload) packets into the destination port's same landing slot, the last one fused
//                  with an increment of the destination entry's arrival counter (header-only increment for an empty
//                  segment). Items leave strictly in (w, k) order on the one connection.
//   notify (w, i)  per landed entry (lamp L4): once arrival[i] >= w + 1 (and every earlier (w', i') was notified),
//                  increment the link's roots' landed semaphore of this direction. A root reads entry i of wave w once
//                  that semaphore reaches w * n_in + i + 1, while later entries are still on the Fabric.
//   drain          re-arm the counters by what was consumed (back to 0: trace-safe)
// Raw fabric API: the measured path on this box is the AG raw sender (op_design.md -> Helpers considered and rejected).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tools/profiler/kernel_profiler.hpp"

using namespace tt::tt_fabric;

namespace {

// Route a header to (chip, mesh) — `hops` away along this port's direction: 2D fabrics route by destination fabric
// node, 1D fabrics by hop count (ROUTING_MODE from the build).
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint16_t hops) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)hops;
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), hops);
#endif
}

// RT records (host: port args). One packet header per distinct destination chip, routed once at setup (a 2D route
// lookup per packet sits on every relay / send latency chain).
constexpr uint32_t DEST_WORDS = 3;  // chip id, mesh id, hops
constexpr uint32_t SEND_WORDS =
    4;  // slot q, destination header, destination arrival address, wait (0 own, j+1 relay j)
constexpr uint32_t READY_WORDS = 3;  // destination header, target port x, target port y

}  // namespace

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);
    constexpr uint32_t num_waves = get_compile_time_arg_val(1);
    constexpr uint32_t payload = get_compile_time_arg_val(2);
    constexpr uint32_t num_headers = get_compile_time_arg_val(3);
    constexpr uint32_t sem_own_stats_id = get_compile_time_arg_val(4);
    constexpr uint32_t sem_counted_id = get_compile_time_arg_val(6);
    static_assert(num_headers >= 1 && num_headers <= 12, "packet header pool: <= 12 headers per RISC");

    size_t a = 0;
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);   // distinct destination chips (one header each)
    const uint32_t n_send = get_arg_val<uint32_t>(a++);   // send items per wave
    const uint32_t n_in = get_arg_val<uint32_t>(a++);     // entries landing on this port per wave
    const uint32_t n_roots = get_arg_val<uint32_t>(a++);  // roots publishing into my own slot (0 if I send nothing)
    const uint32_t n_ready_out = get_arg_val<uint32_t>(a++);   // ready packets this port sends
    const uint32_t n_ready_wait = get_arg_val<uint32_t>(a++);  // readies to wait for (destination chips)
    const uint32_t my_x = get_arg_val<uint32_t>(a++);      // this port's virtual coords = the destination port's (every
    const uint32_t my_y = get_arg_val<uint32_t>(a++);      // chip uses the same logical port cores)
    const uint32_t n_notify = get_arg_val<uint32_t>(a++);  // roots I notify per entry (0 if I receive nothing)
    const bool has_conn = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t n_counted = get_arg_val<uint32_t>(a++);  // roots to signal counted(w) (0 if I send nothing / 1 wave)
    // Notify quantum: 1 = per landed entry (lamp L4), n_in = once per wave (every entry of the wave landed).
    const uint32_t notify_quantum = get_arg_val<uint32_t>(a++);
    // My own landed semaphore on the roots (one per direction: each port's notifies are counted separately, in order).
    const uint32_t landed_sem_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t arrival_idx = a;  // n_in arrival counter addresses (entry i)
    a += n_in;
    const uint32_t dest_idx = a;
    a += n_dest * DEST_WORDS;
    const uint32_t send_idx = a;
    a += n_send * SEND_WORDS;
    const uint32_t ready_idx = a;
    a += n_ready_out * READY_WORDS;
    const uint32_t seg_idx = a;
    a += num_waves;
    const uint32_t roots_idx = a;  // the link's roots (present if I notify or send)
    a += (n_notify > n_roots ? n_notify : n_roots);

    auto arrived = [&](uint32_t i) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arrival_idx + i));
    };
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* own_stats =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_stats_id));
    const uint32_t counted_sem_addr = get_semaphore(sem_counted_id);

    // Per-entry notify state, (w, i) order: `notified` entries are out; inc the roots by the newly landed run at once.
    const uint32_t notify_total = n_in > 0 ? num_waves * n_in : 0;
    uint32_t notified = 0;
    auto poll_notify = [&]() {
        uint32_t n = notified;
        while (n < notify_total && *arrived(n % n_in) >= n / n_in + 1) {
            ++n;
        }
        n -= n % notify_quantum;
        if (n > notified) {
            DeviceZoneScopedN("prt_notify");
            for (uint32_t k = 0; k < n_notify; ++k) {
                const uint32_t r = get_arg_val<uint32_t>(roots_idx + k);
                noc_semaphore_inc(get_noc_addr(r >> 16, r & 0xFFFF, landed_sem_addr), n - notified);
            }
            notified = n;
        }
    };

    if (has_conn) {
        DeviceZoneScopedN("prt_conn");
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        ASSERT(n_dest <= num_headers);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t k = 0; k < n_dest; ++k) {
            const uint32_t t = dest_idx + k * DEST_WORDS;
            hdrs[k] = PacketHeaderPool::allocate_header();
            route_to(
                hdrs[k],
                static_cast<uint16_t>(get_arg_val<uint32_t>(t)),
                static_cast<uint16_t>(get_arg_val<uint32_t>(t + 1)),
                static_cast<uint16_t>(get_arg_val<uint32_t>(t + 2)));
        }
        conn.open();
        // A header is rewritten (command fields only; the route stays) once the NoC has read its previous packet.
        uint32_t dirty = 0;
        auto next_header = [&](uint32_t k) {
            if (dirty & (1u << k)) {
                noc_async_writes_flushed();  // every in-flight header has left L1
                dirty = 0;
            }
            dirty |= 1u << k;
            return hdrs[k];
        };
        // Fence between calls: this program running here means everything earlier on this chip is done, including
        // the roots that read the previous call's landing; the ports writing into it may now.
        for (uint32_t k = 0; k < n_ready_out; ++k) {
            const uint32_t t = ready_idx + k * READY_WORDS;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(get_arg_val<uint32_t>(t));
            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                get_noc_addr(get_arg_val<uint32_t>(t + 1), get_arg_val<uint32_t>(t + 2), ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
        if (n_ready_wait > 0) {
            DeviceZoneScopedN("prt_ready_wait");
            noc_semaphore_wait_min(ready, n_ready_wait);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - n_ready_wait);
            noc_async_atomic_barrier();
        }
        // One item: the segment at `src` to the item's destination port (same landing address), the destination
        // entry's arrival counter incremented with the last packet.
        auto send_item = [&](uint32_t item, uint32_t src, uint32_t bytes) {
            const uint32_t d = get_arg_val<uint32_t>(send_idx + item * SEND_WORDS + 1);  // destination header
            const uint64_t dst_arrival =
                get_noc_addr(my_x, my_y, get_arg_val<uint32_t>(send_idx + item * SEND_WORDS + 2));
            if (bytes == 0) {
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(d);
                hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{dst_arrival, 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
                return;
            }
            for (uint32_t off = 0; off < bytes; off += payload) {
                const uint32_t n = (bytes - off) < payload ? bytes - off : payload;
                const uint64_t dst = get_noc_addr(my_x, my_y, src + off);  // same landing address on the peer
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = next_header(d);
                if (off + n == bytes) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, dst_arrival, 1, true}, n);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src + off, n, reinterpret_cast<uint32_t>(hdr));
            }
        };
        // Sends advance in (w, k) order, notifies in (w, i) order, each whenever ready (with WAVE_PIPELINE the roots
        // publish wave w+1 while wave w is still arriving, and its own segment should not wait for wave w's last
        // arrival).
        const uint32_t send_waves = n_send > 0 ? num_waves : 0;
        uint32_t sw = 0, si = 0;
        {
            DeviceZoneScopedN("prt_exchange");
            while (sw < send_waves || notified < notify_total) {
                invalidate_l1_cache();
                if (sw < send_waves) {
                    const uint32_t item = send_idx + si * SEND_WORDS;
                    const uint32_t wait = get_arg_val<uint32_t>(item + 3);
                    const bool ready_to_send =
                        wait == 0 ? *own_stats >= (sw + 1) * n_roots : *arrived(wait - 1) >= sw + 1;
                    if (ready_to_send) {
                        if (si == 0 && sw + 1 < num_waves) {
                            // Wave sw counted (item 0 is always the own segment): release the roots' wave-(sw+1)
                            // increments (SEM_OWN_COUNTED on the host).
                            for (uint32_t k = 0; k < n_counted; ++k) {
                                const uint32_t r = get_arg_val<uint32_t>(roots_idx + k);
                                noc_semaphore_inc(get_noc_addr(r >> 16, r & 0xFFFF, counted_sem_addr), 1);
                            }
                        }
                        const uint32_t q = get_arg_val<uint32_t>(item);
                        DeviceZoneScopedN("prt_send");
                        send_item(si, landing + (sw * G + q) * seg_stride, get_arg_val<uint32_t>(seg_idx + sw));
                        if (++si == n_send) {
                            si = 0;
                            ++sw;
                        }
                    }
                }
                poll_notify();
            }
        }
        {
            DeviceZoneScopedN("prt_close");
            noc_async_writes_flushed();
            conn.close();
        }
    } else {
        while (notified < notify_total) {
            invalidate_l1_cache();
            poll_notify();
        }
    }

    // Re-arm for the next call: consume exactly what arrived (cumulative waits above saw all of it).
    for (uint32_t i = 0; i < n_in; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(arrival_idx + i)), 0u - num_waves);
    }
    noc_async_atomic_barrier();
}

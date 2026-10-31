// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm port core (TRANSPORT; one per (direction, link); RISCV_0 / NoC1). Adapted from
// ttnn/ttnn/operations/examples/fabric_ring_exchange/kernels/port.cpp (the measured relay transport), with the
// op's roots wired in: the roots of this port's link publish their record runs into its own slot and count
// SEM_OWN_STATS; the port notifies those roots (SEM_LANDED_<direction>) once per wave when every entry of the wave
// has landed here, and releases their next-wave publication (SEM_OWN_COUNTED) when it counts a wave.
//
// landing[w][q] (this core's shard of the persistent landing scratch; the same address on every chip) holds slot q of
// wave w. Every slot that lands on this port is one "entry" i with its own arrival counter (a global semaphore,
// arrival_addr[i]) counting waves.
//   ready fence    once per call: a header-only atomic inc of the ready counter of every port that writes into this
//                  chip through this port's direction; a sender waits for one ready per destination chip before its
//                  first send
//   sends (w, k)   item k of wave w once its source is in L1 (own: every root of the link published wave w; relay of
//                  entry j: arrival[j] >= w + 1), as ceil(bytes / payload) packets into the destination port's same
//                  landing slot, the last one fused with an increment of the destination entry's arrival counter
//                  (header-only for an empty segment)
//   drain          re-arm the counters by what was consumed (back to 0: trace-safe)
// Raw fabric API: the measured transport this op cites is this raw sender (op_design.md -> Helpers considered and
// rejected: the kernel_lib CCL helper is unmeasured and has no relay-with-arrival-counter item model).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef DRMS_ABLATE_EXCHANGE
#define DRMS_ABLATE_EXCHANGE 0
#endif

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
    constexpr uint32_t sem_counted_id = get_compile_time_arg_val(5);
    // Own segment of wave w+1 may go ahead of wave w's relays (true) or strictly (w, k) order (false).
    constexpr bool own_ahead = get_compile_time_arg_val(6) != 0;
    static_assert(num_headers >= 1 && num_headers <= 12, "packet header pool: <= 12 headers per RISC");

    size_t a = 0;
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_dest = get_arg_val<uint32_t>(a++);       // distinct destination chips (one header each)
    const uint32_t n_send = get_arg_val<uint32_t>(a++);       // send items per wave
    const uint32_t n_in = get_arg_val<uint32_t>(a++);         // entries landing on this port per wave
    const uint32_t n_roots = get_arg_val<uint32_t>(a++);      // roots publishing into my own slot (0 if I send nothing)
    const uint32_t n_ready_out = get_arg_val<uint32_t>(a++);  // ready packets this port sends
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

#if DRMS_ABLATE_EXCHANGE
    // Measurement-only (host DRMS_ABLATE_EXCHANGE=1): no connection, fence or sends. Every entry "lands" at once (its
    // arrival counter is bumped locally by every wave), the roots' publications are still counted (SEM_OWN_COUNTED)
    // and notified as usual. Output is wrong, timing is valid: t(op) - t(op, ablated) = the exposed transport.
    for (uint32_t i = 0; i < n_in; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(arrival_idx + i)), num_waves);
    }
    noc_async_atomic_barrier();
    {
        const uint32_t count_waves = n_send > 0 ? num_waves : 0;
        uint32_t sw = 0;
        while (sw < count_waves || notified < notify_total) {
            invalidate_l1_cache();
            if (sw < count_waves && *own_stats >= (sw + 1) * n_roots) {
                if (sw + 1 < num_waves) {
                    for (uint32_t k = 0; k < n_counted; ++k) {
                        const uint32_t r = get_arg_val<uint32_t>(roots_idx + k);
                        noc_semaphore_inc(get_noc_addr(r >> 16, r & 0xFFFF, counted_sem_addr), 1);
                    }
                }
                ++sw;
            }
            poll_notify();
        }
    }
    if (false) {
#else
    if (has_conn) {
#endif
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
        // Two send cursors: the own segment (item 0) by wave, and the relays (items 1..n_send-1) in (w, k) order. Each
        // destination entry has its own arrival counter, so the two streams are independent. own_ahead: the own segment
        // of wave w+1 need not queue behind wave w's relays (lower wave first when both are ready); otherwise the
        // strict (w, k) order of the relay transport (host knob PORT_OWN_AHEAD).
        const uint32_t send_waves = n_send > 0 ? num_waves : 0;
        const uint32_t relay_waves = n_send > 1 ? num_waves : 0;
        uint32_t ow = 0;          // own cursor: next wave of item 0
        uint32_t rw = 0, rk = 1;  // relay cursor: (wave, item)
        {
            DeviceZoneScopedN("prt_exchange");
            while (ow < send_waves || rw < relay_waves || notified < notify_total) {
                invalidate_l1_cache();
                const bool own_ready =
                    ow < send_waves && (own_ahead || relay_waves == 0 || ow <= rw) && *own_stats >= (ow + 1) * n_roots;
                bool relay_ready = false;
                if (rw < relay_waves && (own_ahead || rw < ow)) {
                    const uint32_t wait = get_arg_val<uint32_t>(send_idx + rk * SEND_WORDS + 3);
                    relay_ready = *arrived(wait - 1) >= rw + 1;
                }
                if (own_ready && !(relay_ready && rw < ow)) {
                    if (ow + 1 < num_waves) {
                        // Wave ow counted: release the roots' wave-(ow+1) increments (SEM_OWN_COUNTED on the host).
                        for (uint32_t k = 0; k < n_counted; ++k) {
                            const uint32_t r = get_arg_val<uint32_t>(roots_idx + k);
                            noc_semaphore_inc(get_noc_addr(r >> 16, r & 0xFFFF, counted_sem_addr), 1);
                        }
                    }
                    const uint32_t q = get_arg_val<uint32_t>(send_idx);
                    DeviceZoneScopedN("prt_send");
                    send_item(0, landing + (ow * G + q) * seg_stride, get_arg_val<uint32_t>(seg_idx + ow));
                    ++ow;
                } else if (relay_ready) {
                    const uint32_t q = get_arg_val<uint32_t>(send_idx + rk * SEND_WORDS);
                    DeviceZoneScopedN("prt_send");
                    send_item(rk, landing + (rw * G + q) * seg_stride, get_arg_val<uint32_t>(seg_idx + rw));
                    if (++rk == n_send) {
                        rk = 1;
                        ++rw;
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

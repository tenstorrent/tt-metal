// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — port core sender (RISCV_0, NoC1); one port per (direction, link) per chip.
// Adapted from the fabric_all_gather example's sender (ready fence, fused write + arrival increment, re-arm).
//
// The board (persistent L1, same address on every chip) holds one region per exchange round, each G slots of
// `slot_bytes`: slot g = the fp32 partial vectors of group position g for this link's tile rows of that round.
// Per call:
//   1. credit (once): wait for my peer's ready credit (its board is free) and consume it;
//   then per round r (disjoint board regions, cumulative counters):
//   2. own slot: once every local worker of the link has written round r (sem_port_local), send it to the peer;
//   3. relays: entry k (k >= 1) once the arrival counter says upstream's k-th slot of round r has landed; every
//      slot's last packet is a fused write + increment of the peer port's arrival counter;
//   4. when round r's region is complete: tell the link's leaders (sem_board_ready);
//   finally re-arm the counters and hand the credit back for the next call: a receiving port, once every leader of
//   the link has finished reading the board (sem_gather_done) and its arrival counter is re-armed, tells the opposite
//   port of this chip (sem_rx_free), which owns the Fabric connection toward the sender and returns its ready.
// The ready credit (global semaphore, 1 at rest) replaces a start-of-call fence: the next call's sends need no
// round trip to the peer, so the exchange starts as soon as the own slot is written.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

constexpr uint32_t sem_port_local_id = 0;
constexpr uint32_t sem_board_ready_id = 1;
constexpr uint32_t sem_port_local_odd_id = 3;
constexpr uint32_t sem_board_ready_odd_id = 4;
constexpr uint32_t sem_gather_done_id = 7;  // leaders of the link: board fully read this call
constexpr uint32_t sem_rx_free_id = 8;      // the opposite port: my board (it receives into) is free

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

void kernel_main() {
    constexpr uint32_t num_headers = get_compile_time_arg_val(0);  // HEADER_RING

    size_t a = 0;
    const uint32_t board_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t max_payload = get_arg_val<uint32_t>(a++);
    const uint32_t n_local = get_arg_val<uint32_t>(a++);  // local workers that write my board's own slot
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);    // slots arriving from upstream per round
    const uint32_t num_rounds = get_arg_val<uint32_t>(a++);   // one board region per round, counters cumulative
    const uint32_t round_bytes = get_arg_val<uint32_t>(a++);  // round r's region: board_addr + r * round_bytes
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);   // return the peer's credit at the end of the call
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);      // the opposite-direction port (same core on every chip)
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t n_leaders = get_arg_val<uint32_t>(a++);
    const uint32_t leaders_idx = a;
    a += n_leaders;
    const uint32_t n_sends = get_arg_val<uint32_t>(a++);
    const uint32_t sends_idx = a;
    a += n_sends;

    // Local and board_ready counters have several sources, so they are kept per round parity: a worker publishes
    // round r+2 only after its leader gathered round r (which needed this port's round-r signal), so parity r & 1
    // separates rounds. Arrivals have one in-order source and stay a single cumulative counter.
    volatile tt_l1_ptr uint32_t* local_par[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_port_local_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_port_local_odd_id))};
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    // Counters (local, arrival, leaders' board_ready) are cumulative over rounds and re-armed at the end.
    const uint32_t board_ready_par[2] = {get_semaphore(sem_board_ready_id), get_semaphore(sem_board_ready_odd_id)};
    auto signal_leaders = [&](uint32_t r) {
        // round r's board region is complete: own slot written by every local worker, every upstream slot landed
        MaybeDeviceZoneScope("P_sigwait");
        noc_semaphore_wait_min(local_par[r & 1], n_local * (r / 2 + 1));
        if (expect_in > 0) {
            noc_semaphore_wait_min(arrived, expect_in * (r + 1));
        }
        for (uint32_t i = 0; i < n_leaders; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(leaders_idx + i);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, board_ready_par[r & 1]), 1);
        }
    };

    // End of call (receiving port): once every leader of the link has read the board, re-arm the arrival counter
    // (every increment of this call has been waited for) and tell the opposite port, which returns the credit.
    auto release_board = [&]() {
        if (expect_in > 0) {
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather_done_id)), n_leaders);
            noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in * num_rounds);
            noc_async_atomic_barrier();
            noc_semaphore_inc(get_noc_addr(ready_x, ready_y, get_semaphore(sem_rx_free_id)), 1);
            noc_async_atomic_barrier();
        }
    };

    if (n_sends > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        // Credit: the peer returned it at the end of its previous call, once that call had finished reading the
        // board (or it is the initial credit). Rounds of one call use disjoint board regions: no fence between them.
        if (n_sends > 0) {
            MaybeDeviceZoneScope("P_fence");
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        uint32_t h = 0, unflushed = 0;
        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t round_addr = board_addr + r * round_bytes;
            for (uint32_t k = 0; k < n_sends; ++k) {
                const uint32_t slot = get_arg_val<uint32_t>(sends_idx + k);
                if (k == 0) {
                    MaybeDeviceZoneScope("P_ownwait");
                    // own slot: every local worker's round-r partials
                    noc_semaphore_wait_min(local_par[r & 1], n_local * (r / 2 + 1));
                } else {
                    MaybeDeviceZoneScope("P_relaywait");
                    // upstream's entry k-1 of round r (= my entry k) has landed; arrivals are in order per link
                    noc_semaphore_wait_min(arrived, expect_in * r + k);
                }
                const uint32_t slot_addr = round_addr + slot * slot_bytes;
                for (uint32_t off = 0; off < slot_bytes; off += max_payload) {
                    const uint32_t bytes = (slot_bytes - off) < max_payload ? (slot_bytes - off) : max_payload;
                    const uint64_t dst = get_noc_addr(peer_x, peer_y, slot_addr + off);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    if (++h == num_headers) {
                        h = 0;
                    }
                    if (off + bytes == slot_bytes) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(slot_addr + off, bytes, reinterpret_cast<uint32_t>(hdr));
                    if (++unflushed == num_headers) {
                        noc_async_writes_flushed();  // headers can be reused
                        unflushed = 0;
                    }
                }
            }
            signal_leaders(r);
        }
        noc_async_writes_flushed();
        release_board();
        if (send_ready) {
            // my opposite port's board is free and re-armed: the peer that sends into it may start its next call
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rx_free_id)), 1);
            hdrs[h]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[h]), sizeof(PACKET_HEADER_TYPE));
        }
        conn.close();
    } else {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            signal_leaders(r);
        }
        release_board();
    }
    noc_semaphore_set(local_par[0], 0);
    noc_semaphore_set(local_par[1], 0);
}

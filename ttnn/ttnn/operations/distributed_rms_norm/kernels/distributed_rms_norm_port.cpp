// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — port core (BRISC, NoC1): exchange_round for one (link, direction).
// Adapted from fabric_all_gather's sender: ready fence, per-entry arrival counter, one-hop relay.
//   1. (connection) tell the neighbour port that writes into this device that it may (ready), and before sending
//      anything wait for my own ready (the fence between calls).
//   2. wait until every worker of my link has published its records into my slot p (local arrivals).
//   3. send my entries over Fabric, in order: entry 0 is my own slot, entry k >= 1 is a relay of the upstream's
//      entry k-1 (sent once my arrival counter says it has landed). Each entry lands at the same scratch offset of
//      the next device's same-direction port; its last packet carries a fused increment of that port's arrival.
//   4. wait until every inbound entry has landed, multicast +1 to the workers' stats-ready, re-arm the counter.
//
// Raw-API justification: Fabric transport — mcast_pipe is on-chip only and the CCL helper library is
// unmeasured / 1D-typed (op_design.md "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

// One hop: 2D fabrics route by destination fabric node, 1D fabrics by hop count.
inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

void kernel_main() {
    constexpr uint32_t sem_local_arrivals = get_compile_time_arg_val(0);
    constexpr uint32_t sem_stats_ready = get_compile_time_arg_val(1);
    constexpr uint32_t num_headers = get_compile_time_arg_val(2);

    size_t a = 0;
    const uint32_t scratch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t entry_bytes = get_arg_val<uint32_t>(a++);   // recs_link * 128
    const uint32_t packet_bytes = get_arg_val<uint32_t>(a++);  // max payload, whole records
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // global; the downstream port's is at the same address
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);    // global
    const uint32_t expect_local = get_arg_val<uint32_t>(a++);  // workers that publish into my slot
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);     // entries the upstream port sends me
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);  // the neighbour's opposite-direction port
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t has_conn = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);  // the neighbour's same-direction port
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t mc_x0 = get_arg_val<uint32_t>(a++);  // worker rectangle (NoC coords, start <= end)
    const uint32_t mc_y0 = get_arg_val<uint32_t>(a++);
    const uint32_t mc_x1 = get_arg_val<uint32_t>(a++);
    const uint32_t mc_y1 = get_arg_val<uint32_t>(a++);
    const uint32_t mc_num_dests = get_arg_val<uint32_t>(a++);
    const uint32_t num_entries = get_arg_val<uint32_t>(a++);
    const uint32_t entries_idx = a;
    a += num_entries;

    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    if (has_conn) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // appended last
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        // Fence: this program running means everything earlier on this device is done, so the neighbour may
        // now write into my scratch.
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_entries > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();

            volatile tt_l1_ptr uint32_t* local =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local_arrivals));
            noc_semaphore_wait_min(local, expect_local);

            const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
            uint32_t h = 0;
            for (uint32_t k = 0; k < num_entries; ++k) {
                if (k > 0) {
                    noc_semaphore_wait_min(arrived, k);  // upstream's entry k-1 has landed in my scratch
                }
                const uint32_t slot = get_arg_val<uint32_t>(entries_idx + k);
                const uint32_t base = scratch_addr + slot * slot_bytes;
                for (uint32_t off = 0; off < entry_bytes; off += packet_bytes) {
                    const uint32_t bytes = (entry_bytes - off) < packet_bytes ? (entry_bytes - off) : packet_bytes;
                    const uint64_t dst = get_noc_addr(peer_x, peer_y, base + off);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    if (off + bytes >= entry_bytes) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(base + off, bytes, reinterpret_cast<uint32_t>(hdr));
                    if (++h == num_headers) {
                        noc_async_writes_flushed();  // headers of the in-flight packets have left L1
                        h = 0;
                    }
                }
            }
            noc_async_writes_flushed();
        }
        conn.close();
    }

    // Every inbound entry has landed (relays already needed part of it); my own slot was waited for above.
    if (expect_in > 0) {
        noc_semaphore_wait_min(arrived, expect_in);
    }
    // stats-ready: one increment per port; workers wait for 2L of them.
    const uint32_t stats_ready = get_semaphore(sem_stats_ready);
    const uint64_t mc_addr = noc_index == 0 ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, stats_ready)
                                            : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, stats_ready);
    noc_semaphore_inc_multicast(mc_addr, 1, mc_num_dests);
    // re-arm for the next call
    if (expect_in > 0) {
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
    }
    noc_async_atomic_barrier();
}

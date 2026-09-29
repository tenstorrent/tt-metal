// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — port_fwd (RISCV_0 of the lane's port core). Owns the Fabric connection
// toward p+1 (absent on a line's last device, where this kernel is idle).
//
// forward_partial_block, served per reducer: reducer r's non-tail blocks leave in r's block order
// (s = fwd_seq[r] = chunks of reducer r forwarded so far — also r's staging ordinal here and its
// receive ordinal on p+1, since a chunk p forwards is exactly a chunk p+1 receives), but the
// reducers are served round-robin by readiness, never in one global lane order: with a rotating
// ring head a global order would make every hop wait on the previous device's previous chunk.
//   ready: staged[r] > s (reducer r staged it) and gsem_partial_credit[r] > s (downstream reducer
//   r granted a landing slot); send packets_per_chunk packets (the last one a fused
//   write+atomic-inc) into downstream reducer r's landing slot (s mod RECV_DEPTH), then free the
//   staging slot (egress_credit of local reducer r).
// Tail blocks (per-block role, high_bw_all_reduce_roles.hpp) are skipped: their sum is final here.
// Every wait loop also forwards final-landing credits per reducer (final_freed[r] + FD, in r's
// block units, as gsem_final_credit[r], coalesced CREDIT_BATCH blocks per packet) to p+1's port_bwd —
// the deadlock-freedom rule of the design.
// No kernel_lib Fabric helper exists (recorded gap in op_design.md): raw EDM sender API.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "high_bw_all_reduce_roles.hpp"
#include "high_bw_all_reduce_route.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace tt::tt_fabric;

void kernel_main() {
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t packet_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(3);
    constexpr uint32_t recv_depth = get_compile_time_arg_val(4);
    constexpr uint32_t final_depth = get_compile_time_arg_val(5);
    constexpr bool has_next = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t egress_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t CONTROL_WORD_STRIDE = get_compile_time_arg_val(9);  // bytes between control words
    // Data packet headers in flight per connection (host MAX_DATA_HEADERS; one more header is
    // used for credits). Each header is rewritten only after its previous send has been flushed,
    // so packets of a chunk issue back to back without a per-packet flush.
    constexpr uint32_t MAX_DATA_HEADERS = get_compile_time_arg_val(10);
    constexpr uint32_t pos = get_compile_time_arg_val(11);
    constexpr uint32_t group_size = get_compile_time_arg_val(12);
    constexpr uint32_t num_slices = get_compile_time_arg_val(13);
    constexpr uint32_t MAX_REDUCERS = get_compile_time_arg_val(14);  // host REDUCERS_PER_LANE (W <= it)
    // Final-landing credits are coalesced: one credit packet per CREDIT_BATCH freed blocks of a
    // reducer (or the reducer's last credit). Every credit is a whole EDM packet slot, so this is the
    // knob that cuts the credit stream's share of the link (host credit_batch, from CREDIT_BATCH_CHUNKS; the host keeps
    // final_depth >= CREDIT_BATCH, which is what keeps a withheld credit from deadlocking).
    constexpr uint32_t CREDIT_BATCH = get_compile_time_arg_val(15);
    static_assert(CREDIT_BATCH >= 1 && final_depth >= CREDIT_BATCH, "final_depth must cover the credit batch");
    constexpr uint32_t packets_per_chunk = chunk_tiles / packet_tiles;
    constexpr uint32_t packet_bytes = packet_tiles * tile_bytes;
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    constexpr uint32_t num_data_headers = packets_per_chunk < MAX_DATA_HEADERS ? packets_per_chunk : MAX_DATA_HEADERS;

    if constexpr (!has_next) {
        return;  // no downstream neighbour: every chunk is a tail chunk, nothing to forward
    }

    size_t a = 0;
    const uint32_t num_chunks = get_arg_val<uint32_t>(a++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(a++);
    const uint32_t port_x = get_arg_val<uint32_t>(a++);  // this (forward-port) core
    const uint32_t port_y = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_x = get_arg_val<uint32_t>(a++);  // the lane's port_bwd core (== this one if co-located)
    const uint32_t bwd_y = get_arg_val<uint32_t>(a++);
    const uint32_t staging_addr = get_arg_val<uint32_t>(a++);
    const uint32_t staged_arr = get_arg_val<uint32_t>(a++);
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gsem_partial_arrival = get_arg_val<uint32_t>(a++);
    const uint32_t final_freed_arr = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t reducer_xy_idx = a;
    a += 2 * num_reducers;
    const uint32_t partial_credit_idx = a;  // per-reducer gsem_partial_credit[r] addresses
    a += num_reducers;
    const uint32_t final_credit_idx = a;  // per-reducer gsem_final_credit[r] (on p+1's port)
    a += num_reducers;

    auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_hdrs[num_data_headers];
    uint32_t hdr_idx = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* credit_hdr;
    {
        MaybeDeviceZoneScope("fwd_setup");  // headers + connection open + go
        for (uint32_t h = 0; h < num_data_headers; ++h) {
            data_hdrs[h] = PacketHeaderPool::allocate_header();
            route_to_neighbor(data_hdrs[h], dst_chip_id, dst_mesh_id);
        }
        credit_hdr = PacketHeaderPool::allocate_header();
        route_to_neighbor(credit_hdr, dst_chip_id, dst_mesh_id);
        connection.open();

        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id)), 1);
    }

    const uint32_t egress_sem_addr = get_semaphore(egress_sem_id);

    const ChainRoles<pos, group_size, num_slices> roles(num_reducers);
    uint32_t nb[MAX_REDUCERS];                  // blocks of reducer r in this lane
    uint32_t next_k[MAX_REDUCERS];              // reducer r's next non-tail block (nb[r] once none remain)
    uint32_t fwd_seq[MAX_REDUCERS];             // reducer r's blocks forwarded so far
    uint32_t final_credits_sent[MAX_REDUCERS];  // final-landing credits of reducer r sent to p+1
    auto next_forward_block = [&](uint32_t r, uint32_t k) {
        while (k < nb[r] && roles.is_tail(r, k, nb[r])) {
            ++k;
        }
        return k;
    };

    uint64_t final_credit_noc[MAX_REDUCERS];  // p+1's gsem_final_credit[r]
    for (uint32_t r = 0; r < num_reducers; ++r) {
        // The neighbour's port_bwd core is the same logical (and virtual) core as ours.
        final_credit_noc[r] = get_noc_addr(bwd_x, bwd_y, get_arg_val<uint32_t>(final_credit_idx + r));
    }
    // O(1) per call: checks one reducer, round-robin across calls. It runs once per data packet,
    // so every reducer is still polled several times per chunk without a W-wide scan per packet.
    uint32_t credit_r = 0;
    MaybePerfAccum(acc_credit);     // final-credit packets sent toward p+1 (slot wait + flush)
    MaybePerfAccum(acc_send);       // one chunk's packets issued + flushed (includes slot waits)
    MaybePerfAccum(acc_slot_wait);  // EDM sender slot not free (link / router back-pressure)
    auto forward_final_credits = [&]() {
        const uint32_t r = credit_r;
        credit_r = (credit_r + 1 == num_reducers) ? 0 : credit_r + 1;
        uint32_t target =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_freed_arr + r * CONTROL_WORD_STRIDE)[0] + final_depth;
        if (target > nb[r]) {
            target = nb[r];
        }
        if (target > final_credits_sent[r] && (target - final_credits_sent[r] >= CREDIT_BATCH || target == nb[r])) {
            MaybePerfBegin(acc_credit);
            credit_hdr->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{final_credit_noc[r], target - final_credits_sent[r]});
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(credit_hdr), sizeof(PACKET_HEADER_TYPE));
            final_credits_sent[r] = target;
            MaybePerfEnd(acc_credit);
        }
    };

    uint32_t pending = 0;
    for (uint32_t r = 0; r < num_reducers; ++r) {
        nb[r] = roles.blocks_of(r, num_chunks);
        fwd_seq[r] = 0;
        final_credits_sent[r] = 0;
        for (uint32_t k = 0; k < nb[r]; ++k) {
            pending += roles.is_tail(r, k, nb[r]) ? 0 : 1;
        }
        next_k[r] = next_forward_block(r, 0);
    }

    uint32_t r = 0;
    {
        MaybeDeviceZoneScope("fwd_main");
        while (pending > 0) {
            for (uint32_t scan = 0; scan < num_reducers; ++scan, r = (r + 1 == num_reducers) ? 0 : r + 1) {
                if (next_k[r] >= nb[r]) {
                    continue;
                }
                const uint32_t k = fwd_seq[r];
                const uint32_t staged =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staged_arr + r * CONTROL_WORD_STRIDE)[0];
                const uint32_t credit =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(partial_credit_idx + r))[0];
                if (staged <= k || credit <= k) {
                    continue;
                }
                const uint32_t rx = get_arg_val<uint32_t>(reducer_xy_idx + 2 * r);
                const uint32_t ry = get_arg_val<uint32_t>(reducer_xy_idx + 2 * r + 1);

                // forward_partial_block
                const uint32_t src = staging_addr + ((k % staging_depth) * num_reducers + r) * chunk_bytes;
                const uint32_t dst = landing_addr + (k % recv_depth) * chunk_bytes;
                const uint64_t arrival_noc = get_noc_addr(rx, ry, gsem_partial_arrival);
                MaybePerfBegin(acc_send);
                // Plain writes for all but the last packet; the last packet carries the chunk's arrival
                // increment (fused, flushed) so the receiver sees one increment per landed chunk.
                for (uint32_t pkt = 0; pkt < packets_per_chunk; ++pkt) {
                    if (hdr_idx == num_data_headers) {
                        noc_async_writes_flushed();  // every data header's previous send has left L1
                        hdr_idx = 0;
                    }
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_hdr = data_hdrs[hdr_idx++];
                    const uint64_t pkt_dst = get_noc_addr(rx, ry, dst + pkt * packet_bytes);
                    if (pkt + 1 < packets_per_chunk) {
                        data_hdr->to_noc_unicast_write(NocUnicastCommandHeader{pkt_dst}, packet_bytes);
                    } else {
                        data_hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{pkt_dst, arrival_noc, 1, true}, packet_bytes);
                    }
                    forward_final_credits();  // keep downstream final grants flowing while we stream
                    MaybePerfBegin(acc_slot_wait);
                    connection.wait_for_empty_write_slot();
                    MaybePerfEnd(acc_slot_wait);
                    connection.send_current_slot_non_blocking(
                        src + pkt * packet_bytes, packet_bytes, reinterpret_cast<uint32_t>(data_hdr));
                }
                // Flush the chunk's payload reads before the staging slot may be reused by reducer r.
                noc_async_writes_flushed();
                MaybePerfEnd(acc_send);
                hdr_idx = num_data_headers;
                noc_semaphore_inc(get_noc_addr(rx, ry, egress_sem_addr), 1);

                ++fwd_seq[r];
                next_k[r] = next_forward_block(r, next_k[r] + 1);
                --pending;
                r = (r + 1 == num_reducers) ? 0 : r + 1;  // round-robin: resume after the served reducer
                break;
            }
            for (uint32_t i = 0; i < num_reducers; ++i) {  // full sweep between chunks / while idle
                forward_final_credits();
            }
        }
    }

    MaybeDeviceZoneScope("fwd_teardown");
    for (uint32_t rr = 0; rr < num_reducers; ++rr) {
        while (final_credits_sent[rr] < nb[rr]) {
            forward_final_credits();
        }
    }
    // End of invocation: remove exactly the credits this invocation consumed per reducer. Each
    // forward waited credit > s, and downstream grants exactly fwd_seq[r] in total, so every
    // credit has landed.
    for (uint32_t rr = 0; rr < num_reducers; ++rr) {
        if (fwd_seq[rr] > 0) {
            noc_semaphore_inc(
                get_noc_addr(port_x, port_y, get_arg_val<uint32_t>(partial_credit_idx + rr)), 0u - fwd_seq[rr]);
        }
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    connection.close();
    MaybePerfReport("fwd_send", acc_send);
    MaybePerfReport("fwd_slot_wait", acc_slot_wait);
    MaybePerfReport("fwd_credit", acc_credit);
}

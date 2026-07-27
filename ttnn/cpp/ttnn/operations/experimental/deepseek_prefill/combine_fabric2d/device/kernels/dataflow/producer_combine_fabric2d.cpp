// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer kernel (writer RISC). Shares its core with the receiver kernel and owns the ONE fabric
// sender connection that its eth channel allows (the L1 conn table is indexed by eth channel and the
// EDM stores a single worker_xy per channel, so a second core on the same channel would just hang).
// It therefore does two jobs:
//   1. send payload tokens to the peer worker on the neighbor chip, gated by `write_up_to`;
//   2. forward the co-located receiver's credit returns, since the receiver has no connection of its
//      own (being a fabric DESTINATION needs none — the peer's eth RISC writes into our L1 unasked).
//
// Credits go first, unconditionally, every iteration. That is what keeps the ring deadlock-free: a
// producer blocked on its own `write_up_to` still forwards the credits the peer producer is waiting
// for. The loop also cannot exit at sent == num_tokens — it must have forwarded all num_tokens
// credits too, or the peer hangs.
//
// All three counters are single-writer monotonic, so no atomics are needed on the read side: each
// reader keeps its own local count and works on the difference.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "fabric/fabric_edm_packet_header.hpp"

void kernel_main() {
    constexpr uint32_t num_tokens = get_compile_time_arg_val(0);
    constexpr uint32_t num_slots = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t peer_chip_id = get_compile_time_arg_val(3);
    constexpr uint32_t peer_mesh_id = get_compile_time_arg_val(4);
    constexpr uint32_t peer_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t peer_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t prod_buf_addr = get_compile_time_arg_val(7);
    constexpr uint32_t recv_buf_addr = get_compile_time_arg_val(8);  // same address on every chip
    constexpr uint32_t pkt_hdr_payload_addr = get_compile_time_arg_val(9);
    constexpr uint32_t pkt_hdr_credit_addr = get_compile_time_arg_val(10);
    constexpr uint32_t write_up_to_addr = get_compile_time_arg_val(11);
    constexpr uint32_t data_ready_addr = get_compile_time_arg_val(12);
    constexpr uint32_t credits_to_return_addr = get_compile_time_arg_val(13);

    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& sender = fabric_connections.get(0).sender;

    volatile PACKET_HEADER_TYPE* pkt_hdr_payload = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_payload_addr);
    volatile PACKET_HEADER_TYPE* pkt_hdr_credit = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_credit_addr);
    // Written only by the remote eth RISC (credit packets); we just read it.
    volatile tt_l1_ptr uint32_t* write_up_to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_up_to_addr);
    // Written only by the receiver kernel on this core; we just read it.
    volatile tt_l1_ptr uint32_t* credits_to_return =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credits_to_return_addr);

    // Remote targets on the peer worker core: its data_ready (bumped by the fused payload write) and
    // its write_up_to (bumped by our credit returns).
    const uint64_t peer_data_ready_noc = get_noc_addr(peer_noc_x, peer_noc_y, data_ready_addr);
    const uint64_t peer_write_up_to_noc = get_noc_addr(peer_noc_x, peer_noc_y, write_up_to_addr);

    uint32_t sent = 0;          // tokens handed to the fabric
    uint32_t credits_sent = 0;  // credits forwarded on the receiver's behalf

    {
        // One zone per token, plus this outer one. The profiler L1 buffer holds 250 optional markers
        // per RISC, so at num_tokens=100 a second per-token zone would overflow and silently drop.
        DeviceZoneScopedN("PRODUCER_LOOP");
        while (sent < num_tokens || credits_sent < num_tokens) {
            // ---- 1. Credits first, always. Never gated on anything, so a stalled producer cannot
            // ---- stall its peer. One packet carries the whole pending batch as its inc value.
            invalidate_l1_cache();
            const uint32_t returnable = *credits_to_return;
            if (returnable > credits_sent) {
                const uint32_t batch = returnable - credits_sent;
                pkt_hdr_credit->to_noc_unicast_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_write_up_to_noc, batch, /*flush=*/false});
                fabric_set_unicast_route(
                    (volatile tt::tt_fabric::HybridMeshPacketHeader*)pkt_hdr_credit, peer_chip_id, peer_mesh_id);
                sender.wait_for_empty_write_slot();
                sender.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr_credit, sizeof(PACKET_HEADER_TYPE));
                credits_sent += batch;
            }

            // ---- 2. Then a token, if the peer's ring has room for it.
            if (sent < num_tokens) {
                invalidate_l1_cache();
                if (sent < *write_up_to) {
                    DeviceZoneScopedN("PRODUCER_SEND");
                    const uint32_t slot = sent % num_slots;
                    const uint64_t dst_noc =
                        get_noc_addr(peer_noc_x, peer_noc_y, recv_buf_addr + slot * chunk_size_bytes);
                    fabric_set_unicast_route(
                        (volatile tt::tt_fabric::HybridMeshPacketHeader*)pkt_hdr_payload, peer_chip_id, peer_mesh_id);
                    pkt_hdr_payload->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            dst_noc, peer_data_ready_noc, /*val=*/1, /*flush=*/true},
                        chunk_size_bytes);
                    sender.wait_for_empty_write_slot();
                    sender.send_payload_without_header_non_blocking_from_address(
                        prod_buf_addr + slot * chunk_size_bytes, chunk_size_bytes);
                    // Blocking flush: the header is reused next iteration, so the send must have
                    // drained out of L1 before we overwrite it.
                    sender.send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_hdr_payload, sizeof(PACKET_HEADER_TYPE));
                    sent++;
                }
            }
        }
    }

    noc_async_writes_flushed();
    fabric_connections.close();
}

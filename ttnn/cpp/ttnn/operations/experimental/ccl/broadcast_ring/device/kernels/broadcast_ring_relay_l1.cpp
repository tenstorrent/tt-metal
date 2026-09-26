// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// broadcast_ring L1-relay kernel: single sender, bidirectional around the ring (FABRIC_1D / _RING).
//
// Removes the per-hop DRAM read of the default relay: the upstream fabric-writes each chunk straight into
// this device's L1 recv buffer; this device writes it to its own DRAM output and forwards it from L1. A
// bounded recv buffer (num_slots) needs a backward slot-free credit: after consuming a slot a device
// increments its upstream's cred sem, and a sender/relay waits for that credit before refilling the
// downstream slot. data_ready flows with the data (forward on the fwd arc, backward on the bwd arc); the
// credit flows the opposite way. Runs per orthogonal-axis line; payload split across links.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>

// ---- compile-time args ----
constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t sender_ring_index = get_compile_time_arg_val(1);
constexpr uint32_t my_ring_index = get_compile_time_arg_val(2);
constexpr uint32_t num_tiles = get_compile_time_arg_val(3);  // per-device shard tiles (unused; range is RT)
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);  // tiles per chunk == slot pages
constexpr uint32_t cb_id = get_compile_time_arg_val(6);                 // recv buffer (num_slots slots)
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t fwd_route_arg0 = get_compile_time_arg_val(8);   // to idx+1 dst_mesh_id
constexpr uint32_t fwd_route_arg1 = get_compile_time_arg_val(9);   // to idx+1 dst_chip_id
constexpr uint32_t bwd_route_arg0 = get_compile_time_arg_val(10);  // to idx-1 dst_mesh_id
constexpr uint32_t bwd_route_arg1 = get_compile_time_arg_val(11);  // to idx-1 dst_chip_id
constexpr uint32_t num_slots = get_compile_time_arg_val(12);       // recv-buffer depth (== credit window)
constexpr uint32_t tensor_args_base = 13;

// Arc roles: the sender sends both ways; the ring splits into a forward arc (HF hops) and backward arc (HB
// hops). A non-sender receives from its one upstream and forwards away from the sender until its arc end.
// L1 credit roles: after consuming a slot a device credits its upstream one hop back — fwd-arc devices
// credit idx-1 over the backward conn, bwd-arc devices credit idx+1 over the forward conn.
constexpr uint32_t fwd_hops = (my_ring_index + ring_size - sender_ring_index) % ring_size;
constexpr uint32_t bwd_hops = (ring_size - fwd_hops) % ring_size;
constexpr uint32_t HF = ring_size / 2;
constexpr uint32_t HB = (ring_size - 1) / 2;
constexpr bool is_sender = (fwd_hops == 0);
constexpr bool on_fwd_arc = !is_sender && (fwd_hops <= HF);
constexpr bool on_bwd_arc = !is_sender && !on_fwd_arc;
constexpr bool send_data_fwd = is_sender || (on_fwd_arc && fwd_hops < HF);
constexpr bool send_data_bwd = is_sender || (on_bwd_arc && bwd_hops < HB);
constexpr bool credit_via_forward = on_bwd_arc;   // inc idx+1.cred_bwd
constexpr bool credit_via_backward = on_fwd_arc;  // inc idx-1.cred_fwd
constexpr bool open_forward = send_data_fwd || credit_via_forward;
constexpr bool open_backward = send_data_bwd || credit_via_backward;
constexpr bool opens_any = open_forward || open_backward;

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);     // local input shard (valid on sender)
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);    // local output
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // data_ready (upstream increments)
    const uint32_t ds_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);   // same logical core on every device
    const uint32_t ds_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ds_data_ready_addr = get_arg_val<uint32_t>(arg_idx++);  // downstream data_ready (== recv_sem_addr)
    const uint32_t cred_fwd_addr = get_arg_val<uint32_t>(arg_idx++);       // my/neighbour cred_fwd (idx+1 frees)
    const uint32_t cred_bwd_addr = get_arg_val<uint32_t>(arg_idx++);       // my/neighbour cred_bwd (idx-1 frees)
    const uint32_t tile_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_count = get_arg_val<uint32_t>(arg_idx++);
    size_t fab_arg = arg_idx;

    constexpr auto in_args = TensorAccessorArgs<tensor_args_base>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in_addrgen = TensorAccessor(in_args, input_addr, page_size);
    const auto out_addrgen = TensorAccessor(out_args, output_addr, page_size);

    volatile tt_l1_ptr uint32_t* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);
    volatile tt_l1_ptr uint32_t* my_cred_fwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred_fwd_addr);
    volatile tt_l1_ptr uint32_t* my_cred_bwd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred_bwd_addr);

    constexpr ccl_routing_utils::line_unicast_route_info_t fwd_route = {
        .dst_mesh_id = static_cast<uint16_t>(fwd_route_arg0), .dst_chip_id = static_cast<uint16_t>(fwd_route_arg1)};
    constexpr ccl_routing_utils::line_unicast_route_info_t bwd_route = {
        .dst_mesh_id = static_cast<uint16_t>(bwd_route_arg0), .dst_chip_id = static_cast<uint16_t>(bwd_route_arg1)};

    auto fabric_connection = FabricConnectionManager::build_from_args(fab_arg);
    // Headers: payload + sem-inc per data direction; a credit-only direction needs just the sem-inc.
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd_data = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd_sem = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd_data = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd_sem = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_cred_fwd = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_cred_bwd = nullptr;
    if constexpr (opens_any) {
        CircularBuffer cb_hdr(reserved_packet_header_cb_id);
        auto next_hdr = [&]() {
            cb_hdr.reserve_back(1);
            auto* p = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_hdr.get_write_ptr());
            cb_hdr.push_back(1);
            return p;
        };
        if constexpr (send_data_fwd) {
            pkt_hdr_fwd_data = next_hdr();
            pkt_hdr_fwd_sem = next_hdr();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fwd_data, fwd_route);
        }
        if constexpr (send_data_bwd) {
            pkt_hdr_bwd_data = next_hdr();
            pkt_hdr_bwd_sem = next_hdr();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_bwd_data, bwd_route);
        }
        if constexpr (credit_via_forward) {
            pkt_hdr_cred_fwd = next_hdr();
        }
        if constexpr (credit_via_backward) {
            pkt_hdr_cred_bwd = next_hdr();
        }
        fabric_connection.open();
    }
    auto* fwd_conn = open_forward ? &fabric_connection.get_forward_connection() : nullptr;
    auto* bwd_conn = open_backward ? &fabric_connection.get_backward_connection() : nullptr;

    (void)num_tiles;
    const uint32_t relay_base = get_write_ptr(cb_id);  // recv buffer base; homogeneous, so == downstream base
    const uint32_t slot_bytes = packet_size_in_pages * page_size;
    const uint64_t ds_data_ready_noc = safe_get_noc_addr(ds_sem_noc_x, ds_sem_noc_y, ds_data_ready_addr, 0);
    const uint64_t up_cred_fwd_noc = safe_get_noc_addr(ds_sem_noc_x, ds_sem_noc_y, cred_fwd_addr, 0);  // idx-1.cred_fwd
    const uint64_t up_cred_bwd_noc = safe_get_noc_addr(ds_sem_noc_x, ds_sem_noc_y, cred_bwd_addr, 0);  // idx+1.cred_bwd

    // Fabric-write a chunk from my slot into the downstream's same-index slot (2 tiles/packet + odd tail),
    // then bump the downstream data_ready.
    auto forward_data = [&](volatile PACKET_HEADER_TYPE* pkt_data,
                            volatile PACKET_HEADER_TYPE* pkt_sem,
                            tt::tt_fabric::WorkerToFabricEdmSender* conn,
                            const ccl_routing_utils::line_unicast_route_info_t& route,
                            uint32_t slot_addr,
                            uint32_t chunk_tiles) {
        const uint64_t ds_slot_noc = safe_get_noc_addr(ds_sem_noc_x, ds_sem_noc_y, slot_addr, 0);
        uint32_t t = 0;
        for (; t + 1 < chunk_tiles; t += 2) {
            fabric_write_unidir(ds_slot_noc + t * page_size, pkt_data, *conn, slot_addr + t * page_size, 2 * page_size);
        }
        if (t < chunk_tiles) {
            fabric_write_unidir(ds_slot_noc + t * page_size, pkt_data, *conn, slot_addr + t * page_size, page_size);
        }
        pkt_sem->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ds_data_ready_noc, static_cast<uint32_t>(1)});
        conn->wait_for_empty_write_slot();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_sem, route);
        conn->send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(pkt_sem), sizeof(PACKET_HEADER_TYPE));
    };

    // Increment the upstream's slot-free credit sem (no payload).
    auto send_credit = [&](volatile PACKET_HEADER_TYPE* pkt_sem,
                           tt::tt_fabric::WorkerToFabricEdmSender* conn,
                           const ccl_routing_utils::line_unicast_route_info_t& route,
                           uint64_t sem_noc) {
        pkt_sem->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc, static_cast<uint32_t>(1)});
        conn->wait_for_empty_write_slot();
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_sem, route);
        conn->send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(pkt_sem), sizeof(PACKET_HEADER_TYPE));
    };

    const uint32_t tile_end = tile_start + tile_count;
    uint32_t tiles_done = tile_start;
    uint32_t chunk = 0;
    while (tiles_done < tile_end) {
        const uint32_t chunk_tiles = std::min(tile_end - tiles_done, packet_size_in_pages);
        const uint32_t slot_addr = relay_base + (chunk % num_slots) * slot_bytes;

        // 1) Get this chunk into my slot; make sure the downstream slot I'll write is free (credit).
        if constexpr (is_sender) {
            if (chunk >= num_slots) {
                if constexpr (send_data_fwd) {
                    noc_semaphore_wait_min(my_cred_fwd, chunk - num_slots + 1);
                }
                if constexpr (send_data_bwd) {
                    noc_semaphore_wait_min(my_cred_bwd, chunk - num_slots + 1);
                }
            }
            uint32_t wr = slot_addr;
            for (uint32_t t = 0; t < chunk_tiles; ++t) {
                noc_async_read_page(tiles_done + t, in_addrgen, wr);
                wr += page_size;
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(data_ready, chunk + 1);  // upstream wrote this chunk into my slot
            if (chunk >= num_slots) {
                if constexpr (send_data_fwd) {
                    noc_semaphore_wait_min(my_cred_fwd, chunk - num_slots + 1);
                }
                if constexpr (send_data_bwd) {
                    noc_semaphore_wait_min(my_cred_bwd, chunk - num_slots + 1);
                }
            }
        }

        // 2) Persist to my DRAM output (every device keeps the shard).
        for (uint32_t t = 0; t < chunk_tiles; ++t) {
            const uint64_t dst = get_noc_addr(tiles_done + t, out_addrgen);
            noc_async_write(slot_addr + t * page_size, dst, page_size);
        }

        // 3) Forward the chunk from L1 to the downstream slot(s).
        if constexpr (send_data_fwd) {
            forward_data(pkt_hdr_fwd_data, pkt_hdr_fwd_sem, fwd_conn, fwd_route, slot_addr, chunk_tiles);
        }
        if constexpr (send_data_bwd) {
            forward_data(pkt_hdr_bwd_data, pkt_hdr_bwd_sem, bwd_conn, bwd_route, slot_addr, chunk_tiles);
        }

        // The output write and both forwards read this slot; wait before crediting (upstream may refill it).
        noc_async_write_barrier();

        // 4) Credit my upstream that this slot is consumed.
        if constexpr (credit_via_backward) {
            send_credit(pkt_hdr_cred_bwd, bwd_conn, bwd_route, up_cred_fwd_noc);  // inc idx-1.cred_fwd
        }
        if constexpr (credit_via_forward) {
            send_credit(pkt_hdr_cred_fwd, fwd_conn, fwd_route, up_cred_bwd_noc);  // inc idx+1.cred_bwd
        }

        tiles_done += chunk_tiles;
        ++chunk;
    }

    if constexpr (opens_any) {
        fabric_connection.close();
    }
}

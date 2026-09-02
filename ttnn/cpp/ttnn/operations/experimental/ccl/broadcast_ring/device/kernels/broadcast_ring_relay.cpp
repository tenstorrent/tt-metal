// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// broadcast_ring relay kernel (v1: single sender, one-way around the ring, FABRIC_1D / _RING).
//
// Data flow: the sender's shard is relayed shard -> +1 -> +2 -> ... around the ring. Each hop fabric-writes
// the chunk into the DOWNSTREAM device's OUTPUT tensor (same tile ids), then atomic-incs the downstream
// recv-semaphore. A receiver waits on its recv-sem (chunk landed in its output), then forwards from its own
// output to the next device. Runs per orthogonal (tp) line, so each tp row broadcasts its own heads.
//
// Fabric calls mirror ring_attention_all_gather_writer.cpp. Marked spots (addrgen build, packet-header CB,
// downstream sem noc addr) still need on-device confirmation.

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
constexpr uint32_t num_tiles = get_compile_time_arg_val(3);  // per-device shard tiles (== output tiles)
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);  // tiles per chunk
constexpr uint32_t cb_id = get_compile_time_arg_val(6);                 // staging CB
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(7);
constexpr uint32_t unicast_route_arg0 = get_compile_time_arg_val(8);  // forward-neighbour dst_mesh_id
constexpr uint32_t unicast_route_arg1 = get_compile_time_arg_val(9);  // forward-neighbour dst_chip_id
// Input/output TensorAccessorArgs follow (base index 10); output addrgen used for the fabric write target.
constexpr uint32_t tensor_args_base = 10;

constexpr uint32_t hops_from_sender = (my_ring_index + ring_size - sender_ring_index) % ring_size;
constexpr bool is_sender = (hops_from_sender == 0);
constexpr bool is_last = (hops_from_sender == ring_size - 1);
constexpr bool forwards = !is_last;

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);     // local input shard (valid on sender)
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);    // local output
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // my recv-sem (upstream increments)
    const uint32_t ds_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);   // downstream sem noc coords
    const uint32_t ds_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ds_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // downstream recv-sem L1 addr
    size_t fab_arg = arg_idx;                                       // remaining args -> fabric connection

    // Input + output addrgens. Input CT args start at tensor_args_base; output args follow immediately.
    // Same output spec on every device, so the output addrgen also names the downstream fabric-write target.
    constexpr auto in_args = TensorAccessorArgs<tensor_args_base>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in_addrgen = TensorAccessor(in_args, input_addr, page_size);
    const auto out_addrgen = TensorAccessor(out_args, output_addr, page_size);

    volatile tt_l1_ptr uint32_t* recv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);
    CircularBuffer cb(cb_id);

    // Fabric connection + packet headers (payload + sem-inc), only when we forward.
    auto fabric_connection = FabricConnectionManager::build_from_args(fab_arg);
    volatile PACKET_HEADER_TYPE* pkt_hdr = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_sem = nullptr;
    constexpr ccl_routing_utils::line_unicast_route_info_t route = {
        .dst_mesh_id = static_cast<uint16_t>(unicast_route_arg0),
        .dst_chip_id = static_cast<uint16_t>(unicast_route_arg1)};
    if constexpr (forwards) {
        CircularBuffer cb_hdr(reserved_packet_header_cb_id);
        cb_hdr.reserve_back(1);
        pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_hdr.get_write_ptr());
        cb_hdr.push_back(1);
        cb_hdr.reserve_back(1);
        pkt_hdr_sem = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_hdr.get_write_ptr());
        cb_hdr.push_back(1);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, route);
        fabric_connection.open();
    }
    auto* fwd_conn = forwards ? &fabric_connection.get_forward_connection() : nullptr;

    uint32_t tiles_done = 0;
    uint32_t chunk = 0;
    while (tiles_done < num_tiles) {
        const uint32_t chunk_tiles = std::min(num_tiles - tiles_done, packet_size_in_pages);

        // 1) Get this chunk into the CB. Sender reads its local input; receivers wait for upstream's fabric
        //    write to land it in their OUTPUT, then read it back to L1 so they can forward it.
        cb.reserve_back(chunk_tiles);
        uint32_t l1_wr = cb.get_write_ptr();
        if constexpr (is_sender) {
            for (uint32_t t = 0; t < chunk_tiles; ++t) {
                noc_async_read_page(tiles_done + t, in_addrgen, l1_wr);
                l1_wr += page_size;
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(recv_sem, chunk + 1);  // upstream wrote this chunk into our output
            for (uint32_t t = 0; t < chunk_tiles; ++t) {
                noc_async_read_page(tiles_done + t, out_addrgen, l1_wr);
                l1_wr += page_size;
            }
            noc_async_read_barrier();
        }
        cb.push_back(chunk_tiles);

        cb.wait_front(chunk_tiles);
        const uint32_t cb_rd = cb.get_read_ptr();

        // 2) Sender persists its own output locally (receivers already have it in output via the fabric write).
        if constexpr (is_sender) {
            for (uint32_t t = 0; t < chunk_tiles; ++t) {
                const uint64_t dst = get_noc_addr(tiles_done + t, out_addrgen);
                noc_async_write(cb_rd + t * page_size, dst, page_size);
            }
            noc_async_write_barrier();
        }

        // 3) Forward each tile of the chunk to the downstream device's OUTPUT, then bump its recv-sem.
        if constexpr (forwards) {
            for (uint32_t t = 0; t < chunk_tiles; ++t) {
                fabric_write_unidir(tiles_done + t, out_addrgen, pkt_hdr, *fwd_conn, cb_rd + t * page_size, page_size);
            }
            const uint64_t ds_sem_noc = safe_get_noc_addr(ds_sem_noc_x, ds_sem_noc_y, ds_sem_addr, 0);
            pkt_hdr_sem->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ds_sem_noc, static_cast<uint32_t>(1)});
            fwd_conn->wait_for_empty_write_slot();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem, route);
            fwd_conn->send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(pkt_hdr_sem), sizeof(PACKET_HEADER_TYPE));
        }

        cb.pop_front(chunk_tiles);
        tiles_done += chunk_tiles;
        ++chunk;
    }

    if constexpr (forwards) {
        fabric_connection.close();
    }
}

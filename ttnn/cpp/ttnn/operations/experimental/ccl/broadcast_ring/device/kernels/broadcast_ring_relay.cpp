// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// broadcast_ring relay kernel: single sender, bidirectional around the ring (FABRIC_1D / _RING).
//
// Sender sends both ways; the ring splits into a forward arc (ring_size/2 hops) and backward arc
// ((ring_size-1)/2 hops). Each hop fabric-writes the chunk into the downstream OUTPUT + atomic-incs its
// recv-sem; a receiver waits, reads its output back to L1, forwards onward. Runs per orthogonal-axis line;
// payload split across links (tile_start/tile_count). PERF TODO (OPTIMIZATION_NOTES.md): the output re-read
// is a per-hop DRAM round-trip an L1 relay would remove.

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
constexpr uint32_t fwd_route_arg0 = get_compile_time_arg_val(8);   // to idx+1 (forward neighbour) dst_mesh_id
constexpr uint32_t fwd_route_arg1 = get_compile_time_arg_val(9);   // to idx+1 dst_chip_id
constexpr uint32_t bwd_route_arg0 = get_compile_time_arg_val(10);  // to idx-1 (backward neighbour) dst_mesh_id
constexpr uint32_t bwd_route_arg1 = get_compile_time_arg_val(11);  // to idx-1 dst_chip_id
// Input/output TensorAccessorArgs follow (base index 12); output addrgen names the fabric-write target.
constexpr uint32_t tensor_args_base = 12;

// Bidirectional broadcast: the sender sends both ways; the ring splits into a forward arc (HF hops) and a
// backward arc (HB hops) so no device is more than ~ring_size/2 hops away. Each non-sender relays in the
// one direction pointing away from the sender, until the arc's far end.
constexpr uint32_t fwd_hops = (my_ring_index + ring_size - sender_ring_index) % ring_size;  // 0 = sender
constexpr uint32_t bwd_hops = (ring_size - fwd_hops) % ring_size;                           // 0 = sender
constexpr uint32_t HF = ring_size / 2;        // forward-arc length (ceil((P-1)/2) for even P)
constexpr uint32_t HB = (ring_size - 1) / 2;  // backward-arc length (floor((P-1)/2))
constexpr bool is_sender = (fwd_hops == 0);
constexpr bool on_fwd_arc = !is_sender && (fwd_hops <= HF);
constexpr bool on_bwd_arc = !is_sender && !on_fwd_arc;
// Send to idx+1 if sender or a non-terminal forward-arc device; to idx-1 if sender or non-terminal backward.
constexpr bool send_fwd = is_sender || (on_fwd_arc && fwd_hops < HF);
constexpr bool send_bwd = is_sender || (on_bwd_arc && bwd_hops < HB);
constexpr bool forwards = send_fwd || send_bwd;

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);     // local input shard (valid on sender)
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);    // local output
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // my recv-sem (upstream increments)
    const uint32_t ds_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);   // downstream sem noc coords
    const uint32_t ds_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ds_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // downstream recv-sem L1 addr
    const uint32_t tile_start = get_arg_val<uint32_t>(arg_idx++);   // this core/link's tile range (payload split)
    const uint32_t tile_count = get_arg_val<uint32_t>(arg_idx++);
    size_t fab_arg = arg_idx;                                       // remaining args -> fabric connection

    // Input + output addrgens. Input CT args start at tensor_args_base; output args follow immediately.
    // Same output spec on every device, so the output addrgen also names the downstream fabric-write target.
    constexpr auto in_args = TensorAccessorArgs<tensor_args_base>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in_addrgen = TensorAccessor(in_args, input_addr, page_size);
    const auto out_addrgen = TensorAccessor(out_args, output_addr, page_size);

    volatile tt_l1_ptr uint32_t* recv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);
    CircularBuffer cb(cb_id);

    constexpr ccl_routing_utils::line_unicast_route_info_t fwd_route = {
        .dst_mesh_id = static_cast<uint16_t>(fwd_route_arg0), .dst_chip_id = static_cast<uint16_t>(fwd_route_arg1)};
    constexpr ccl_routing_utils::line_unicast_route_info_t bwd_route = {
        .dst_mesh_id = static_cast<uint16_t>(bwd_route_arg0), .dst_chip_id = static_cast<uint16_t>(bwd_route_arg1)};

    // One connection manager holds both directions (build_from_args reads a forward flag then a backward
    // flag). The sender opens both; an arc-relay opens only its own direction.
    auto fabric_connection = FabricConnectionManager::build_from_args(fab_arg);
    // Payload + sem-inc packet headers, one pair per active direction (up to 4 total).
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd_sem = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd_sem = nullptr;
    if constexpr (forwards) {
        CircularBuffer cb_hdr(reserved_packet_header_cb_id);
        auto next_hdr = [&]() {
            cb_hdr.reserve_back(1);
            auto* p = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_hdr.get_write_ptr());
            cb_hdr.push_back(1);
            return p;
        };
        if constexpr (send_fwd) {
            pkt_hdr_fwd = next_hdr();
            pkt_hdr_fwd_sem = next_hdr();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_fwd, fwd_route);
        }
        if constexpr (send_bwd) {
            pkt_hdr_bwd = next_hdr();
            pkt_hdr_bwd_sem = next_hdr();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_bwd, bwd_route);
        }
        fabric_connection.open();
    }
    auto* fwd_conn = send_fwd ? &fabric_connection.get_forward_connection() : nullptr;
    auto* bwd_conn = send_bwd ? &fabric_connection.get_backward_connection() : nullptr;

    (void)num_tiles;  // per-core range now comes from RT tile_start/tile_count (payload split across links)
    const uint32_t tile_end = tile_start + tile_count;
    uint32_t tiles_done = tile_start;
    uint32_t chunk = 0;
    while (tiles_done < tile_end) {
        const uint32_t chunk_tiles = std::min(tile_end - tiles_done, packet_size_in_pages);

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

        // 3) Forward the chunk to each active neighbour's OUTPUT, then bump its recv-sem. The downstream
        //    worker core is the same logical core on every device, so the sem noc coords are shared; only
        //    the fabric route (and connection) differ per direction. Two tiles per scatter packet (half the
        //    fabric writes), with a single-tile tail for an odd chunk.
        const uint64_t ds_sem_noc = safe_get_noc_addr(ds_sem_noc_x, ds_sem_noc_y, ds_sem_addr, 0);
        auto send_dir = [&](volatile PACKET_HEADER_TYPE* pkt_hdr,
                            volatile PACKET_HEADER_TYPE* pkt_hdr_sem,
                            tt::tt_fabric::WorkerToFabricEdmSender* conn,
                            const ccl_routing_utils::line_unicast_route_info_t& route) {
            uint32_t t = 0;
            for (; t + 1 < chunk_tiles; t += 2) {
                scatter_fabric_write_unidir(
                    tiles_done + t,
                    tiles_done + t + 1,
                    out_addrgen,
                    pkt_hdr,
                    *conn,
                    cb_rd + t * page_size,
                    static_cast<uint16_t>(page_size));
            }
            if (t < chunk_tiles) {  // odd tail
                fabric_write_unidir(tiles_done + t, out_addrgen, pkt_hdr, *conn, cb_rd + t * page_size, page_size);
            }
            pkt_hdr_sem->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ds_sem_noc, static_cast<uint32_t>(1)});
            conn->wait_for_empty_write_slot();
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem, route);
            conn->send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(pkt_hdr_sem), sizeof(PACKET_HEADER_TYPE));
        };
        if constexpr (send_fwd) {
            send_dir(pkt_hdr_fwd, pkt_hdr_fwd_sem, fwd_conn, fwd_route);
        }
        if constexpr (send_bwd) {
            send_dir(pkt_hdr_bwd, pkt_hdr_bwd_sem, bwd_conn, bwd_route);
        }

        cb.pop_front(chunk_tiles);
        tiles_done += chunk_tiles;
        ++chunk;
    }

    if constexpr (forwards) {
        fabric_connection.close();
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_reduce_async one_shot: link sender (fabric) side. See create_one_shot_descriptor in the program factory.
//   1. Reduce-scatter: every remote-owned chunk of this sender's link and direction (staged by one_shot_reader in the
//      RS CB, in send order) goes to its owner as a 1D unicast of <hops> hops into the owner core's reduction-buffer
//      slot my_chip_id. The last packet of a chunk is a fused write + increment of that core's semaphore.
//   2. All-gather: every reduced chunk this device owns on this link (staged by the reader in the AG CB once the owner
//      cores signalled) is line-multicast in this sender's direction(s) into the same core's output shard on every
//      peer, the last packet again fused with the semaphore increment.

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <algorithm>
#include <cstdint>

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_rs_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_ag_id = get_compile_time_arg_val(3);
constexpr uint32_t in_page_size = get_compile_time_arg_val(4);
constexpr uint32_t out_page_size = get_compile_time_arg_val(5);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(6);
constexpr uint32_t in_pages_per_packet = get_compile_time_arg_val(7);
constexpr uint32_t out_pages_per_packet = get_compile_time_arg_val(8);
constexpr uint32_t reduction_cb_id = get_compile_time_arg_val(9);

FORCE_INLINE void send_chunk_packet(
    tt::tt_fabric::WorkerToFabricEdmSender& connection,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    uint32_t l1_read_addr,
    uint64_t dest_noc_addr,
    uint64_t sem_noc_addr,
    uint32_t payload_bytes,
    bool last) {
    if (last) {
        pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest_noc_addr, sem_noc_addr, 1, false},
            payload_bytes);
    } else {
        pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr}, payload_bytes);
    }
    perform_payload_send(connection, l1_read_addr, payload_bytes, pkt_hdr);
}

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_rs_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_ag_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fwd_range = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bwd_range = get_arg_val<uint32_t>(arg_idx++);
    const size_t rs_arg_base = arg_idx;  // per chunk: (noc x << 16 | y), (direction << 8 | hops)
    arg_idx += 2 * num_rs_chunks;
    const size_t ag_arg_base = arg_idx;  // per chunk: (noc x << 16 | y)
    arg_idx += num_ag_chunks;

    // Only start the open here; open_finish (a barrier) runs after the header setup below.
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    const uint32_t sem_addr = get_common_arg_val<uint32_t>(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    // Our slot in every owner's reduction buffer (same address on every core and device).
    const uint32_t slot_addr = get_write_ptr(reduction_cb_id) + my_chip_id * chunk_tiles * in_page_size;

    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    CircularBuffer cb_rs(cb_rs_id);
    CircularBuffer cb_ag(cb_ag_id);

    cb_packet_header.reserve_back(1);
    auto* pkt_hdr_forward = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_packet_header.get_write_ptr());
    cb_packet_header.push_back(1);
    cb_packet_header.reserve_back(1);
    auto* pkt_hdr_backward = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_packet_header.get_write_ptr());
    cb_packet_header.push_back(1);

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    // 1. Reduce-scatter: unicast each remote-owned chunk into its owner's slot my_chip_id.
    for (uint32_t i = 0; i < num_rs_chunks; i++) {
        const uint32_t xy = get_arg_val<uint32_t>(rs_arg_base + 2 * i);
        const uint32_t route = get_arg_val<uint32_t>(rs_arg_base + 2 * i + 1);
        const uint8_t noc_x = xy >> 16;
        const uint8_t noc_y = xy & 0xFFFF;
        const bool forward = (route >> 8) == 0;
        const uint8_t hops = route & 0xFF;
        auto& connection =
            forward ? fabric_connection.get_forward_connection() : fabric_connection.get_backward_connection();
        volatile PACKET_HEADER_TYPE* pkt_hdr = forward ? pkt_hdr_forward : pkt_hdr_backward;
        pkt_hdr->to_chip_unicast(hops);
        const uint64_t sem_noc_addr = safe_get_noc_addr(noc_x, noc_y, sem_addr, 0);

        cb_rs.wait_front(chunk_tiles);
        const uint32_t l1_read_addr = cb_rs.get_read_ptr();
        for (uint32_t t = 0; t < chunk_tiles; t += in_pages_per_packet) {
            const uint32_t n = std::min(in_pages_per_packet, chunk_tiles - t);
            send_chunk_packet(
                connection,
                pkt_hdr,
                l1_read_addr + t * in_page_size,
                safe_get_noc_addr(noc_x, noc_y, slot_addr + t * in_page_size, 0),
                sem_noc_addr,
                n * in_page_size,
                t + n >= chunk_tiles);
            // The header (and payload) must have left L1 before the header is rewritten.
            noc_async_writes_flushed();
        }
        cb_rs.pop_front(chunk_tiles);
    }

    // 2. All-gather: multicast each reduced chunk of this link into the same core's output shard on every peer.
    if (num_ag_chunks > 0) {
        if (fwd_range > 0) {
            pkt_hdr_forward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(fwd_range)});
        }
        if (bwd_range > 0) {
            pkt_hdr_backward->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(bwd_range)});
        }
        cb_ag.wait_front(num_ag_chunks * chunk_tiles);
        uint32_t l1_read_addr = cb_ag.get_read_ptr();
        for (uint32_t j = 0; j < num_ag_chunks; j++) {
            const uint32_t xy = get_arg_val<uint32_t>(ag_arg_base + j);
            const uint8_t noc_x = xy >> 16;
            const uint8_t noc_y = xy & 0xFFFF;
            const uint64_t sem_noc_addr = safe_get_noc_addr(noc_x, noc_y, sem_addr, 0);
            for (uint32_t t = 0; t < chunk_tiles; t += out_pages_per_packet) {
                const uint32_t n = std::min(out_pages_per_packet, chunk_tiles - t);
                const uint64_t dest_noc_addr = safe_get_noc_addr(noc_x, noc_y, out_addr + t * out_page_size, 0);
                const bool last = t + n >= chunk_tiles;
                if (fwd_range > 0) {
                    send_chunk_packet(
                        fabric_connection.get_forward_connection(),
                        pkt_hdr_forward,
                        l1_read_addr + t * out_page_size,
                        dest_noc_addr,
                        sem_noc_addr,
                        n * out_page_size,
                        last);
                }
                if (bwd_range > 0) {
                    send_chunk_packet(
                        fabric_connection.get_backward_connection(),
                        pkt_hdr_backward,
                        l1_read_addr + t * out_page_size,
                        dest_noc_addr,
                        sem_noc_addr,
                        n * out_page_size,
                        last);
                }
                noc_async_writes_flushed();
            }
            l1_read_addr += chunk_tiles * out_page_size;
        }
        cb_ag.pop_front(num_ag_chunks * chunk_tiles);
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_finish();
    }
    noc_async_write_barrier();
}

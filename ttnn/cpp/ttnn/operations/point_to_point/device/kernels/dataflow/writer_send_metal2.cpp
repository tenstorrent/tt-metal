// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t alignment = get_arg(args::alignment);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const auto page_idx_start = get_arg(args::page_idx_start);
    const auto page_idx_end = get_arg(args::page_idx_end);
    const uint8_t dst_num_hops = get_arg(args::dst_num_hops);
    const auto page_size_bytes = get_arg(args::page_size_bytes);
    const auto payload_size_bytes = get_arg(args::payload_size_bytes);
    const auto max_pages_per_packet = get_arg(args::max_pages_per_packet);
    const auto page_segments = get_arg(args::page_segments);
    experimental::GlobalSemaphore receive_semaphore(global_sem::receive_semaphore);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    Noc noc;

    auto fabric_connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::connection);
    fabric_connection.open_start();

    DataflowBuffer dfb_sender(dfb::sender);
    DataflowBuffer dfb_packet_header(dfb::packet_header);
    DataflowBuffer dfb_packet(dfb::packet);

    // set up packet header buffer
    dfb_packet_header.reserve_back(1);
    uint32_t packet_header_addr = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    const auto dst_buffer = TensorAccessor(tensor::dst);

    // working memory to hold coalesced packet
    dfb_packet.reserve_back(1);
    const uint32_t packet_base_addr = dfb_packet.get_write_ptr();
    dfb_packet.push_back(1);

    // initial packet size
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    // wait for receiver to signal it is ready
    receive_semaphore.wait(1);
    // clean up semaphore – needs to be done before the sender side semaphore increment if we're reusing the semaphore
    // in subsequent program cache hits
    receive_semaphore.set(0);

    fabric_connection.open_finish();

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        dfb_sender.wait_front(1);
        const uint32_t src_page_base_addr = dfb_sender.get_read_ptr();
        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            const uint32_t page_offset = page_segment_idx * payload_size_bytes;
            const uint32_t src_addr = src_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, payload_size_bytes);

            // copy page to packet buffer with offset
            const uint32_t packet_addr = packet_base_addr + packet_page_idx * aligned_page_size_bytes;
            tt_memmove<false, false, false, 0>(noc, packet_addr, src_addr, transfer_size_bytes);
            ++packet_page_idx;
            if (packet_page_idx >= curr_pages_per_packet) {
                tt::tt_fabric::linear::to_noc_unicast_write(
                    align(payload_size_bytes, alignment), packet_header_ptr, packet_idx, dst_buffer);
                // blocking+flush: drain the payload out of packet_base_addr before the next tt_memmove
                // reuses the single-slot dfb_packet (the default overload leaves the source read in flight).
                perform_payload_send<true, true>(
                    fabric_connection, packet_base_addr, payload_size_bytes, packet_header_ptr);

                // reset counters
                packet_page_idx = 0;
                curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx - 1);

                ++packet_idx;
            }
        }
        dfb_sender.pop_front(1);
    }

    dfb_packet_header.reserve_back(1);
    const uint32_t sem_header_addr = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);

    const uint64_t receive_sem_noc_addr = get_noc_addr(receive_semaphore.address());

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, dst_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{receive_sem_noc_addr, 1});

    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();
}

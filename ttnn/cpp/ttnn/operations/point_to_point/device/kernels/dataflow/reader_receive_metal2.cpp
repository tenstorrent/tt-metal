// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t alignment = get_arg(args::alignment);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const auto page_idx_start = get_arg(args::page_idx_start);
    const auto page_idx_end = get_arg(args::page_idx_end);
    const auto max_pages_per_packet = get_arg(args::max_pages_per_packet);
    const auto packet_size_bytes = get_arg(args::packet_size_bytes);
    const auto page_size_bytes = get_arg(args::page_size_bytes);
    const auto page_segments = get_arg(args::page_segments);
    experimental::GlobalSemaphore sender_semaphore(global_sem::sender_semaphore);
    const uint8_t sender_num_hops = get_arg(args::sender_num_hops);

    Noc noc;
    DataflowBuffer dfb_packet_header(dfb::packet_header);
    DataflowBuffer dfb_packet(dfb::packet);
    DataflowBuffer dfb_receiver(dfb::receiver);

    auto fabric_connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::connection);
    fabric_connection.open_start();

    dfb_packet_header.reserve_back(1);
    const uint32_t sem_header_addr = dfb_packet_header.get_write_ptr();
    dfb_packet_header.push_back(1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_semaphore.address());

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    fabric_connection.open_finish();

    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();

    const auto packet_buffer = TensorAccessor(tensor::packet_buffer);

    dfb_packet.reserve_back(1);
    const uint64_t packet_l1_addr = dfb_packet.get_write_ptr();

    sender_semaphore.wait(1);

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        dfb_receiver.reserve_back(1);
        const uint32_t dest_page_base_addr = dfb_receiver.get_write_ptr();

        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            if (page_idx == page_idx_start || packet_page_idx == curr_pages_per_packet) {
                noc.async_read(
                    packet_buffer, dfb_packet, packet_size_bytes, {.page_id = packet_idx}, {.offset_bytes = 0});
                noc.async_read_barrier();

                packet_page_idx = 0;
                curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx);
                ++packet_idx;
            }

            const uint32_t page_offset = page_segment_idx * packet_size_bytes;
            const uint32_t dest_addr = dest_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, packet_size_bytes);
            const uint32_t packet_l1_page_addr = packet_l1_addr + packet_page_idx * aligned_page_size_bytes;

            tt_memmove<false, false, false, 0>(noc, dest_addr, packet_l1_page_addr, transfer_size_bytes);
            ++packet_page_idx;
        }
        dfb_receiver.push_back(1);
    }
    dfb_packet.push_back(1);

    // clean up semaphore in case it is reused
    sender_semaphore.set(0);
}

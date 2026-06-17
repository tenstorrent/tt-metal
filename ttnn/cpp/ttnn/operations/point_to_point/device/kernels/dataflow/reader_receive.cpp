// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using tt::data_movement::common::tt_memmove;
using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr auto packet_buffer_args = TensorAccessorArgs<3>();

    const auto page_idx_start = get_arg_val<uint32_t>(0);
    const auto page_idx_end = get_arg_val<uint32_t>(1);
    const auto max_pages_per_packet = get_arg_val<uint32_t>(2);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(3);
    const auto packet_size_bytes = get_arg_val<uint32_t>(4);
    const auto page_size_bytes = get_arg_val<uint32_t>(5);
    const auto page_segments = get_arg_val<uint32_t>(6);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(7);
    const uint8_t sender_num_hops = get_arg_val<uint32_t>(8);

    // The fabric arg block (appended by ttnn::ccl::dataflow::append_ccl_fabric_rt_args)
    // begins at index 9; its leading has_forward flag also encodes the route direction.
    size_t conn_arg_idx = 9;
    const bool sender_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> ack(conn_arg_idx, sender_is_forward, alignment);

    // Signal the sender we are "ready" to receive (atomic-inc over fabric), then tear down.
    ack.open();
    ack.set_route_unicast(sender_num_hops);
    ack.arm_inc(1);
    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_semaphore_addr);
    ack.inc(sender_sem_noc_addr);
    ack.close();

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto packet_buffer = TensorAccessor(packet_buffer_args, intermediate_base_addr, packet_size_bytes);

    cb_reserve_back(packet_cb_id, 1);
    const uint64_t packet_l1_addr = get_write_ptr(packet_cb_id);

    // Wait for the sender's "done" — the payload has fully landed in the intermediate buffer.
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait_min(local_semaphore_ptr, 1);

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    // Op-owned de-coalescing: read each landed packet locally and scatter its pages into the output CB.
    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        cb_reserve_back(receiver_cb_id, 1);
        const uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id);

        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            if (page_idx == page_idx_start || packet_page_idx == curr_pages_per_packet) {
                const uint64_t packet_noc_addr = packet_buffer.get_noc_addr(packet_idx, 0, 0);
                noc_async_read(packet_noc_addr, packet_l1_addr, packet_size_bytes);
                noc_async_read_barrier();

                packet_page_idx = 0;
                curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx);
                ++packet_idx;
            }

            const uint32_t page_offset = page_segment_idx * packet_size_bytes;
            const uint32_t dest_addr = dest_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, packet_size_bytes);
            const uint32_t packet_l1_page_addr = packet_l1_addr + packet_page_idx * aligned_page_size_bytes;

            tt_memmove<false, false, false, 0>(dest_addr, packet_l1_page_addr, transfer_size_bytes);
            ++packet_page_idx;
        }
        cb_push_back(receiver_cb_id, 1);
    }
    cb_push_back(packet_cb_id, 1);

    // clean up the semaphore in case it is reused on a program-cache hit
    noc_semaphore_set(local_semaphore_ptr, 0);
}

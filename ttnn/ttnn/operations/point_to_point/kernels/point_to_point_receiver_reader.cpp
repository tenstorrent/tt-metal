// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — RECEIVE program reader (NCRISC): the fabric handshake + ingress kernel.
//
// Uses `dataflow_kernel_lib::ccl::FabricStreamSender::signal()` (the one-shot
// open->arm_inc->inc->close collapse) to send the sender the "ready" ack. The
// SENDING half of the sync is the only fabric operation here; the op owns:
//   * the "ready" ack timing (before the payload wait);
//   * the WAITING half — noc_semaphore_wait_min on the local "done" cell;
//   * the receive INGRESS — a LOCAL noc_async_read of the landed intermediate buffer;
//   * packet->page de-coalescing — tt_memmove;
//   * the cache-reuse re-arm — noc_semaphore_set(sem, 0) AFTER the wait.
//
// Handshake ordering (matches ccl_helpers_dataflow.hpp:74-77 cache-reuse rule):
//   1. signal the sender "ready" (fabric atomic-inc onto the sender's cell)
//   2. wait for the sender's "done" (local cell, incremented by the sender's fabric inc)
//   3. local-read each landed packet, de-coalesce into output pages (cb_recv_pages)
//   4. reset the local cell to 0 AFTER the wait (re-arm for program-cache reuse)

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

    // The fabric arg block (appended by the host, mirroring append_ccl_fabric_rt_args)
    // begins at index 9; its leading has_forward flag also encodes the route direction.
    size_t conn_arg_idx = 9;
    const bool sender_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> ack_sender(conn_arg_idx, sender_is_forward, alignment);

    // Signal the sender we are "ready" to receive: one fabric atomic-inc, then tear down.
    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_semaphore_addr);
    ack_sender.signal(sender_num_hops, sender_sem_noc_addr);

    // page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize (stale on cache hits).
    const auto packet_buffer = TensorAccessor(packet_buffer_args, intermediate_base_addr, packet_size_bytes);

    cb_reserve_back(packet_cb_id, 1);
    // L1 write pointer is a 32-bit address (matches the sender's packet_base_addr); it is
    // used as the local NoC-read destination and as the base for per-page tt_memmove offsets.
    const uint32_t packet_l1_addr = get_write_ptr(packet_cb_id);

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

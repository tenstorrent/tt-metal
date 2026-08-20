// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — SEND program, writer kernel (BRISC).
//
// Coalesces the streamed input pages into fabric packets and unicast-writes
// them over the Tenstorrent fabric into the receiver device's intermediate
// (landing) buffer, then signals "done" with a cross-device atomic-inc.
//
// Fabric egress uses the safety-by-construction CCL helper
// (FabricStreamSender -> FabricStream -> UnicastWriteChannel / AtomicIncChannel).
// The op owns: page<->packet coalescing (tt_memmove), the WAITING half of the
// handshake (noc_semaphore_wait_min) and the cache-reuse re-arm
// (noc_semaphore_set) — none of which the helper owns by design.

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;
using namespace dataflow_kernel_lib::ccl;

void kernel_main() {
    constexpr uint32_t cb_input_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_packet_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr auto intermediate_args = TensorAccessorArgs<3>();

    const uint32_t receiver_base_address = get_arg_val<uint32_t>(0);
    const auto page_idx_start = get_arg_val<uint32_t>(1);
    const auto page_idx_end = get_arg_val<uint32_t>(2);
    const uint8_t dst_num_hops = get_arg_val<uint32_t>(3);
    const auto page_size_bytes = get_arg_val<uint32_t>(4);
    const auto payload_size_bytes = get_arg_val<uint32_t>(5);
    const auto max_pages_per_packet = get_arg_val<uint32_t>(6);
    const auto page_segments = get_arg_val<uint32_t>(7);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(8);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    // The fabric arg block (laid out by append_ccl_fabric_rt_args) begins at
    // index 9; its leading has_forward flag also encodes the send direction.
    size_t conn_arg_idx = 9;
    const bool dst_is_forward = get_arg_val<uint32_t>(conn_arg_idx);
    FabricStreamSender<> sender(conn_arg_idx, dst_is_forward, alignment);

    // Intermediate is addressed PER-PACKET: page size override = payload_size_bytes,
    // page index = packet_idx.
    const auto intermediate = TensorAccessor(intermediate_args, receiver_base_address, payload_size_bytes);

    // Working L1 buffer that holds one coalesced packet (reserve once; scratch).
    cb_reserve_back(cb_packet_scratch, 1);
    const uint32_t packet_base_addr = get_write_ptr(cb_packet_scratch);
    cb_push_back(cb_packet_scratch, 1);

    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    // Wait for the receiver's "ready", then reset BEFORE our own outgoing inc so a
    // program-cache hit re-arms the semaphore cleanly (cache-reuse footgun).
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr);
    noc_semaphore_wait_min(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);

    // open(route) binds the stream's route once; arm_* yield the only objects that can issue and
    // reuse that route, so an unrouted send cannot be written.
    auto stream = sender.open(unicast_route(dst_num_hops));
    auto writer = stream.arm_unicast_write(payload_size_bytes);  // invariant per-packet payload size
    auto done = stream.arm_inc(1);                               // invariant inc value for the "done" signal

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        cb_wait_front(cb_input_pages, 1);
        const uint32_t src_page_base_addr = get_read_ptr(cb_input_pages);
        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            const uint32_t page_offset = page_segment_idx * payload_size_bytes;
            const uint32_t src_addr = src_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, payload_size_bytes);

            // Copy page (or segment) into the packet buffer at the running offset.
            const uint32_t packet_addr = packet_base_addr + packet_page_idx * aligned_page_size_bytes;
            tt_memmove<false, false, false, 0>(packet_addr, src_addr, transfer_size_bytes);
            ++packet_page_idx;
            if (packet_page_idx >= curr_pages_per_packet) {
                // Op owns the coalescing (page->packet, packet_idx); helper owns the fabric write.
                writer.write_page(packet_base_addr, packet_idx, intermediate);

                packet_page_idx = 0;
                curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx - 1);
                ++packet_idx;
            }
        }
        cb_pop_front(cb_input_pages, 1);
    }

    // Signal the receiver "done" (payload fully landed in its intermediate).
    const uint64_t receive_sem_noc_addr = get_noc_addr(receive_semaphore_addr);
    done.inc(receive_sem_noc_addr);

    stream.close();  // drains the trailing inc, then closes (the dtor would also close — idempotent)
}

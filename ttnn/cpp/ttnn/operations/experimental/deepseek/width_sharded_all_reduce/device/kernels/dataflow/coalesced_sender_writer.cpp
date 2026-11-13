// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

// Sends each reducer core's contiguous chunk as one fabric multicast packet with fused
// atomic increment. Route maps are computed once per direction and replicated across
// a 4-slot header pipeline so packets stream as cb0 chunks arrive.
void kernel_main() {
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
    constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
    constexpr uint32_t group_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t group_pages = get_compile_time_arg_val(6);
    constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
    constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
    constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
        ccl_routing_utils::get_line_multicast_route_info_from_args<9>();
    constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
        ccl_routing_utils::get_line_multicast_route_info_from_args<9 + ccl_routing_utils::num_line_multicast_args>();

    size_t arg_idx = 0;
    const uint32_t scratch_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_even_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_odd_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_iter_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(arg_idx++);

    tt_l1_ptr uint32_t* red_x = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_reducers;
    tt_l1_ptr uint32_t* red_y = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_reducers;

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* iter_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_iter_addr);
    const uint32_t iter = *iter_ptr;
    *iter_ptr = iter + 1u;
    const uint32_t half = iter & 1u;

    constexpr uint32_t kHeaderSlots = 4;
    constexpr uint32_t kHeaderWords = sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t);
    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    cb_packet_header.reserve_back(2u * kHeaderSlots);
    const uint32_t hdr_base = cb_packet_header.get_write_ptr();

    auto copy_header = [&](uint32_t dst_addr, uint32_t src_addr) {
        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
        for (uint32_t w = 0; w < kHeaderWords; ++w) {
            dst[w] = src[w];
        }
    };

    if (fabric_connection.has_forward_connection()) {
        auto* hdr0 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_base);
        ccl_routing_utils::fabric_set_line_multicast_route(hdr0, forward_multicast_route_info);
        for (uint32_t s = 1; s < kHeaderSlots; ++s) {
            copy_header(hdr_base + s * packet_header_size_bytes, hdr_base);
        }
    }
    if (fabric_connection.has_backward_connection()) {
        const uint32_t bwd_base = hdr_base + kHeaderSlots * packet_header_size_bytes;
        auto* hdr0 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(bwd_base);
        ccl_routing_utils::fabric_set_line_multicast_route(hdr0, backward_multicast_route_info);
        for (uint32_t s = 1; s < kHeaderSlots; ++s) {
            copy_header(bwd_base + s * packet_header_size_bytes, bwd_base);
        }
    }
    cb_packet_header.push_back(2u * kHeaderSlots);

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    Noc noc;
    CircularBuffer cb0(cb0_id);

    const uint32_t dst_slot_addr = scratch_addr + (half * ring_size + my_chip_id) * group_bytes;
    const uint32_t sem_addr = half ? sem_odd_addr : sem_even_addr;

    for (uint32_t r = 0; r < num_reducers; ++r) {
        cb0.wait_front(group_pages);
        const uint32_t l1_read_addr = cb0.get_read_ptr();
        const uint32_t slot = r & (kHeaderSlots - 1u);
        if (r >= kHeaderSlots && slot == 0) {
            noc.async_writes_flushed();
        }

        const uint64_t dst_noc_addr = safe_get_noc_addr(red_x[r], red_y[r], dst_slot_addr);
        const uint64_t sem_noc_addr = safe_get_noc_addr(red_x[r], red_y[r], sem_addr);

        if (fabric_connection.has_forward_connection()) {
            auto* hdr_fwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_base + slot * packet_header_size_bytes);
            hdr_fwd->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, false},
                group_bytes);
            perform_payload_send<false, true>(
                fabric_connection.get_forward_connection(), l1_read_addr, group_bytes, hdr_fwd);
        }
        if (fabric_connection.has_backward_connection()) {
            auto* hdr_bwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                hdr_base + (kHeaderSlots + slot) * packet_header_size_bytes);
            hdr_bwd->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, false},
                group_bytes);
            perform_payload_send<false, true>(
                fabric_connection.get_backward_connection(), l1_read_addr, group_bytes, hdr_bwd);
        }
        cb0.pop_front(group_pages);
    }

    noc.async_writes_flushed();

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_finish();
    }

    noc.async_write_barrier();
}

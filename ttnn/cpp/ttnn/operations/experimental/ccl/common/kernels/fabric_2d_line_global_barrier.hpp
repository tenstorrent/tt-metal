// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

namespace ttnn::experimental::ccl {

// A Fabric2D multicast range can only continue in one physical direction. A
// logical line may turn in the physical mesh, so build the global barrier from
// representable one-hop multicasts instead:
//
//   arrival: device 0 -> ... -> terminal device
//   release: device 0 <- ... <- terminal device
//
// Each device has a forward and backward worker. The penultimate forward worker
// wakes both terminal workers; each backward worker then releases both workers
// on the preceding device. Consequently, no worker leaves the barrier until the
// arrival token has reached the terminal device.
template <typename FabricSenderType>
FORCE_INLINE void fabric_2d_line_global_barrier(
    tt_l1_ptr FabricSenderType* fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    const ccl_routing_utils::line_multicast_route_info_t& one_hop_route,
    uint32_t barrier_semaphore,
    uint8_t same_direction_worker_noc_x,
    uint8_t same_direction_worker_noc_y,
    uint8_t opposite_direction_worker_noc_x,
    uint8_t opposite_direction_worker_noc_y,
    bool is_forward_worker,
    bool is_first_device,
    uint32_t num_targets_in_direction) {
    const uint64_t same_direction_semaphore =
        safe_get_noc_addr(same_direction_worker_noc_x, same_direction_worker_noc_y, barrier_semaphore, 0);
    const uint64_t opposite_direction_semaphore =
        safe_get_noc_addr(opposite_direction_worker_noc_x, opposite_direction_worker_noc_y, barrier_semaphore, 0);

    if (num_targets_in_direction > 0) {
        ccl_routing_utils::fabric_set_line_multicast_route(packet_header, one_hop_route);
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc_set_state<
            tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask::Val |
            tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask::Flush>(
            packet_header,
            static_cast<uint8_t>(one_hop_route.start_distance_in_hops),
            static_cast<uint8_t>(one_hop_route.range_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
    }

    const auto signal_neighbor = [&](uint64_t destination) {
        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc_with_state<
            tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask::DstAddr>(
            fabric_connection, packet_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{destination, 0});
    };
    const auto wait_for = [&](uint32_t value) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_semaphore), value);
    };

    if (is_forward_worker) {
        if (!is_first_device) {
            wait_for(1);
        }
        if (num_targets_in_direction > 0) {
            signal_neighbor(same_direction_semaphore);
            if (num_targets_in_direction == 1) {
                signal_neighbor(opposite_direction_semaphore);
            }
            wait_for(is_first_device ? 1 : 2);
        }
    } else {
        wait_for(1);
        if (num_targets_in_direction > 0) {
            signal_neighbor(same_direction_semaphore);
            signal_neighbor(opposite_direction_semaphore);
        }
    }

    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_semaphore), 0);
}

}  // namespace ttnn::experimental::ccl

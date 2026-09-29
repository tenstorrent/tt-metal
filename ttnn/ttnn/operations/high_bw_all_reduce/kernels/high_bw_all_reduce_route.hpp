// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — route a packet header to the adjacent device the connection points at.
// Every send in this op is exactly one hop, so the route is the same on every fabric config:
//   * 2D fabrics (FABRIC_2D*): route by the destination's fabric node id (chip, mesh).
//   * 1D fabrics (FABRIC_1D, FABRIC_1D_RING, FABRIC_1D_NEIGHBOR_EXCHANGE): route by hop count (1); the
//     connection already fixes the direction.
// ROUTING_MODE is set by the JIT build from the active fabric config (fabric_routing_mode.h).

#pragma once

#include <cstdint>
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

inline void route_to_neighbor(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t dst_chip_id, uint16_t dst_mesh_id) {
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)tt::tt_fabric::fabric_set_unicast_route(hdr, dst_chip_id, dst_mesh_id);
#else
    (void)dst_chip_id;
    (void)dst_mesh_id;
    constexpr uint16_t one_hop = 1;
    (void)tt::tt_fabric::fabric_set_unicast_route</*target_as_dev=*/false>(
        reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::LowLatencyPacketHeader*>(hdr), one_hop);
#endif
}

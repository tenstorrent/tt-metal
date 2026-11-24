// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric port cores next to their link's Ethernet core. The Ethernet core comes from tt_fabric::get_forwarding_eth_core
// and the hop count from main's experimental Device API (as high_bw_all_gather places its link workers):
// hops(worker -> eth) = hops(worker -> closest worker to eth) + hops(closest -> eth). Ties go to the lower logical row,
// then column.

#pragma once

#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "mesh_routing.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

struct PortKey {
    ChipCoord chip;
    Dir dir;
    uint32_t link_index = 0;  // index into MeshPort::links (not the fabric link id)
    auto operator<=>(const PortKey&) const = default;
};
using PortCores = std::map<PortKey, tt::tt_metal::CoreCoord>;  // logical worker core per port

// One port core per (chip, direction, link), from `allowed` (logical cores, e.g. the ones another op leaves free;
// nullopt: the whole grid), each core used at most once: per chip the ports heaviest first (the destination chips the
// fabric's first hop sends their way), each on the free core with the fewest NoC1 hops from its link's Ethernet core
// (the sender is on NoC1) plus a cost for the load already placed in that core's column - a port's rows go north up
// its column first, so ports sharing a column share its links.
PortCores place_ports(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const MeshPlan& plan,
    const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block

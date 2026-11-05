// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// One fabric send out of a chip, leaving it for the neighbour `next`: a multicast through the neighbour
// `dst` over `hops` chips per physical direction (E, W, N, S; a N or S spine with E/W teeth lands on
// every chip of the rows it passes) or, with no hops, a unicast to `dst`. A unicast to a chip of the
// sender's column can be steered: it travels the column in direction `steer` whichever way is shorter.
struct D2DRoute {
    tt::tt_fabric::FabricNodeId dst;
    std::array<uint32_t, 4> hops;
    tt::tt_fabric::FabricNodeId next;
    uint32_t steer;
};
constexpr uint32_t kD2DNoSteer = 0xFF;

// The sends from `src` that land on exactly the chips `dsts`: one multicast when a line or a spine with
// teeth does, otherwise a unicast to each. All of them leave `src` in one direction, so that one
// fabric connection carries them: when the shortest routes to receivers in `src`'s column split, the
// column's unicasts are all steered the way of the nearest.
std::vector<D2DRoute> d2d_routes(
    const tt::tt_fabric::FabricNodeId& src, const std::vector<tt::tt_fabric::FabricNodeId>& dsts);

// A route as the kernels read it from their runtime args (d2d_send.h): destination chip, its mesh,
// then the hops E, W, N, S, then the steering direction.
constexpr uint32_t kD2DRouteArgs = 7;
void append_d2d_route_args(const D2DRoute& route, std::vector<uint32_t>& args);

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

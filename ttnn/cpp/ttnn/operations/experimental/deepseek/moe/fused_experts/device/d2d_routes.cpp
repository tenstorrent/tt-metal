// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "d2d_routes.hpp"

#include <algorithm>
#include <optional>
#include <set>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

std::vector<D2DRoute> d2d_routes(
    const tt::tt_fabric::FabricNodeId& src, const std::vector<tt::tt_fabric::FabricNodeId>& dsts) {
    using tt::tt_fabric::eth_chan_directions;
    using tt::tt_fabric::FabricNodeId;
    TT_FATAL(!dsts.empty(), "d2d_routes: no receivers");
    // Logical chip ids number a mesh's chips row-major; rows run N to S and columns W to E.
    const auto shape = tt::tt_fabric::get_physical_mesh_shapes().at(src.mesh_id);
    const uint32_t rows = shape[0];
    const uint32_t cols = shape[1];
    auto row = [&](const FabricNodeId& n) { return n.chip_id / cols; };
    auto col = [&](const FabricNodeId& n) { return n.chip_id % cols; };
    auto node = [&](uint32_t r, uint32_t c) { return FabricNodeId(src.mesh_id, (r * cols) + c); };
    // `k` hops from (r, c) going `dir`, wrapping round a torus axis.
    auto step = [&](uint32_t r, uint32_t c, uint32_t dir, uint32_t k) {
        switch (dir) {
            case eth_chan_directions::EAST: return node(r, (c + k) % cols);
            case eth_chan_directions::WEST: return node(r, (c + cols - (k % cols)) % cols);
            case eth_chan_directions::NORTH: return node((r + rows - (k % rows)) % rows, c);
            default: return node((r + k) % rows, c);
        }
    };
    // The direction a packet for `to` leaves `from` in, if cardinal.
    auto direction = [&](const FabricNodeId& from, const FabricNodeId& to) -> std::optional<uint32_t> {
        const auto dir = tt::tt_fabric::get_eth_forwarding_direction(from, to);
        if (!dir.has_value() || *dir >= eth_chan_directions::Z) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(*dir);
    };
    auto distance = [&](const FabricNodeId& from, const FabricNodeId& to, uint32_t dir) {
        switch (dir) {
            case eth_chan_directions::EAST: return (col(to) + cols - col(from)) % cols;
            case eth_chan_directions::WEST: return (col(from) + cols - col(to)) % cols;
            case eth_chan_directions::NORTH: return (row(from) + rows - row(to)) % rows;
            default: return (row(to) + rows - row(from)) % rows;
        }
    };

    std::set<uint32_t> wanted;
    for (const auto& d : dsts) {
        TT_FATAL(
            d.mesh_id == src.mesh_id && d != src, "d2d_routes: receiver {} is not another chip of {}'s mesh", d, src);
        wanted.insert(d.chip_id);
    }

    // The one multicast that lands on exactly the receivers, if any.
    auto multicast = [&]() -> std::optional<D2DRoute> {
        D2DRoute route{.dst = src, .hops = {}, .next = src, .steer = kD2DNoSteer};
        std::set<uint32_t> landed;
        const bool line = std::all_of(dsts.begin(), dsts.end(), [&](const auto& d) { return row(d) == row(src); }) ||
                          std::all_of(dsts.begin(), dsts.end(), [&](const auto& d) { return col(d) == col(src); });
        std::optional<uint32_t> dir;
        if (line) {
            dir = direction(src, dsts.front());
            for (const auto& d : dsts) {
                if (!dir.has_value() || direction(src, d) != dir) {
                    return std::nullopt;
                }
                route.hops[*dir] = std::max(route.hops[*dir], distance(src, d, *dir));
            }
        } else {
            // A spine lands its teeth on every row it crosses, so the receivers have to be whole runs of the rows
            // it crosses, one direction from src: the spine reaches the farthest row, the teeth the widest run.
            // The landed set below has to come out exactly the receivers.
            for (const auto& d : dsts) {
                if (row(d) == row(src)) {
                    return std::nullopt;
                }
                const auto spine_at = node(row(d), col(src));
                const auto spine_dir = direction(src, spine_at);
                if ((spine_dir != eth_chan_directions::NORTH && spine_dir != eth_chan_directions::SOUTH) ||
                    (dir.has_value() && spine_dir != dir)) {
                    return std::nullopt;
                }
                dir = spine_dir;
                route.hops[*dir] = std::max(route.hops[*dir], distance(src, spine_at, *dir));
                if (d == spine_at) {
                    continue;
                }
                const auto tooth = direction(spine_at, d);
                if (tooth != eth_chan_directions::EAST && tooth != eth_chan_directions::WEST) {
                    return std::nullopt;
                }
                route.hops[*tooth] = std::max(route.hops[*tooth], distance(spine_at, d, *tooth));
            }
        }
        route.dst = step(row(src), col(src), *dir, 1);
        route.next = route.dst;
        for (uint32_t k = 1; k <= route.hops[*dir]; ++k) {
            const auto spine = step(row(src), col(src), *dir, k);
            landed.insert(spine.chip_id);
            for (const uint32_t tooth : {eth_chan_directions::EAST, eth_chan_directions::WEST}) {
                if (tooth == *dir) {
                    continue;
                }
                for (uint32_t j = 1; j <= route.hops[tooth]; ++j) {
                    landed.insert(step(row(spine), col(spine), tooth, j).chip_id);
                }
            }
        }
        if (landed != wanted) {
            return std::nullopt;
        }
        return route;
    };

    std::vector<D2DRoute> routes;
    if (auto route = multicast()) {
        routes.push_back(*route);
    } else {
        for (const auto& d : dsts) {
            const auto dir = direction(src, d);
            TT_FATAL(dir.has_value(), "d2d_routes: no cardinal route from {} to {}", src, d);
            routes.push_back(
                D2DRoute{.dst = d, .hops = {}, .next = step(row(src), col(src), *dir, 1), .steer = kD2DNoSteer});
        }
        const bool split =
            std::any_of(routes.begin(), routes.end(), [&](const auto& r) { return r.next != routes.front().next; });
        const bool column = std::all_of(dsts.begin(), dsts.end(), [&](const auto& d) { return col(d) == col(src); });
        if (split && column) {
            const auto nearest = std::min_element(routes.begin(), routes.end(), [&](const auto& a, const auto& b) {
                return distance(src, a.dst, *direction(src, a.dst)) < distance(src, b.dst, *direction(src, b.dst));
            });
            const uint32_t steer = *direction(src, nearest->dst);
            const auto next = nearest->next;
            for (auto& r : routes) {
                r.steer = steer;
                r.next = next;
            }
        }
    }
    for (const auto& route : routes) {
        TT_FATAL(
            route.next == routes.front().next,
            "d2d_routes: the sends from {} to its {} receivers leave it in more than one direction, which one fabric "
            "connection cannot carry",
            src,
            dsts.size());
    }
    return routes;
}

void append_d2d_route_args(const D2DRoute& route, std::vector<uint32_t>& args) {
    args.push_back(route.dst.chip_id);
    args.push_back(*route.dst.mesh_id);
    args.insert(args.end(), route.hops.begin(), route.hops.end());
    args.push_back(route.steer);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

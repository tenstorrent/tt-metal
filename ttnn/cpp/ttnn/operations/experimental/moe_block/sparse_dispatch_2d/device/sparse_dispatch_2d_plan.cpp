// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_dispatch_2d_plan.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

namespace sparse_dispatch_2d_detail {

// The other axis's directions (+, -) of direction d.
std::pair<Dir, Dir> axis_dirs(Dir d) {
    return is_x(d) ? std::pair{Dir::YPlus, Dir::YMinus} : std::pair{Dir::XPlus, Dir::XMinus};
}

bool has_port(const MeshPlan& plan, ChipCoord chip, Dir d) {
    return plan.chips.at(chip)[static_cast<size_t>(d)].has_value();
}

}  // namespace sparse_dispatch_2d_detail

PortWiring wiring(const MeshPlan& plan, ChipCoord chip, Dir d) {
    using namespace sparse_dispatch_2d_detail;
    const auto& topo = plan.topo;
    const auto& port = plan.chips.at(chip)[static_cast<size_t>(d)];
    TT_FATAL(port.has_value(), "sparse_dispatch_2d: chip ({}, {}) has no port {}", chip.r, chip.c, dir_name(d));
    const auto [pa, ma] = axis_dirs(d);
    PortWiring w;
    const auto behind = topo.neighbor(chip, opposite(d));
    if (behind && has_port(plan, *behind, d)) {
        w.ring_mask |= 1;
    } else if (!has_port(plan, chip, opposite(d)) && !has_port(plan, chip, pa) && !has_port(plan, chip, ma)) {
        w.ring_mask |= 1 | 8;  // (the link from the port d's neighbour ends here)
    }
    const auto src_plus = topo.neighbor(chip, ma);  // a row moving + arrives from the - side
    if (src_plus && has_port(plan, *src_plus, pa)) {
        w.ring_mask |= 2;
    }
    const auto src_minus = topo.neighbor(chip, pa);
    if (src_minus && has_port(plan, *src_minus, ma)) {
        w.ring_mask |= 4;
    }
    const ChipCoord peer = port->peer;
    const uint32_t k_turn = is_plus(d) ? 1 : 2;  // (ring 1 at the neighbour if this port moves rows +, else ring 2)
    const std::array<std::pair<Dir, uint32_t>, 3> targets = {{{d, 0}, {pa, k_turn}, {ma, k_turn}}};
    bool any = false;
    for (size_t s = 0; s < 3; ++s) {
        const auto [t, k] = targets[s];
        const uint32_t on = has_port(plan, peer, t) ? 1 : 0;
        w.streams[s] = StreamWiring{.on = on, .target = t, .ring = k};
        any = any || on;
    }
    if (!any) {
        w.streams[0] = StreamWiring{.on = 2, .target = opposite(d), .ring = 0};
    }
    return w;
}

CbSizes cb_sizes(
    const MeshTopology& topo,
    uint32_t S,
    uint32_t row_bytes,
    uint32_t plan_bytes,
    uint32_t group,
    uint32_t manifest_batch) {
    const auto ml = manifest_layout(topo.R, topo.C, topo.wrap_x, topo.wrap_y, S);
    return CbSizes{
        .rows = 2 * group * row_bytes,
        .meta = 2 * group * META_PAGE,
        .plan = plan_bytes,
        .stage = 3 * 2 * manifest_batch * 16,
        .manifest = ml.total_entries * 16,
    };
}

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d

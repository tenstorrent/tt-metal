// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "layer_send.hpp"

#include "device/layer_send_device_operation.hpp"

namespace ttnn {

std::shared_ptr<const operations::layer_send::SendPlan> layer_send_plan(
    uint32_t kind,
    uint32_t rows,
    uint32_t cols,
    bool wrap_x,
    bool wrap_y,
    const std::vector<std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>>& hops,
    const std::vector<uint32_t>& consts,
    const std::vector<uint32_t>& sems,
    const std::vector<uint32_t>& pattern) {
    namespace ls = operations::layer_send;
    TT_FATAL(kind <= static_cast<uint32_t>(ls::SendKind::GLUED), "layer_send_plan: unknown kind {}", kind);
    TT_FATAL(hops.size() == rows * cols, "layer_send_plan: {} hop lists for a {}x{} mesh", hops.size(), rows, cols);
    auto p = std::make_shared<ls::SendPlan>();
    p->kind = static_cast<ls::SendKind>(kind);
    p->rows = rows;
    p->cols = cols;
    p->wrap_x = wrap_x ? 1 : 0;
    p->wrap_y = wrap_y ? 1 : 0;
    std::vector<uint32_t> flat;
    p->hops.resize(hops.size());
    for (size_t q = 0; q < hops.size(); ++q) {
        flat.push_back(static_cast<uint32_t>(hops[q].size()));
        for (const auto& [dir, pr, pc, link, dr, dc] : hops[q]) {
            p->hops[q].push_back(
                ls::Hop{.dir = dir, .peer_r = pr, .peer_c = pc, .link = link, .dst_r = dr, .dst_c = dc});
            flat.insert(flat.end(), {dir, pr, pc, link, dr, dc});
        }
    }
    p->consts = consts;
    p->sems = sems;
    p->pattern = pattern;
    p->hash = ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<ls::LayerSendDeviceOperation>,
        kind,
        rows,
        cols,
        p->wrap_x,
        p->wrap_y,
        flat,
        consts,
        sems,
        pattern);
    return p;
}

Tensor layer_send(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const operations::layer_send::SendPlan>& plan,
    const std::vector<uint32_t>& values) {
    return ttnn::prim::layer_send(io, plan, values);
}

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace ttnn::experimental::prim {

inline std::vector<tt::tt_metal::CoreCoord> order_cores_by_optimal_dram(
    std::vector<tt::tt_metal::CoreCoord> cores, tt::tt_metal::IDevice* device, uint8_t noc = 0) {
    if (cores.empty()) {
        return cores;
    }
    auto* mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
    if (mesh_device == nullptr) {
        return cores;
    }
    auto optimal_ordered =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(static_cast<tt::tt_metal::NOC>(noc));
    std::unordered_map<uint32_t, uint32_t> core_to_priority;
    for (uint32_t i = 0; i < optimal_ordered.size(); ++i) {
        const auto& c = optimal_ordered[i];
        core_to_priority[c.x + (c.y << 16)] = i;
    }
    std::sort(
        cores.begin(),
        cores.end(),
        [&core_to_priority](const tt::tt_metal::CoreCoord& a, const tt::tt_metal::CoreCoord& b) {
            uint32_t pri_a =
                core_to_priority.count(a.x + (a.y << 16)) ? core_to_priority.at(a.x + (a.y << 16)) : 0xFFFFFFFFu;
            uint32_t pri_b =
                core_to_priority.count(b.x + (b.y << 16)) ? core_to_priority.at(b.x + (b.y << 16)) : 0xFFFFFFFFu;
            return pri_a < pri_b;
        });
    return cores;
}

}  // namespace ttnn::experimental::prim

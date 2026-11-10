// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

namespace ttnn::ccl {

// Returns compute() for this mesh and query argument, computing it once.
//
// For facts about how a mesh is wired into the fabric (links per axis, whether an axis is a straight line or closes
// into a ring). CCL ops ask them on every call, and asking walks the mesh's rows and the fabric's routing tables:
// ~20 us per all_gather on a 32-device Galaxy, all of it before the op's first device command. The wiring is fixed
// while the mesh is open (the fabric is configured before it opens). The key holds the mesh id, never reused within a
// process, and the view's shape, the one thing a reshape changes. Each call site (each Compute type) has its own
// cache.
template <typename Compute>
auto cached_mesh_wiring(const tt::tt_metal::distributed::MeshDevice& mesh_device, int64_t query, Compute&& compute) {
    using Value = std::decay_t<decltype(compute())>;
    using Key = std::tuple<int, std::vector<uint32_t>, int64_t>;
    static std::mutex mutex;
    static std::map<Key, Value> cache;

    const auto& shape = mesh_device.get_view().shape();
    Key key{mesh_device.id(), std::vector<uint32_t>(shape.cbegin(), shape.cend()), query};
    {
        std::lock_guard lock(mutex);
        if (auto it = cache.find(key); it != cache.end()) {
            return it->second;
        }
    }
    Value value = std::forward<Compute>(compute)();
    std::lock_guard lock(mutex);
    return cache.emplace(std::move(key), std::move(value)).first->second;
}

}  // namespace ttnn::ccl

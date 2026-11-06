// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_pinned.hpp"

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

// tt_metal internals (this file gets tt_metal's include directories: CMakeLists.txt)
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/impl/context/metal_env_impl.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

size_t pinned_memory_limit(tt::tt_metal::distributed::MeshDevice* mesh) {
    TT_FATAL(mesh != nullptr, "pinned_memory_limit: no mesh device");
    return mesh->impl().metal_env().get_rtoptions().get_pinned_memory_cache_limit_bytes();
}

size_t set_pinned_memory_limit(tt::tt_metal::distributed::MeshDevice* mesh, size_t limit_bytes) {
    TT_FATAL(mesh != nullptr, "set_pinned_memory_limit: no mesh device");
    auto& rt = mesh->impl().metal_env().get_rtoptions();
    const size_t prev = rt.get_pinned_memory_cache_limit_bytes();
    rt.set_pinned_memory_cache_limit_bytes(limit_bytes);
    return prev;
}

size_t release_pinned(const ttnn::Tensor& host_tensor) {
    TT_FATAL(host_tensor.storage_type() == ttnn::StorageType::HOST, "release_pinned: the tensor must be on host");
    size_t bytes = 0;
    auto& cache = tt::tt_metal::experimental::PinnedMemoryCache::instance();
    host_tensor.host_storage().buffer().apply([&](const tt::tt_metal::HostBuffer& shard) {
        const auto view = shard.view_bytes();
        if (view.empty()) {
            return;
        }
        cache.release(static_cast<const void*>(view.data()));
        bytes += view.size();
    });
    return bytes;
}

size_t pinned_cache_entries() { return tt::tt_metal::experimental::PinnedMemoryCache::instance().num_entries(); }

}  // namespace ttnn::operations::bringup::flat_routed_expert

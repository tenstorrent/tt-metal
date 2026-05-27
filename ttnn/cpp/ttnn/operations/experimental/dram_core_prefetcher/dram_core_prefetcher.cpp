// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_core_prefetcher.hpp"

#include <tt-metalium/experimental/dram_core_prefetcher.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace ttnn::operations::experimental {

void start_dram_core_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const GlobalCircularBuffer& global_cb,
    bool enable_performance_mode) {
    std::vector<const tt::tt_metal::MeshTensor*> mesh_tensors;
    mesh_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        mesh_tensors.push_back(&t.mesh_tensor());
    }
    tt::tt_metal::experimental::DramCorePrefetcherConfig config{
        .num_layers = num_layers,
        .enable_performance_mode = enable_performance_mode,
    };
    tt::tt_metal::experimental::StartDramCorePrefetcher(mesh_device, mesh_tensors, global_cb, config);
}

void stop_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StopDramCorePrefetcher(mesh_device);
}

}  // namespace ttnn::operations::experimental

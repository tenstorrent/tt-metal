// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_core_prefetcher.hpp"

#include <tt-metalium/experimental/dram_core_prefetcher.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace ttnn::operations::experimental {

void start_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device, bool enable_performance_mode) {
    tt::tt_metal::experimental::DramCorePrefetcherConfig config{
        .enable_performance_mode = enable_performance_mode,
    };
    tt::tt_metal::experimental::StartDramCorePrefetcher(mesh_device, config);
}

void queue_dram_core_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset) {
    std::vector<const tt::tt_metal::MeshTensor*> mesh_tensors;
    mesh_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        mesh_tensors.push_back(&t.mesh_tensor());
    }
    tt::tt_metal::experimental::QueueDramCorePrefetcherRequest(
        mesh_device, global_cb, device_subset, mesh_tensors, num_layers);
}

void stop_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StopDramCorePrefetcher(mesh_device);
}

}  // namespace ttnn::operations::experimental

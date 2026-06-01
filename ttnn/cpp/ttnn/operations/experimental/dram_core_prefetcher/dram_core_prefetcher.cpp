// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_core_prefetcher.hpp"

#include <tt-metalium/experimental/dram_core_prefetcher.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace ttnn::operations::experimental {

void start_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StartDramCorePrefetcher(*mesh_device, {});
}

void queue_dram_core_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<ttnn::Tensor, uint32_t>>& tensors,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset) {
    std::vector<tt::tt_metal::experimental::DramCorePrefetcherInput> inputs;
    inputs.reserve(tensors.size());
    for (const auto& [tensor, block_count] : tensors) {
        inputs.push_back({&tensor.mesh_tensor(), block_count});
    }
    tt::tt_metal::experimental::QueueDramCorePrefetcherRequest(*mesh_device, global_cb, device_subset, inputs);
}

void stop_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StopDramCorePrefetcher(*mesh_device);
}

}  // namespace ttnn::operations::experimental

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher.hpp"

#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace ttnn::operations::experimental {

bool is_tensor_prefetcher_supported(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    return tt::tt_metal::experimental::IsTensorPrefetcherSupported(*mesh_device);
}

void start_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device, bool dual_senders_per_bank) {
    tt::tt_metal::experimental::StartTensorPrefetcher(*mesh_device, {.dual_senders_per_bank = dual_senders_per_bank});
}

void queue_tensor_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<TensorPrefetcherQueueTensor>& tensors,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset,
    std::optional<uint8_t> cq_id) {
    std::vector<tt::tt_metal::experimental::TensorPrefetcherInput> inputs;
    inputs.reserve(tensors.size());
    for (const auto& item : tensors) {
        // (tensor, block_count) defaults to batched (empty rotation); (tensor, block_count,
        // rotation) supplies the per-receiver streaming rotation table for that tensor.
        if (const auto* pair = std::get_if<std::pair<ttnn::Tensor, uint32_t>>(&item)) {
            inputs.push_back({pair->first.mesh_tensor(), pair->second, /*rotation=*/{}});
        } else {
            const auto& [tensor, block_count, rotation] =
                std::get<std::tuple<ttnn::Tensor, uint32_t, std::vector<uint32_t>>>(item);
            inputs.push_back({tensor.mesh_tensor(), block_count, rotation});
        }
    }
    tt::tt_metal::experimental::QueueTensorPrefetcherRequest(*mesh_device, global_cb, device_subset, inputs, cq_id);
}

void wait_for_cq_on_tensor_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint8_t cq_id,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset) {
    tt::tt_metal::experimental::WaitForCqOnTensorPrefetcher(*mesh_device, cq_id, device_subset);
}

void stop_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StopTensorPrefetcher(*mesh_device);
}

}  // namespace ttnn::operations::experimental

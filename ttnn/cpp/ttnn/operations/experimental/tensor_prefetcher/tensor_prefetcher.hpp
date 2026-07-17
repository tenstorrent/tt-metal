// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental {

// One tensor to prefetch: either (tensor, block_count) or (tensor, block_count, rotation).
// block_count is the number of K-blocks to divide the tensor's K dimension into. rotation
// (receiver-contiguous layout only; omitted/empty == batched) is the per-receiver streaming
// ring-rotation table, indexed by global ring position: it delivers that tensor's K-blocks in
// host-specified ring-rotated FIFO order for a matching stream_in1 matmul. See
// TensorPrefetcherInput for the rotation contract.
using TensorPrefetcherQueueTensor =
    std::variant<std::pair<ttnn::Tensor, uint32_t>, std::tuple<ttnn::Tensor, uint32_t, std::vector<uint32_t>>>;

// Thin ttnn-side wrappers around the queueable
// tt::tt_metal::experimental::Start/Queue/Stop TensorPrefetcher API.
//
// Lifecycle:
//   1. start_tensor_prefetcher(device)
//      - Builds the long-running DRISC kernel on every DRAM sender core and
//        spawns the host worker thread. Returns immediately. Receiver count
//        is per-GCB (read from each GCB's sender state block on every
//        request), so a single prefetcher can serve GCBs with different
//        num_receivers values.
//   2. queue_tensor_prefetcher_request(device, tensors, global_cb, device_subset=None)
//      - Push one request. `tensors` is the full, flattened list of weights (at
//        least one), streamed in list order; each item is (weight, block_count)
//        or (weight, block_count, rotation) (see TensorPrefetcherQueueTensor).
//        block_count is the number of K-blocks to divide that tensor's K
//        dimension into. Pass distinct tensors for distinct layers, or repeat a
//        tensor to replay it. device_subset defaults to the full mesh.
//   3. stop_tensor_prefetcher(device)
//      - Sends the stop sentinel, joins the worker, waits for the kernels
//        to exit. Caller must call this before destroying the device.
// Returns true if the Tensor prefetcher is supported on `mesh_device`
// (programmable DRAM cores are available). Use this to skip rather than fail
// when start_tensor_prefetcher would otherwise raise.
bool is_tensor_prefetcher_supported(tt::tt_metal::distributed::MeshDevice* mesh_device);

void start_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device);

void queue_tensor_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<TensorPrefetcherQueueTensor>& tensors,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset = std::nullopt,
    std::optional<uint8_t> cq_id = std::nullopt);

// Fence the prefetcher against command queue `cq_id`: every prefetch request queued
// after this call waits until all work previously enqueued on `cq_id` has completed
// on device before the prefetcher reads DRAM. Call after the data writes and before
// the dependent queue_tensor_prefetcher_request.
void wait_for_cq_on_tensor_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint8_t cq_id,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset = std::nullopt);

void stop_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device);

}  // namespace ttnn::operations::experimental

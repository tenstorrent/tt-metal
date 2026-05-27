// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental {

// Thin ttnn-side wrappers around the queueable
// tt::tt_metal::experimental::Start/Queue/Stop DramCorePrefetcher API.
//
// Lifecycle:
//   1. start_dram_core_prefetcher(device)
//      - Builds the long-running DRISC kernel on every DRAM sender core and
//        spawns the host worker thread. Returns immediately. Receiver count
//        is per-GCB (read from each GCB's sender state block on every
//        request), so a single prefetcher can serve GCBs with different
//        num_receivers values.
//   2. queue_dram_core_prefetcher_request(device, tensors, num_layers, global_cb, device_subset=None)
//      - Push one request. `tensors` is the list of weight tensors to prefetch
//        (at least one). device_subset defaults to the full mesh.
//   3. stop_dram_core_prefetcher(device)
//      - Sends the stop sentinel, joins the worker, waits for the kernels
//        to exit. Caller must call this before destroying the device.
void start_dram_core_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device, bool enable_performance_mode = false);

void queue_dram_core_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset = std::nullopt);

void stop_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device);

}  // namespace ttnn::operations::experimental

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_to_device_filtered.hpp"

#include "ttnn/graph/graph_serialization.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/experimental/core_subset_write/tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/queue_id.hpp>

#include <tracy/Tracy.hpp>

namespace ttnn::experimental::core_subset_write {

void copy_to_device_filtered(
    const tt::tt_metal::Tensor& host_tensor,
    tt::tt_metal::Tensor& device_tensor,
    const tt::tt_metal::CoreRangeSet& logical_core_filter,
    std::optional<tt::tt_metal::QueueId> cq_id) {
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "ttnn::experimental::core_subset_write::copy_to_device_filtered", host_tensor, device_tensor, cq_id);
    auto& cq = device_tensor.device()->mesh_command_queue(tt::tt_metal::raw_optional(cq_id));
    if (tt::tt_metal::is_uniform_write(host_tensor.host_tensor(), *device_tensor.device())) {
        tt::tt_metal::experimental::core_subset_write::enqueue_write_tensor(
            cq, host_tensor.host_tensor(), device_tensor.device_storage().get_mesh_tensor(), logical_core_filter);
    } else {
        auto coords = tt::tt_metal::experimental::core_subset_write::enqueue_write_tensor_non_uniform(
            cq, host_tensor.host_tensor(), device_tensor.device_storage().get_mesh_tensor(), logical_core_filter);
        device_tensor.device_storage() = tt::tt_metal::DeviceStorage(device_tensor.device_storage(), std::move(coords));
    }
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    tt::tt_metal::GraphTracker::instance().track_function_end(device_tensor);
}

}  // namespace ttnn::experimental::core_subset_write

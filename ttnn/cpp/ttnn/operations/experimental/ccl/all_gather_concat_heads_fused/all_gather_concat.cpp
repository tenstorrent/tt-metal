// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    bool on_subcoregrids = false;
    if (input_tensor.is_sharded()) {
        const auto& input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        if (input_core_ranges.size() > 1 || !(input_core_ranges[0].start_coord == CoreCoord{0, 0})) {
            on_subcoregrids = true;
        }
    }
    return ttnn::operations::experimental::ccl::all_gather_concat(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode,
        on_subcoregrids);
}

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    return invoke(
        ttnn::DefaultQueueId,
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode);
}

}  // namespace ttnn::operations::experimental::ccl

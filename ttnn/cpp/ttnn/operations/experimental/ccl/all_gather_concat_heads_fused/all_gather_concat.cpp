// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include "llrt/tt_cluster.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    const std::array<uint32_t, 2> shard_shape = {32, 128};

    auto core_range_1 = CoreRange(CoreCoord{1, 0}, CoreCoord{3, 9});
    auto core_range_2 = CoreRange(CoreCoord{5, 0}, CoreCoord{6, 0});
    auto shard_spec =
        ShardSpec(CoreRangeSet(std::vector{core_range_1, core_range_2}), shard_shape, ShardOrientation::ROW_MAJOR);
    auto inter_output_mem_config =
        ttnn::MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::L1, shard_spec);

    return ttnn::operations::experimental::ccl::all_gather_concat(
        input_tensor,
        buffer_tensor,
        dim,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode);
}

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
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
        buffer_tensor,
        dim,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode);
}

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    return ttnn::operations::experimental::ccl::all_gather_concat(
        input_tensor,
        buffer_tensor,
        dim,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode);
}

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    return invoke(
        ttnn::DefaultQueueId,
        input_tensor,
        buffer_tensor,
        dim,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        enable_persistent_fabric_mode);
}

}  // namespace ttnn::operations::experimental::ccl

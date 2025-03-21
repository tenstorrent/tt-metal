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

    auto shape = input_tensor.get_padded_shape();
    shape[dim] *= 4;
    auto output_intermediate_tensor_spec = TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), inter_output_mem_config));

    std::unordered_map<chip_id_t, Tensor> temp_tensor_map;
    auto all_devices = input_tensor.get_workers();
    for (auto d : tt::Cluster::instance().user_exposed_chip_ids()) {
        for (auto dev : all_devices) {
            if (dev->id() == d) {
                auto input_buffer =
                    tt::tt_metal::tensor_impl::allocate_buffer_on_device(dev, output_intermediate_tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor temp_tensor = Tensor(input_storage, output_intermediate_tensor_spec);
                temp_tensor_map[d] = temp_tensor;
            }
        }
    }
    return ttnn::operations::experimental::ccl::all_gather_concat(
        input_tensor,
        dim,
        temp_tensor_map,
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

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
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
    const std::array<uint32_t, 2> shard_shape = {32, 128};
    auto shard_spec =
        ShardSpec(CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{7, 3})), shard_shape, ShardOrientation::ROW_MAJOR);
    auto inter_output_mem_config =
        ttnn::MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::L1, shard_spec);

    auto shape = input_tensor.get_padded_shape();
    shape[dim] *= 4;
    auto output_intermediate_tensor_spec = TensorSpec(
        shape,
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_tensor_spec().page_config(), inter_output_mem_config));

    std::unordered_map<chip_id_t, Tensor> temp_tensor_map;
    auto all_devices = input_tensor.get_workers();
    for (auto d : tt::Cluster::instance().user_exposed_chip_ids()) {
        for (auto dev : all_devices) {
            if (dev->id() == d) {
                auto input_buffer =
                    tt::tt_metal::tensor_impl::allocate_buffer_on_device(dev, output_intermediate_tensor_spec);
                auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
                Tensor temp_tensor = Tensor(input_storage, output_intermediate_tensor_spec);
                temp_tensor_map[d] = temp_tensor;
            }
        }
    }
    return ttnn::operations::experimental::ccl::all_gather_concat(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        temp_tensor_map,
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

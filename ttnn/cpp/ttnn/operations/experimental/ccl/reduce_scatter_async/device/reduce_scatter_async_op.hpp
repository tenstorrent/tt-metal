// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
struct ReduceScatterAsync {
    ReduceScatterAsync(
        const ttnn::operations::binary::BinaryOpType binary_op_type,
        const uint32_t scatter_dim,
        const uint32_t ring_size,
        const uint32_t ring_index,
        const std::optional<IDevice*> forward_device,
        const std::optional<IDevice*> backward_device,
        const MemoryConfig& output_mem_config,
        const ttnn::ccl::Topology topology,
        std::optional<std::vector<Tensor>>& foreward_output_tensors,
        std::optional<std::vector<Tensor>>& backward_output_tensors,
        std::optional<size_t> num_links_preferred,
        const GlobalSemaphore& from_remote_sem,
        const GlobalSemaphore& to_remote_sem,
        std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle) :
        binary_op_type(binary_op_type),
        scatter_dim(scatter_dim),
        ring_size(ring_size),
        ring_index(ring_index),
        forward_device(forward_device),
        backward_device(backward_device),
        output_mem_config(output_mem_config),
        topology(topology),
        foreward_output_tensors(foreward_output_tensors),
        backward_output_tensors(backward_output_tensors),
        num_links_preferred(num_links_preferred),
        from_remote_sem(from_remote_sem),
        to_remote_sem(to_remote_sem),
        fabric_handle(fabric_handle),
        sub_device_id(sub_device_id) {}

    const ttnn::operations::binary::BinaryOpType binary_op_type;
    const uint32_t scatter_dim;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<IDevice*> forward_device;
    const std::optional<IDevice*> backward_device;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    // const
    std::optional<std::vector<Tensor>> foreward_output_tensors;
    std::optional<std::vector<Tensor>> backward_output_tensors;
    std::optional<size_t> num_links_preferred;
    const GlobalSemaphore from_remote_sem;
    const GlobalSemaphore to_remote_sem;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;

        attrs.emplace_back("binary_op_type", binary_op_type);
        attrs.emplace_back("dim", scatter_dim);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("ring_index", ring_index);
        attrs.emplace_back("forward_device", forward_device);
        attrs.emplace_back("backward_device", backward_device);
        attrs.emplace_back("num_links_preferred", num_links_preferred);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("topology", topology);

        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

namespace ccl {
namespace reduce_scatter_detail {
tt::tt_metal::operation::ProgramWithCallbacks build_reduce_scatter_async_program(
    const Tensor& input_tensor,
    Tensor& local_output_tensor,
    Tensor& input_tensor_from_remote_forward_direction,
    Tensor& input_tensor_from_remote_backward_direction,
    Tensor& partial_output_tensor_to_forward_direction,
    Tensor& partial_output_tensor_to_backward_direction,
    std::optional<Tensor>& foreward_direction_remote_output_tensor,
    std::optional<Tensor>& backward_direction_remote_output_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    ttnn::operations::binary::BinaryOpType reduce_op,
    const uint32_t dim,
    const uint32_t line_size,
    const uint32_t line_index,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_links_preferred,
    const GlobalSemaphore& from_remote_sem,
    const GlobalSemaphore& to_remote_sem,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle);
}
};  // namespace ccl

namespace ccl {
namespace reduce_scatter_detail {
ReduceScatterAsync create_reduce_scatter_struct(
    const Tensor& input_tensor,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    const uint32_t dim,
    const MemoryConfig& output_mem_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    std::optional<std::vector<Tensor>> foreward_output_tensors,
    std::optional<std::vector<Tensor>> backward_output_tensors,
    std::optional<size_t> num_links_preferred,
    const std::vector<GlobalSemaphore>& from_remote_sems,
    const std::vector<GlobalSemaphore>& to_remote_sems,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle);
}  // namespace reduce_scatter_detail
}  // namespace ccl

namespace operations {
namespace experimental {
namespace ccl {
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    const std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt,   // TODO make reference
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle = std::nullopt);  // TODO make reference
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& from_remote_multi_device_global_semaphore,
    const global_semaphore::MultiDeviceGlobalSemaphore& to_remote_multi_device_global_semaphore,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    const std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt = std::nullopt,   // TODO make reference
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle = std::nullopt);  // TODO make reference

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn

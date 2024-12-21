// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "sub_device/sub_device_types.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

#include "tt_metal/impl/buffers/global_semaphore.hpp"

namespace ttnn {
struct ReduceScatterAsync {
    ReduceScatterAsync(
        const ttnn::operations::binary::BinaryOpType binary_op_type,
        const uint32_t scatter_dim,
        const uint32_t ring_size,
        const uint32_t ring_index,
        const std::optional<Device*> forward_device,
        const std::optional<Device*> backward_device,
        const MemoryConfig& output_mem_config,
        const ttnn::ccl::Topology topology,
        std::optional<std::vector<Tensor>>& foreward_output_tensors,
        std::optional<std::vector<Tensor>>& backward_output_tensors,
        std::optional<size_t> num_links_preferred,
        const GlobalSemaphore& from_remote_sem,
        const GlobalSemaphore& to_remote_sem,
        std::unordered_map<chip_id_t, SubDeviceId>& sub_device_id_map,
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
        sub_device_id_map(sub_device_id_map) {
        TT_FATAL(sub_device_id_map.size() > 0, "Reduce scatter async was given an uninitialized subdevice ID");
    }

    const ttnn::operations::binary::BinaryOpType binary_op_type;
    const uint32_t scatter_dim;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const std::optional<Device*> forward_device;
    const std::optional<Device*> backward_device;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    // const
    std::optional<std::vector<Tensor>> foreward_output_tensors;
    std::optional<std::vector<Tensor>> backward_output_tensors;
    std::optional<size_t> num_links_preferred;
    const GlobalSemaphore from_remote_sem;
    const GlobalSemaphore to_remote_sem;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle;
    std::unordered_map<chip_id_t, SubDeviceId>& sub_device_id_map;

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
        // attrs.emplace_back("from_remote_sem", from_remote_sem);
        // attrs.emplace_back("to_remote_sem", to_remote_sem);

        return attrs;
    }

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};


namespace ccl {
namespace reduce_scatter_detail {
operation::ProgramWithCallbacks build_reduce_scatter_async_program(
    Tensor const& input_tensor,
    Tensor& local_output_tensor,
    Tensor& input_tensor_from_remote_forward_direction,
    Tensor& input_tensor_from_remote_backward_direction,
    Tensor& partial_output_tensor_to_forward_direction,
    Tensor& partial_output_tensor_to_backward_direction,
    std::optional<Tensor>& foreward_direction_remote_output_tensor,
    std::optional<Tensor>& backward_direction_remote_output_tensor,
    std::optional<Device*> forward_device,
    std::optional<Device*> backward_device,
    ttnn::operations::binary::BinaryOpType reduce_op,
    const uint32_t dim,
    const uint32_t line_size,
    const uint32_t line_index,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_links_preferred,
    GlobalSemaphore const& from_remote_sems,
    GlobalSemaphore const& to_remote_sems,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle);
}
}; // namespace ccl

namespace ccl{
namespace reduce_scatter_detail{
ReduceScatterAsync create_reduce_scatter_struct(
    const Tensor& input_tensor,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    const uint32_t dim,
    const MemoryConfig& output_mem_config,
    const std::vector<Device*>& devices,
    const ttnn::ccl::Topology topology,
    std::optional<std::vector<Tensor>> foreward_output_tensors,
    std::optional<std::vector<Tensor>> backward_output_tensors,
    std::optional<size_t> num_links_preferred,
    std::vector<GlobalSemaphore> const& from_remote_sems,
    std::vector<GlobalSemaphore> const& to_remote_sems,
    std::unordered_map<chip_id_t, SubDeviceId>& sub_device_id_map,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& fabric_handle);
} // namespace reduce_scatter_detail
} // namespace ccl

namespace operations{
namespace experimental {
namespace ccl{
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    const std::optional<size_t> num_preferred_links = std::nullopt,
    std::unordered_map<chip_id_t, SubDeviceId> sub_device_id_map = {},                 // TODO make reference
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle = std::nullopt);  // TODO make reference

Tensor reduce_scatter(
    const ttnn::Tensor &input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op = ttnn::operations::reduction::ReduceType::Sum,
    const std::optional<ttnn::MemoryConfig>& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    const std::optional<size_t> num_preferred_links = std::nullopt
    );
} // namespace ccl
} // namespace experimental
} // namespace operations

}  // namespace ttnn

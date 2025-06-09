// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"

#include <cstdint>

namespace ttnn {

void ReduceScatter::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.padded_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
        TT_FATAL(
            t.padded_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
    }
}

std::vector<ttnn::TensorSpec> ReduceScatter::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.logical_shape();
    TT_FATAL(
        shape[this->scatter_dim] % this->ring_size == 0,
        "The size of the scatter dimension must be a multiple of the ring size. Dimension size: {}, ring Size: {}",
        shape[this->scatter_dim],
        this->ring_size);
    shape[this->scatter_dim] /= this->ring_size;
    TensorSpec spec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), output_mem_config));
    return std::vector<ttnn::TensorSpec>(input_tensors.size(), spec);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks ReduceScatter::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks ReduceScatter::create_program_at(
    const MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    uint32_t device_index = 0;
    std::optional<chip_id_t> receiver_device_id;
    std::optional<chip_id_t> sender_device_id;
    auto target_device = input_tensors.at(0).mesh_device() ? input_tensors.at(0).mesh_device()->get_device(mesh_coord)
                                                           : input_tensors.at(0).device();
    ccl::SenderRecieverConfig config =
        this->cluster_axis.has_value()
            ? ccl::get_device_sender_receiver_config_in_ring(mesh_coord, mesh_device, *cluster_axis, ring_size)
            : ccl::get_device_sender_receiver_config(target_device, this->devices, topology);

    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        config.device_index,
        target_device->id(),
        config.receiver_device_id,
        config.sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(
    ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum: return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("Reduce scatter only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace operations::ccl {

namespace {
Tensor reduce_scatter_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "reduce_scatter op is only supported for Fast Dispatch");

    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "reduce_scatter op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int16_t rank = input_tensor.logical_shape().rank();

    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        scatter_dim >= -rank && scatter_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    return tt::tt_metal::operation::run(
               ttnn::ReduceScatter{
                   binary_op_type,
                   scatter_dim,
                   num_links,
                   num_devices,
                   output_mem_config,
                   ccl_topology,
                   user_defined_num_workers,
                   user_defined_num_buffers_per_channel,
                   /*cluster_axis=*/std::nullopt,
                   std::move(devices)},
               {input_tensor})
        .at(0);
}
Tensor reduce_scatter_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(reduce_op);

    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int16_t rank = input_tensor.logical_shape().rank();

    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        scatter_dim >= -rank && scatter_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    return tt::tt_metal::operation::run(
               ttnn::ReduceScatter{
                   binary_op_type,
                   scatter_dim,
                   num_links,
                   num_devices,
                   output_mem_config.value_or(input_tensor.memory_config()),
                   topology,
                   user_defined_num_workers,
                   user_defined_num_buffers_per_channel,
                   /*cluster_axis=*/cluster_axis,
                   /*devices=*/{},
                   &mesh_device},
               {input_tensor})
        .at(0);
}
}  // namespace

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    return reduce_scatter_impl(
        input_tensor,
        dim,
        math_op,
        num_links,
        output_mem_config,
        topology,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        ttnn::ccl::get_active_physical_devices(input_tensor));
}

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor>& input_tensors,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    std::vector<IDevice*> devices;
    devices.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        devices.push_back(input_tensor.device());
    }
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        output_tensors.push_back(reduce_scatter_impl(
            input_tensor,
            dim,
            math_op,
            num_links,
            output_mem_config,
            topology,
            user_defined_num_workers,
            user_defined_num_buffers_per_channel,
            devices));
    }
    return output_tensors;
}

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    return reduce_scatter_impl(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        reduce_op,
        num_links,
        output_mem_config,
        topology,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel);
}

std::vector<Tensor> reduce_scatter(
    const std::vector<Tensor>& input_tensors,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        output_tensors.push_back(reduce_scatter_impl(
            input_tensor,
            dim,
            cluster_axis,
            mesh_device,
            reduce_op,
            num_links,
            output_mem_config,
            topology,
            user_defined_num_workers,
            user_defined_num_buffers_per_channel));
    }
    return output_tensors;
}

}  // namespace operations::ccl

};  // namespace ttnn

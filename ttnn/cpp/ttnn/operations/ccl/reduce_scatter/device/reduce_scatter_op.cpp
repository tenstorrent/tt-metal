// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"

#include <cstdint>

namespace ttnn {

void ReduceScatter::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.get_padded_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
        TT_FATAL(
            t.get_padded_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
    }
}

std::vector<ttnn::TensorSpec> ReduceScatter::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.get_logical_shape();
    TT_FATAL(
        shape[this->scatter_dim] % this->ring_size == 0,
        "The size of the scatter dimension must be a multiple of the ring size. Dimension size: {}, ring Size: {}",
        shape[this->scatter_dim],
        this->ring_size);
    shape[this->scatter_dim] /= this->ring_size;
    TensorSpec spec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), output_mem_config));
    return std::vector<ttnn::TensorSpec>(input_tensors.size(), spec);
}

tt::tt_metal::operation::ProgramWithCallbacks ReduceScatter::create_program_at(
    const MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    uint32_t device_index = 0;
    std::optional<chip_id_t> sender_device_id;
    std::optional<chip_id_t> receiver_device_id;

    const auto& mesh_device = input_tensors[0].mesh_device();

    if (this->cluster_axis.has_value()) {
        auto mesh_view = mesh_device->get_view();
        TT_FATAL(
            mesh_view.is_mesh_2d(),
            "reduce-scatter invoked with cluster_axis API on >2D mesh, which is currently unsupported");
        const auto view_index = (cluster_axis == 0) ? mesh_coord[1] : mesh_coord[0];
        device_index = (cluster_axis == 0) ? mesh_coord[0] : mesh_coord[1];

        auto get_chip_id = [&](std::size_t line_index) -> std::optional<chip_id_t> {
            auto new_row = mesh_coord[0];
            auto new_col = mesh_coord[1];
            if (cluster_axis == 0) {
                new_row = line_index % this->num_links;
            } else {
                new_col = line_index % this->num_links;
            }
            return mesh_view.find_device_id(MeshCoordinate(new_row, new_col));
        };

        bool is_last_chip_in_clockwise_direction = device_index == (this->num_links - 1);
        bool is_last_chip_in_counter_clockwise_direction = device_index == 0;
        receiver_device_id = is_last_chip_in_clockwise_direction ? std::nullopt : get_chip_id(device_index + 1);
        sender_device_id = is_last_chip_in_counter_clockwise_direction
                               ? std::nullopt
                               : get_chip_id(device_index + this->num_links - 1);

    } else {
        std::tie(device_index, sender_device_id, receiver_device_id) = ccl::get_device_index_and_sender_receiver_ids(
            mesh_device->get_device(mesh_coord), this->devices, this->topology);

        TT_FATAL(
            receiver_device_id != std::nullopt || sender_device_id != std::nullopt,
            "Error, Reduce-scatter was unable to identify either a sender or receiver device ID and atleast one must "
            "be identified for a valid Reduce-scatter configuration. The input mesh tensor or Reduce-scatter arguments "
            "may be incorrect");
    }
    chip_id_t target_device_id = mesh_device->get_device(mesh_coord)->id();

    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        device_index,
        target_device_id,
        receiver_device_id,
        sender_device_id,
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

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "reduce_scatter op is only supported for Fast Dispatch");

    std::vector<IDevice*> devices = input_tensor.active_physical_devices();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "reduce_scatter op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int16_t rank = input_tensor.get_logical_shape().rank();

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

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device, /* TODO: This needs to be removed, since the input_tensor has a mesh_device */
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

    int16_t rank = input_tensor.get_logical_shape().rank();

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
                   /*devices=*/{}},
               {input_tensor})
        .at(0);
}

}  // namespace operations::ccl

};  // namespace ttnn

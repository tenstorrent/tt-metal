// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
void OffsetCumsumDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& input_tensor) {
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::UINT32, "Only UINT32 is supported for inputs!");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");
    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(
        input_shape.size() == 2,
        "Input tensor must be 2D [num_devices, n_routed_experts], got {} dimensions",
        input_shape.size());
    TT_FATAL(input_shape[-2] > 0, "H (num_devices) must be > 0, got {}", input_shape[-2]);
    TT_FATAL(input_shape[-1] > 0, "W (n_routed_experts) must be > 0, got {}", input_shape[-1]);
}

OffsetCumsumDeviceOperation::spec_return_value_t OffsetCumsumDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& input_tensor) {
    const auto& logical_shape = input_tensor.logical_shape();
    uint32_t W = logical_shape[-1];

    auto layout = tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::UINT32,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

    auto offsets_spec = TensorSpec(ttnn::Shape({1, W}), layout);
    auto totals_spec = TensorSpec(ttnn::Shape({1, W}), layout);
    return {offsets_spec, totals_spec};
}

OffsetCumsumDeviceOperation::topology_return_value_t OffsetCumsumDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*args*/, const tensor_args_t& input_tensor) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

    const auto& input_topology = input_tensor.tensor_topology();
    const auto& dist_shape = input_topology.distribution_shape();
    size_t ndims = dist_shape.dims();

    // Both outputs are unique per device on all mesh dimensions:
    //  - Along cluster_axis: each device holds a different row of the prefix sum
    //  - Along other axes: masked_bincount uses per-dispatch-group expert masks,
    //    so histograms (and therefore cumsums/totals) differ across dispatch groups
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    for (size_t i = 0; i < ndims; i++) {
        placements.push_back(Shard{static_cast<int>(i)});
    }

    auto offsets_topology = tt::tt_metal::TensorTopology(dist_shape, placements, input_topology.mesh_coords());
    auto totals_topology = tt::tt_metal::TensorTopology(dist_shape, placements, input_topology.mesh_coords());

    return {offsets_topology, totals_topology};
}

tt::stl::hash::hash_t OffsetCumsumDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    const auto& input_shape = input_tensor.padded_shape();
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<OffsetCumsumDeviceOperation>(
        args, input_tensor.dtype(), input_tensor.memory_config(), input_shape);
    return hash;
}

OffsetCumsumDeviceOperation::tensor_return_value_t OffsetCumsumDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    auto output_specs = compute_output_specs(args, input_tensor);
    auto offsets_tensor = create_device_tensor(output_specs[0], input_tensor.device());
    auto totals_tensor = create_device_tensor(output_specs[1], input_tensor.device());
    return {offsets_tensor, totals_tensor};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 2> offset_cumsum(const Tensor& input_tensor, uint32_t cluster_axis) {
    using OperationType = ttnn::experimental::prim::OffsetCumsumDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{cluster_axis};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, input_tensor);
}

}  // namespace ttnn::prim

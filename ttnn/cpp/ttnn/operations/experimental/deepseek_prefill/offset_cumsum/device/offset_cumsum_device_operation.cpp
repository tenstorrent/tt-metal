// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
void OffsetCumsumDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
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
    TT_FATAL(
        args.experts_per_chip > 0 && input_shape[-1] % args.experts_per_chip == 0,
        "n_routed_experts ({}) must be divisible by experts_per_chip ({})",
        input_shape[-1],
        args.experts_per_chip);
    const size_t ndims = input_tensor.tensor_topology().distribution_shape().dims();
    TT_FATAL(
        args.cluster_axis < ndims,
        "cluster_axis ({}) is out of range for a {}-dimensional distribution",
        args.cluster_axis,
        ndims);
}

OffsetCumsumDeviceOperation::spec_return_value_t OffsetCumsumDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& input_tensor) {
    const auto& logical_shape = input_tensor.logical_shape();
    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];

    auto layout = tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::UINT32,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

    auto offsets_spec = tt::tt_metal::TensorSpec(ttnn::Shape({1, W}), layout);
    auto totals_spec = tt::tt_metal::TensorSpec(ttnn::Shape({1, W}), layout);
    auto expert_region_spec = tt::tt_metal::TensorSpec(ttnn::Shape({1, W}), layout);
    auto all_offsets_spec = tt::tt_metal::TensorSpec(ttnn::Shape({H, W}), layout);
    return {offsets_spec, totals_spec, expert_region_spec, all_offsets_spec};
}

OffsetCumsumDeviceOperation::topology_return_value_t OffsetCumsumDeviceOperation::compute_output_topologies(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    const auto& input_topology = input_tensor.tensor_topology();
    const auto& dist_shape = input_topology.distribution_shape();
    size_t ndims = dist_shape.dims();

    // Along axes other than cluster_axis every output is genuinely per-device: masked_bincount uses
    // per-dispatch-group expert masks, so histograms -- and everything derived from them -- differ
    // across dispatch groups.
    //
    // Along cluster_axis only `offsets` differs, since it is that device's row of the prefix sum.
    // `totals` and `expert_region` are computed from the all-gathered histogram and so are identical
    // across the group; they are nonetheless declared Shard here, which is what callers that read
    // them per device already expect. Changing that is a separate call.
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    for (size_t i = 0; i < ndims; i++) {
        placements.push_back(Shard{static_cast<int>(i)});
    }

    auto offsets_topology = tt::tt_metal::TensorTopology(dist_shape, placements, input_topology.mesh_coords());
    auto totals_topology = tt::tt_metal::TensorTopology(dist_shape, placements, input_topology.mesh_coords());
    auto expert_region_topology = tt::tt_metal::TensorTopology(dist_shape, placements, input_topology.mesh_coords());

    // all_offsets holds every row of the prefix sum, so unlike `offsets` it is the same tensor on
    // every device along cluster_axis, and is declared as such: a consumer reads the whole table.
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> all_placements = placements;
    all_placements[args.cluster_axis] = Replicate{};
    auto all_offsets_topology = tt::tt_metal::TensorTopology(dist_shape, all_placements, input_topology.mesh_coords());

    return {offsets_topology, totals_topology, expert_region_topology, all_offsets_topology};
}

OffsetCumsumDeviceOperation::tensor_return_value_t OffsetCumsumDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& input_tensor) {
    auto output_specs = compute_output_specs(args, input_tensor);
    auto offsets_tensor = create_device_tensor(output_specs[0], input_tensor.device());
    auto totals_tensor = create_device_tensor(output_specs[1], input_tensor.device());
    auto expert_region_tensor = create_device_tensor(output_specs[2], input_tensor.device());
    auto all_offsets_tensor = create_device_tensor(output_specs[3], input_tensor.device());
    return {offsets_tensor, totals_tensor, expert_region_tensor, all_offsets_tensor};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 4> offset_cumsum(const Tensor& input_tensor, uint32_t cluster_axis, uint32_t experts_per_chip) {
    using OperationType = ttnn::experimental::prim::OffsetCumsumDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{cluster_axis, experts_per_chip};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, input_tensor);
}

}  // namespace ttnn::prim

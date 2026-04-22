// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subgroup_gather_histograms_device_operation.hpp"

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

void SubgroupGatherHistogramsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.dtype() == DataType::UINT32,
        "Input dtype must be UINT32 (got {})",
        tensor_args.input_tensor.dtype());
    TT_FATAL(tensor_args.input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input layout must be ROW_MAJOR");

    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    // Input is the per-chip histogram, shape [n_routed_experts] or [1, n_routed_experts].
    TT_FATAL(
        input_shape.size() == 1 || input_shape.size() == 2, "Input must be 1D or 2D, got {} dims", input_shape.size());
    if (input_shape.size() == 2) {
        TT_FATAL(input_shape[-2] == 1, "For 2D input first dim must be 1, got {}", input_shape[-2]);
    }
    TT_FATAL(args.cluster_axis == 0, "Only cluster_axis=0 is supported (got {})", args.cluster_axis);
    TT_FATAL(
        args.num_dispatch_subgroups >= 1, "num_dispatch_subgroups must be >= 1 (got {})", args.num_dispatch_subgroups);
    TT_FATAL(!args.output_mem_config.is_sharded(), "Output memory config must be interleaved, not sharded");
}

void SubgroupGatherHistogramsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

SubgroupGatherHistogramsDeviceOperation::spec_return_value_t
SubgroupGatherHistogramsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    const uint32_t n_routed_experts = input_shape[-1];

    auto* mesh_device = tensor_args.input_tensor.device();
    const uint32_t axis_size = mesh_device->shape()[args.cluster_axis];
    TT_FATAL(
        axis_size % args.num_dispatch_subgroups == 0,
        "mesh axis {} size ({}) must be divisible by num_dispatch_subgroups ({})",
        args.cluster_axis,
        axis_size,
        args.num_dispatch_subgroups);
    const uint32_t dispatch_group_size = axis_size / args.num_dispatch_subgroups;

    auto output_shape = ttnn::Shape({dispatch_group_size, n_routed_experts});
    auto layout = tt::tt_metal::TensorLayout(
        DataType::UINT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), args.output_mem_config);
    return TensorSpec(output_shape, layout);
}

SubgroupGatherHistogramsDeviceOperation::topology_return_value_t
SubgroupGatherHistogramsDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;

    const auto& input_topology = tensor_args.input_tensor.tensor_topology();
    const auto& dist_shape = input_topology.distribution_shape();
    size_t ndims = dist_shape.dims();

    // Within each subgroup every chip ends up with an identical [dispatch_group_size, W]
    // tensor, but across subgroups the content can differ (different histograms). The
    // placement metadata should reflect "unique per device" along the cluster axis; we mirror
    // what offset_cumsum does for simplicity.
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    for (size_t i = 0; i < ndims; i++) {
        placements.push_back(Shard{static_cast<int>(i)});
    }
    auto output_topology = tt::tt_metal::TensorTopology(dist_shape, placements, input_topology.mesh_coords());
    return {output_topology};
}

SubgroupGatherHistogramsDeviceOperation::tensor_return_value_t
SubgroupGatherHistogramsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms

namespace ttnn::prim {

ttnn::Tensor subgroup_gather_histograms(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_dispatch_subgroups,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms::
        SubgroupGatherHistogramsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .cluster_axis = cluster_axis,
            .num_dispatch_subgroups = num_dispatch_subgroups,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}

}  // namespace ttnn::prim

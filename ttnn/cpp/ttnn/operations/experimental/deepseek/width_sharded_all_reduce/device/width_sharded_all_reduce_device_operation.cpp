// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_sharded_all_reduce_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::prim {

void WidthShardedAllReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "width_sharded_all_reduce: input must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "width_sharded_all_reduce: input must be allocated");
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "width_sharded_all_reduce: input must be ROW_MAJOR");
    TT_FATAL(
        input_tensor.is_sharded() &&
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            input_tensor.shard_spec().has_value(),
        "width_sharded_all_reduce: input must be WIDTH_SHARDED with a shard spec");
    TT_FATAL(
        input_tensor.memory_config().buffer_type() == BufferType::L1, "width_sharded_all_reduce: input must be L1");
    TT_FATAL(args.num_links > 0, "width_sharded_all_reduce: num_links must be > 0");
    TT_FATAL(args.ring_size > 1, "width_sharded_all_reduce: need more than one device");

    const auto& shard = input_tensor.shard_spec().value();
    TT_FATAL(
        shard.shape[1] % tt::constants::TILE_WIDTH == 0,
        "width_sharded_all_reduce: shard width {} must be a multiple of {}",
        shard.shape[1],
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        (shard.shape[0] * shard.shape[1]) % tt::constants::TILE_WIDTH == 0,
        "width_sharded_all_reduce: shard volume {} must be a multiple of {}",
        shard.shape[0] * shard.shape[1],
        tt::constants::TILE_WIDTH);
}

tt::tt_metal::TensorSpec WidthShardedAllReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.input.tensor_spec();
}

Tensor WidthShardedAllReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

std::vector<tt::tt_metal::TensorTopology> WidthShardedAllReduceDeviceOperation::compute_output_topologies(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_topology = tensor_args.input.tensor_topology();
    auto output_placements = input_topology.placements();
    if (args.cluster_axis.has_value() && args.cluster_axis.value() < output_placements.size()) {
        output_placements[args.cluster_axis.value()] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
    } else {
        for (auto& placement : output_placements) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    }
    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(output_placements), input_topology.mesh_coords())};
}

Tensor width_sharded_all_reduce(
    const Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    const uint32_t ring_size = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    return ttnn::device_operation::launch<WidthShardedAllReduceDeviceOperation>(
        WidthShardedAllReduceParams{
            .num_links = num_links,
            .ring_size = ring_size,
            .cluster_axis = cluster_axis,
            .sub_device_id = subdevice_id,
            .topology = topology,
        },
        WidthShardedAllReduceInputs{.input = input_tensor});
}

}  // namespace ttnn::prim

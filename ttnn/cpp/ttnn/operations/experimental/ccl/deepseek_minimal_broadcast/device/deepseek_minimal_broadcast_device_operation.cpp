// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/deepseek_minimal_broadcast_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

DeepseekMinimalBroadcastDeviceOperation::program_factory_t
DeepseekMinimalBroadcastDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return DeepseekMinimalBroadcastProgramFactory{};
}

void DeepseekMinimalBroadcastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void DeepseekMinimalBroadcastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to broadcast need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to broadcast need to be allocated in buffers on device!");
    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    // input tensor dtype should be bfloat16
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    // input shard grid should be a single core
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;
    std::vector<CoreCoord> cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto c = corerange_to_cores(core_range, std::nullopt);
        cores.insert(cores.end(), c.begin(), c.end());
    }
    TT_FATAL(cores.size() == 1, "Input tensor must be sharded to a single core");

    // input should be tiny tile (1,32)
    const auto tile_width = input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor.tensor_spec().tile().get_height();
    TT_FATAL(input_tensor.layout() == ttnn::TILE_LAYOUT, "Input tensor must be in TILE_LAYOUT");
    TT_FATAL(
        tile_width == 32 && tile_height == 1,
        "Input tensor must be in tile size (1,32). Got tile size: ({}, {})",
        tile_height,
        tile_width);
    // input shape should be (1,1536)
    // To do add shape (1,7168) once fabric supports larger packets
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(
        input_shape[0] == 1 && input_shape[1] == 1536,
        "Input tensor shape must be (1,1536). Got shape: ({}, {})",
        input_shape[0],
        input_shape[1]);
}

TensorSpec DeepseekMinimalBroadcastDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));
}

Tensor DeepseekMinimalBroadcastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t DeepseekMinimalBroadcastDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "DeepseekMinimalBroadcastDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return operation::hash_operation<DeepseekMinimalBroadcastDeviceOperation>(
        operation_attributes.sender_coord,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor deepseek_minimal_broadcast(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    using OperationType = ttnn::experimental::prim::DeepseekMinimalBroadcastDeviceOperation;

    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& tensor_topology_shape = tensor_topology.distribution_shape();

    if (!cluster_axis.has_value()) {
        TT_FATAL(
            tensor_topology_shape.is_line_topology(),
            "minimal deepseek broadcast op is only supported for a linear tensor topology shape");
    }

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    TT_FATAL(
        num_devices > 1,
        "broadcast op will only work for num_devices > 1, but has {}, shape: {}",
        num_devices,
        tensor_topology_shape);

    tt::tt_fabric::Topology ccl_topology = topology;
    TT_FATAL(
        ccl_topology == tt::tt_fabric::Topology::Linear,
        "Currently only Linear topology is supported in deepseek minimal broadcast op, but got {}",
        ccl_topology);

    auto operation_attributes = OperationType::operation_attributes_t{
        .sender_coord = sender_coord,
        .num_links = num_links,
        .ring_size = num_devices,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
        .topology = ccl_topology,
        .cluster_axis = cluster_axis,
        .sub_device_id = sub_device_id};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

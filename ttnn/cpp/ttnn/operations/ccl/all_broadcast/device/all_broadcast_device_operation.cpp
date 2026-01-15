// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl::all_broadcast {

AllBroadcastDeviceOperation::program_factory_t AllBroadcastDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return program::AllBroadcastProgramFactory{};
}

void AllBroadcastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void AllBroadcastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to all_broadcast need to be on device! Storage type: {}",
        static_cast<int>(input_tensor.storage_type()));
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_broadcast need to be allocated in buffers on device!");

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());
}

spec_return_value_t AllBroadcastDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    const uint32_t ring_size = operation_attributes.ring_size;
    std::vector<TensorSpec> output_specs;
    output_specs.reserve(ring_size);
    for (uint32_t i = 0; i < ring_size; ++i) {
        output_specs.push_back(TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                operation_attributes.output_mem_config)));
    }
    return output_specs;
}

tensor_return_value_t AllBroadcastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(operation_attributes, tensor_args);
    std::vector<Tensor> outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(create_device_tensor(spec, tensor_args.input_tensor.device()));
    }
    return outputs;
}

tt::stl::hash::hash_t AllBroadcastDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllBroadcastDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<AllBroadcastDeviceOperation>(
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::operations::ccl::all_broadcast

namespace ttnn::prim {
std::vector<ttnn::Tensor> all_broadcast(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const ttnn::MemoryConfig& output_mem_config,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    using OperationType = ttnn::operations::ccl::all_broadcast::AllBroadcastDeviceOperation;
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& tensor_topology_shape = tensor_topology.distribution_shape();

    if (!cluster_axis.has_value()) {
        TT_FATAL(
            tensor_topology_shape.is_line_topology(),
            "all_broadcast op is only supported for a linear tensor topology shape");
    }

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    TT_FATAL(
        num_devices > 1,
        "all_broadcast op will only work for num_devices > 1, but has {}, shape: {}",
        num_devices,
        tensor_topology_shape);

    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_links = num_links,
            .ring_size = num_devices,
            .output_mem_config = output_mem_config,
            .cluster_axis = cluster_axis,
            .sub_device_id = sub_device_id,
            .topology = topology},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim

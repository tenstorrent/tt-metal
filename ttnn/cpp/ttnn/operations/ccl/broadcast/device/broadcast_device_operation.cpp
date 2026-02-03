// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/broadcast/device/broadcast_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::prim {

BroadcastDeviceOperation::program_factory_t BroadcastDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return BroadcastProgramFactory{};
}

void BroadcastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void BroadcastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to broadcast need to be on device! Storage type: {}",
        static_cast<int>(input_tensor.storage_type()));
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to broadcast need to be allocated in buffers on device!");

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

TensorSpec BroadcastDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));
}

Tensor BroadcastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t BroadcastDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "BroadcastDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<BroadcastDeviceOperation>(
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

Tensor broadcast(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& tensor_topology_shape = tensor_topology.distribution_shape();

    if (!cluster_axis.has_value()) {
        TT_FATAL(
            tensor_topology_shape.is_line_topology(),
            "broadcast op is only supported for a linear tensor topology shape");
    }

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    TT_FATAL(
        num_devices > 1,
        "broadcast op will only work for num_devices > 1, but has {}, shape: {}",
        num_devices,
        tensor_topology_shape);

    tt::tt_fabric::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = tt::tt_fabric::Topology::Linear;
    }
    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    return ttnn::device_operation::launch<BroadcastDeviceOperation>(
        BroadcastParams(
            sender_coord,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            ccl_topology,
            cluster_axis,
            sub_device_id),
        BroadcastInputs{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim

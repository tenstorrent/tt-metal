// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce {
void DeepseekMinimalAllReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");
    TT_FATAL(
        operation_attributes.num_links == 2,
        "Error, num_links must be exactly 2 for deepseek_minimal_all_reduce but has {}",
        operation_attributes.num_links);

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
    // input shape should be (1, 7168)
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(
        input_shape[0] == 1 && input_shape[1] == 7168,
        "Input tensor shape must be (1, 7168). Got shape: ({}, {})",
        input_shape[0],
        input_shape[1]);
}

TensorSpec DeepseekMinimalAllReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    const auto& input_memory_config = input_tensor.memory_config();
    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), input_memory_config));
}

Tensor DeepseekMinimalAllReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.persistent_output_tensor.has_value()) {
        return tensor_args.persistent_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t DeepseekMinimalAllReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return operation::hash_operation<DeepseekMinimalAllReduceDeviceOperation>(
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.device()->id());
}

}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce

namespace ttnn::prim {

ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::DeepseekMinimalAllReduceDeviceOperation::
    tensor_return_value_t
    deepseek_minimal_all_reduce(
        const ttnn::Tensor& input_tensor,
        uint32_t num_links,
        tt::tt_fabric::Topology topology,
        std::optional<uint32_t> cluster_axis,
        const std::optional<ttnn::Tensor>& intermediate_tensor,
        const std::optional<ttnn::Tensor>& residual_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_tensor) {
    using OperationType =
        ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::DeepseekMinimalAllReduceDeviceOperation;

    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& tensor_topology_shape = tensor_topology.distribution_shape();

    if (!cluster_axis.has_value()) {
        TT_FATAL(
            tensor_topology_shape.is_line_topology(),
            "minimal deepseek all_reduce op is only supported for a linear tensor topology shape");
    }

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices == 2,
        "TP deepseek minimal all_reduce op currently only supports 2 devices, but has {}",
        num_devices);
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_links = num_links, .ring_size = num_devices, .topology = topology, .cluster_axis = cluster_axis};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .intermediate_tensor = intermediate_tensor,
        .residual_tensor = residual_tensor,
        .persistent_output_tensor = persistent_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

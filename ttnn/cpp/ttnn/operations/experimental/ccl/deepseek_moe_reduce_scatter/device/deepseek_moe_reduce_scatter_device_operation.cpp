// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include "deepseek_moe_reduce_scatter_device_operation.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

DeepseekMoEReduceScatterDeviceOperation::program_factory_t
DeepseekMoEReduceScatterDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return DeepseekMoEReduceScatterMeshWorkloadFactory{};
}

void DeepseekMoEReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // lightweight validation for cache hits
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    for (const ttnn::Tensor& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
    }
}

void DeepseekMoEReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    // hardcoded constants
    const uint32_t required_ring_size = 8;
    const uint32_t num_directions_per_link = 2;

    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    uint32_t dim = operation_attributes.dim;
    uint32_t num_links = operation_attributes.num_links;
    std::optional<uint32_t> cluster_axis = operation_attributes.cluster_axis;

    // number of input tensors
    TT_FATAL(
        input_tensors.size() == required_ring_size,
        "deepseek_moe_reduce_scatter requires 8 input tensors, but has {}",
        input_tensors.size());

    // input tensor properties
    const ttnn::Tensor& first_input_tensor = input_tensors.at(0);
    const uint32_t input_tensor_rank = first_input_tensor.logical_shape().rank();
    const auto& input_tensor_shape = first_input_tensor.logical_shape();
    TT_FATAL(
        first_input_tensor.mesh_buffer()->page_size() % first_input_tensor.buffer()->alignment() == 0,
        "deepseek_moe_reduce_scatter requires aligned pages");
    TT_FATAL(
        first_input_tensor.buffer()->is_l1(), "deepseek_moe_reduce_scatter requires input tensors allocated in L1");
    TT_FATAL(
        first_input_tensor.layout() == ttnn::Layout::TILE, "deepseek_moe_reduce_scatter requires tiled input tensors");
    TT_FATAL(
        first_input_tensor.element_size() <= 2,
        "deepseek_moe_reduce_scatter requires element size <= 2 bytes for scatter_write usage");
    TT_FATAL(
        input_tensor_rank >= 2,
        "deepseek_moe_reduce_scatter requires input tensor must have rank at least 2, but has {}",
        input_tensor_rank);

    // input tensor must be 1 tile high
    uint32_t outer_dims_product = 1;
    for (uint32_t dim = 0; dim < input_tensor_rank - 1; ++dim) {
        outer_dims_product *= input_tensor_shape[dim];
    }
    TT_FATAL(
        outer_dims_product == tt::constants::TILE_HEIGHT,
        "deepseek_moe_reduce_scatter requires the collapsed upper dims to be the height of a single tile, but has {}",
        outer_dims_product);

    // input tensor shard spec properties
    TT_FATAL(
        first_input_tensor.nd_shard_spec().has_value(),
        "deepseek_moe_reduce_scatter requires nd sharded input tensors");
    const uint32_t num_pages_per_shard =
        first_input_tensor.nd_shard_spec().value().shard_shape.volume() / num_tile_elements;
    const uint32_t num_shards = first_input_tensor.logical_volume() / (num_tile_elements * num_pages_per_shard);
    TT_FATAL(num_pages_per_shard % 2 == 0, "deepseek_moe_reduce_scatter requires shards have an even number pages");
    TT_FATAL(
        num_shards <= num_directions_per_link * num_links,
        "deepseek_moe_reduce_scatter requires number of shards per input tensor is <= num_directions_per_link * "
        "num_links");

    // all input tensors must be identical
    for (uint32_t i = 1; i < input_tensors.size(); ++i) {
        const ttnn::Tensor& input_tensor = input_tensors.at(i);

        TT_FATAL(
            first_input_tensor.dtype() == input_tensor.dtype(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same dtype");
        TT_FATAL(
            first_input_tensor.layout() == input_tensor.layout(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same layout");
        TT_FATAL(
            first_input_tensor.logical_shape() == input_tensor.logical_shape(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same logical shape");
        TT_FATAL(
            first_input_tensor.padded_shape() == input_tensor.padded_shape(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same padded shape");
        TT_FATAL(
            first_input_tensor.tensor_spec() == input_tensor.tensor_spec(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same tensor spec");
        TT_FATAL(
            first_input_tensor.memory_config() == input_tensor.memory_config(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same memory config");
        TT_FATAL(
            first_input_tensor.buffer()->alignment() == input_tensor.buffer()->alignment(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same page alignment");
        TT_FATAL(
            first_input_tensor.mesh_buffer()->page_size() == input_tensor.mesh_buffer()->page_size(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same page size");
        TT_FATAL(
            first_input_tensor.mesh_buffer()->num_pages() == input_tensor.mesh_buffer()->num_pages(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same number of pages");
        TT_FATAL(
            input_tensor.nd_shard_spec().has_value() &&
                first_input_tensor.nd_shard_spec().value() == input_tensor.nd_shard_spec().value(),
            "deepseek_moe_reduce_scatter requires all input tensors have the same nd shard spec");
    }

    // number of devices
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensors.at(0), cluster_axis);
    TT_FATAL(
        num_devices == required_ring_size,
        "deepseek_moe_reduce_scatter is hardcoded for 8 devices, but has {}",
        num_devices);

    // dim
    TT_FATAL(
        dim == input_tensor_rank - 1,
        "deepseek_moe_reduce_scatter only supports scattering on the last dim, but has dim {}",
        dim);
}

std::vector<ttnn::TensorSpec> DeepseekMoEReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;

    const auto& intermediate_shape = input_tensors.at(0).logical_shape();
    const tt::tt_metal::MemoryConfig& intermediate_memory_config = input_tensors.at(0).memory_config();

    const auto& output_shape = input_tensors.at(0).logical_shape();
    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    return {
        TensorSpec(
            intermediate_shape,
            TensorLayout(
                input_tensors.at(0).dtype(),
                input_tensors.at(0).tensor_spec().page_config(),
                intermediate_memory_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                input_tensors.at(0).dtype(), input_tensors.at(0).tensor_spec().page_config(), output_memory_config)),
    };
}

std::vector<ttnn::Tensor> DeepseekMoEReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;

    std::vector<ttnn::TensorSpec> tensor_specs = compute_output_specs(operation_attributes, tensor_args);
    const ttnn::TensorSpec& intermediate_tensor_spec = tensor_specs.at(0);
    const ttnn::TensorSpec& output_tensor_spec = tensor_specs.at(1);

    return {
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(intermediate_tensor_spec, input_tensors.at(0).device()),  // intermediate
        create_device_tensor(output_tensor_spec, input_tensors.at(0).device()),        // output
    };
}

tt::stl::hash::hash_t DeepseekMoEReduceScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "DeepseekMoEReduceScatterDeviceOperation::compute_program_hash is called");

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<DeepseekMoEReduceScatterDeviceOperation>(
        operation_attributes.output_memory_config,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.cluster_axis,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_moe_reduce_scatter(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    uint32_t dim,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::experimental::prim::DeepseekMoEReduceScatterDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_memory_config, dim, num_links, cluster_axis},
        OperationType::tensor_args_t{input_tensors});
}

}  // namespace ttnn::prim

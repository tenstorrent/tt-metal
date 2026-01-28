// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "llama_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

LlamaReduceScatterDeviceOperation::program_factory_t LlamaReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return LlamaReduceScatterAdd{};
}

void LlamaReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();

    TT_FATAL(attributes.dim == 3, "dim must be 1, got {}", attributes.dim);
    TT_FATAL(attributes.cluster_axis == 1, "cluster_axis must be 1, got {}", attributes.cluster_axis);
    TT_FATAL(
        attributes.ring_devices == 4 or attributes.ring_devices == 2,
        "ring_devices must be 4 or 2, got {}",
        attributes.ring_devices);
    TT_FATAL(attributes.cross_device_semaphore.has_value(), "Cross device semaphore is not present");

    TT_FATAL(input_tensor.shard_spec().has_value(), "input_tensor must have a shard spec");
    TT_FATAL(
        input_tensor.shard_spec().value().shape[0] == 32,
        "input_tensor shard height must be 32 but got {}",
        input_tensor.shard_spec().value().shape[0]);

    TT_FATAL(
        tensor_args.intermediate_packet_buffer.shard_spec().has_value(),
        "intermediate_packet_buffer must have a shard spec");
    TT_FATAL(
        tensor_args.intermediate_packet_buffer.shard_spec().value().shape[0] == 32,
        "intermediate_packet_buffer shard height must be 32 but got {}",
        tensor_args.intermediate_packet_buffer.shard_spec().value().shape[0]);
    TT_FATAL(
        tensor_args.intermediate_packet_buffer.tensor_spec().tile().get_tile_shape() == tile_shape,
        "intermediate_packet_buffer must have the same tile shape ({}, {}) as input_tensor",
        tile_shape[0],
        tile_shape[1]);
    if (attributes.output_mem_config.has_value()) {
        TT_FATAL(
            attributes.output_mem_config.value().shard_spec().has_value(), "output_mem_config must have a shard spec");
        TT_FATAL(
            attributes.output_mem_config.value().shard_spec().value().shape[0] == 32,
            "output_mem_config shard height must be 32 but got {}",
            attributes.output_mem_config.value().shard_spec().value().shape[0]);
    }
}

void LlamaReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

LlamaReduceScatterDeviceOperation::spec_return_value_t LlamaReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    // input is unpadded, output is padded. Ex, input: 3584, 112 tiles, padded to 5 tiles per core, total width is 120
    // tiles (3840). this should be changed to use unpadded output in the future.
    auto input_tensor = tensor_args.input_tensor;
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& input_spec = input_tensor.tensor_spec();
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto input_grid = input_shard_spec.grid;
    auto input_shard_width = input_shard_spec.shape[1];
    auto input_num_cores = input_grid.num_cores();
    auto input_shape = input_spec.logical_shape();
    auto input_width = input_shape[attributes.dim];
    auto input_width_in_tiles = input_width / tile_shape[1];
    auto padded_input_width_in_tiles =
        input_num_cores * ((input_width_in_tiles + input_num_cores - 1) / input_num_cores);
    auto padded_input_width = padded_input_width_in_tiles * tile_shape[1];

    uint32_t final_width = input_width % input_shard_width != 0 ? padded_input_width / attributes.ring_devices
                                                                : input_width / attributes.ring_devices;
    TT_FATAL(input_width % attributes.ring_devices == 0, "input shape width must be divisible by num_devices");

    auto output_shape = input_shape;
    output_shape[attributes.dim] = final_width;
    if (attributes.output_mem_config.has_value()) {
        return {TensorSpec(
            Shape(output_shape),
            TensorLayout(
                input_tensor.dtype(), PageConfig(input_tensor.layout()), attributes.output_mem_config.value()))};
    }

    input_shard_spec = input_tensor.shard_spec().value();
    uint32_t num_cores = final_width / input_spec.tile().get_tile_shape()[1];
    auto* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    bool row_wise = input_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto core_range = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, row_wise);

    // this op only supports one tile per output core for now
    ShardSpec shard_spec{core_range, {input_shape[-2], tile_shape[1]}};
    tt::tt_metal::MemoryConfig out_memory_config = input_tensor.memory_config().with_shard_spec(shard_spec);

    return {TensorSpec(
        Shape(output_shape), TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), out_memory_config))};
}

LlamaReduceScatterDeviceOperation::tensor_return_value_t LlamaReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation::tensor_return_value_t llama_reduce_scatter(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    uint32_t cluster_axis,
    uint32_t ring_devices,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    bool use_noc1_only) {
    using OperationType = ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
        .cross_device_semaphore = semaphore,
        .subdevice_id = subdevice_id,
        .cluster_axis = cluster_axis,
        .output_mem_config = memory_config,
        .ring_devices = ring_devices,
        .num_links = num_links,
        .topology = topology,
        .use_noc1_only = use_noc1_only,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .intermediate_packet_buffer = intermediate_packet_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

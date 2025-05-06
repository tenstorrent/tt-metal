// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "cpp/ttnn/tensor/types.hpp"
#include "llama_reduce_scatter_create_heads_device_op.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

LlamaReduceScatterCreateHeadsDeviceOperation::program_factory_t
LlamaReduceScatterCreateHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return LlamaReduceScatterCreateHeads{};
}

void LlamaReduceScatterCreateHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // auto input_tensor = tensor_args.input_tensor;
    // auto tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    // auto input_spec = input_tensor.get_tensor_spec();

    // TT_FATAL(attributes.dim == 3, "dim must be 1, got {}", attributes.dim);
    // TT_FATAL(attributes.cluster_axis == 1, "cluster_axis must be 1, got {}", attributes.cluster_axis);
    // TT_FATAL(
    //     attributes.ring_devices == 4 or attributes.ring_devices == 2,
    //     "ring_devices must be 4 or 2, got {}",
    //     attributes.ring_devices);
    // TT_FATAL(attributes.cross_device_semaphore.has_value(), "Cross device semaphore is not present");

    // TT_FATAL(input_tensor.shard_spec().has_value(), "input_tensor must have a shard spec");
    // TT_FATAL(
    //     input_tensor.shard_spec().value().shape[0] == 32,
    //     "input_tensor shard height must be 32 but got {}",
    //     input_tensor.shard_spec().value().shape[0]);

    // TT_FATAL(
    //     tensor_args.intermediate_packet_buffer.shard_spec().has_value(),
    //     "intermediate_packet_buffer must have a shard spec");
    // TT_FATAL(
    //     tensor_args.intermediate_packet_buffer.shard_spec().value().shape[0] == 32,
    //     "intermediate_packet_buffer shard height must be 32 but got {}",
    //     tensor_args.intermediate_packet_buffer.shard_spec().value().shape[0]);
    // TT_FATAL(
    //     tensor_args.intermediate_packet_buffer.get_tensor_spec().tile().get_tile_shape() == tile_shape,
    //     "intermediate_packet_buffer must have the same tile shape ({}, {}) as input_tensor",
    //     tile_shape[0],
    //     tile_shape[1]);
    // if (attributes.output_mem_config.has_value()) {
    //     TT_FATAL(
    //         attributes.output_mem_config.value().shard_spec.has_value(), "output_mem_config must have a shard spec");
    //     TT_FATAL(
    //         attributes.output_mem_config.value().shard_spec.value().shape[0] == 32,
    //         "output_mem_config shard height must be 32 but got {}",
    //         attributes.output_mem_config.value().shard_spec.value().shape[0]);
    // }
}

void LlamaReduceScatterCreateHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

LlamaReduceScatterCreateHeadsDeviceOperation::spec_return_value_t
LlamaReduceScatterCreateHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    // input is unpadded, output is padded. Ex, input: 3584, 112 tiles, padded to 5 tiles per core, total width is 120
    // tiles (3840). this should be changed to use unpadded output in the future.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.get_logical_shape();
    const auto batch = attributes.slice_size;
    const auto head_dim = attributes.head_dim;
    const Shape q_output_shape({input_shape[0], batch, attributes.num_heads, head_dim});
    const Shape v_output_shape({input_shape[0], batch, attributes.num_kv_heads, head_dim});
    const Shape k_output_shape = v_output_shape;
    CoreRangeSet q_shard_grid, k_shard_grid, v_shard_grid;
    auto sub_core_grid = attributes.qkv_memory_config->shard_spec->grid;
    auto start_core_coord = sub_core_grid.bounding_box().start_coord;
    auto next_core_coord = start_core_coord;

    q_shard_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch + 1, sub_core_grid, true);
    if (!q_batch_grid.ranges().empty()) {
        next_core_coord = q_batch_grid.ranges().back().end_coord;
    }
    k_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_two_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, 2 * batch + 1, sub_core_grid, true);
    if (!q_two_batch_grid.ranges().empty()) {
        next_core_coord = q_two_batch_grid.ranges().back().end_coord;
    }
    v_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    tt::tt_metal::MemoryConfig q_mem_config = attributes.qkv_memory_config.value();
    tt::tt_metal::MemoryConfig k_mem_config = attributes.qkv_memory_config.value();
    tt::tt_metal::MemoryConfig v_mem_config = attributes.qkv_memory_config.value();
    tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {attributes.num_heads, head_dim}};
    q_mem_config.shard_spec = q_shard_spec;
    tt::tt_metal::ShardSpec k_shard_spec{k_shard_grid, {attributes.num_kv_heads, head_dim}};
    k_mem_config.shard_spec = k_shard_spec;
    tt::tt_metal::ShardSpec v_shard_spec{v_shard_grid, {attributes.num_kv_heads, head_dim}};
    v_mem_config.shard_spec = v_shard_spec;

    return {
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), q_mem_config)),
        TensorSpec(
            k_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), k_mem_config)),
        TensorSpec(
            v_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), v_mem_config))};

    /* auto tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    auto input_spec = input_tensor.get_tensor_spec();
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto input_grid = input_shard_spec.grid;
    auto input_shard_height = input_shard_spec.shape[0];
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
                input_tensor.get_dtype(),
                PageConfig(input_tensor.get_layout()),
                attributes.output_mem_config.value()))};
    }

    input_shard_spec = input_tensor.shard_spec().value();
    uint32_t num_cores = final_width / input_spec.tile().get_tile_shape()[1];
    auto device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    bool row_wise = input_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto core_range = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, row_wise);

    // this op only supports one tile per output core for now
    ShardSpec shard_spec{core_range, {input_shape[-2], tile_shape[1]}};
    tt::tt_metal::MemoryConfig out_memory_config = input_tensor.memory_config();
    out_memory_config.shard_spec = shard_spec;

    return {TensorSpec(
        Shape(output_shape),
        TensorLayout(input_tensor.get_dtype(), PageConfig(input_tensor.get_layout()), out_memory_config))};
    */
}

LlamaReduceScatterCreateHeadsDeviceOperation::tensor_return_value_t
LlamaReduceScatterCreateHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    std::vector<ttnn::Tensor> tensors{};
    for (auto& output_spec : output_specs) {
        auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
        tensors.push_back(tensor);
    }
    // auto tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    return tensors;
}

std::tuple<
    LlamaReduceScatterCreateHeadsDeviceOperation::operation_attributes_t,
    LlamaReduceScatterCreateHeadsDeviceOperation::tensor_args_t>
LlamaReduceScatterCreateHeadsDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    const int32_t dim,
    const GlobalSemaphore& semaphore,
    const tt::tt_metal::SubDeviceId subdevice_id,
    const uint32_t cluster_axis,
    const uint32_t ring_devices,
    const ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const uint32_t num_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const uint32_t slice_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& qkv_memory_config) {
    return {
        operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.get_logical_shape().rank() + dim) : (uint32_t)dim),
            .cross_device_semaphore = semaphore,
            .subdevice_id = subdevice_id,
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config,
            .ring_devices = ring_devices,
            .topology = topology,
            .num_links = num_links,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .slice_size = slice_size,
            .qkv_memory_config = qkv_memory_config,
        },
        tensor_args_t{.input_tensor = input_tensor, .intermediate_packet_buffer = intermediate_packet_buffer}};
}

}  // namespace ttnn::operations::experimental::ccl

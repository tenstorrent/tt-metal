// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "cpp/ttnn/tensor/types.hpp"
#include "llama_reduce_scatter_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

LlamaReduceScatterDeviceOperation::program_factory_t LlamaReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return LlamaReduceScatterAdd{};  // When both the tiled dimensions are moved
}

void LlamaReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void LlamaReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

LlamaReduceScatterDeviceOperation::spec_return_value_t LlamaReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    constexpr uint32_t num_devices = 2;  // distributed will give me an API here one day

    // sharding APIs are terrible
    auto input_tensor = tensor_args.input_tensor;
    auto tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    auto input_spec = input_tensor.get_tensor_spec();
    auto input_shape = input_spec.logical_shape();
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t final_width = input_shape[attributes.dim] / num_devices;

    auto output_shape = input_shape;
    output_shape[attributes.dim] = final_width;

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
}

LlamaReduceScatterDeviceOperation::tensor_return_value_t LlamaReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

std::tuple<LlamaReduceScatterDeviceOperation::operation_attributes_t, LlamaReduceScatterDeviceOperation::tensor_args_t>
LlamaReduceScatterDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const GlobalSemaphore semaphore,
    const SubDeviceId subdevice_id,
    const uint32_t ring_index,
    const uint32_t cluster_axis,
    std::optional<IDevice*>& forward_device,
    std::optional<IDevice*>& backward_device,
    const uint32_t ring_devices,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    std::cout << "Primitive operation called" << std::endl;
    return {
        operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.get_logical_shape().rank() + dim) : (uint32_t)dim),
            .cross_device_semaphore = semaphore,
            .subdevice_id = subdevice_id,
            .ring_index = ring_index,
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
            .ring_devices = ring_devices,
            .num_links = num_links,
            .forward_device = forward_device,
            .backward_device = backward_device,
        },
        tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::experimental::ccl

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "redistribute_to_memory_config_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <ttnn/operation.hpp>
#include "redistribute_to_memory_config_device_operation_types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {

namespace CMAKE_UNIQUE_NAMESPACE {
bool has_large_pages(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    if (input_tensor.layout() == Layout::TILE) {
        return false;
    }
    const auto max_l1_size = operations::data_movement::get_max_l1_space(input_tensor);

    uint32_t output_page_size = input_tensor.logical_shape()[-1] * input_tensor.element_size();
    if (output_mem_config.is_sharded() &&
        output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t shard_width = output_mem_config.shard_spec().has_value()
                                   ? output_mem_config.shard_spec().value().shape[1]
                                   : output_mem_config.nd_shard_spec().value().shard_shape[-1];
        output_page_size = shard_width * input_tensor.element_size();
    }
    auto output_buffer_type = output_mem_config.buffer_type();
    auto alignment = (output_buffer_type == tt::tt_metal::BufferType::DRAM) ? tt::tt_metal::hal::get_dram_alignment()
                                                                            : tt::tt_metal::hal::get_l1_alignment();
    auto output_aligned_page_size = tt::align(output_page_size, alignment);

    return (input_tensor.buffer()->page_size() + 2 * output_aligned_page_size > max_l1_size);
}
}  // namespace CMAKE_UNIQUE_NAMESPACE

RedistributeToMemoryConfigDeviceOperation::program_factory_t
RedistributeToMemoryConfigDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    bool has_large_pages =
        CMAKE_UNIQUE_NAMESPACE::has_large_pages(input_tensor, operation_attributes.output_mem_config);

    if (operation_attributes.output_mem_config.is_sharded()) {
        if (input_tensor.layout() == Layout::TILE) {
            return RedistributeToMemoryConfigTilizedShardedProgramFactory{};
        }
        if (!has_large_pages) {
            return RedistributeToMemoryConfigRowMajorShardedProgramFactory{};
        }
    }
    if (input_tensor.layout() == Layout::TILE) {
        return RedistributeToMemoryConfigTilizedDefaultProgramFactory{};
    }
    return RedistributeToMemoryConfigRowMajorDefaultProgramFactory{};
}

void RedistributeToMemoryConfigDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_mem_config = operation_attributes.output_mem_config;
    const auto& output_dtype = operation_attributes.output_dtype;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor needs to be allocated in buffers on device!");

    if (tensor_args.output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.output_tensor.value();
        TT_FATAL(output_tensor.logical_shape() == input_tensor.logical_shape(), "Mismatched output shape");
        TT_FATAL(output_tensor.memory_config() == output_mem_config, "Mismatched output memory config");
        TT_FATAL(output_tensor.dtype() == output_dtype, "Mismatched output dtype");
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor needs to be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor needs to be allocated in buffers on device!");
        TT_FATAL(
            output_tensor.device() == input_tensor.device(),
            "Output tensor needs to be on the same device as the input tensor!");
    }

    if (input_tensor.dtype() != output_dtype) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Input tensor layout must be TILE when dtype conversion is needed but got {}",
            input_tensor.layout());
    }
    if (input_tensor.layout() == Layout::TILE) {
        const auto output_tile =
            tensor_args.output_tensor.has_value()
                ? tensor_args.output_tensor.value().tensor_spec().tile()
                : tt::tt_metal::TensorLayout(
                      output_dtype, tt::tt_metal::PageConfig(input_tensor.layout()), output_mem_config)
                      .get_tile();
        TT_FATAL(
            input_tensor.tensor_spec().tile().get_tile_shape() == output_tile.get_tile_shape(),
            "Input and output tensors must have the same tile shape when layout is TILE");
    }
}

RedistributeToMemoryConfigDeviceOperation::spec_return_value_t
RedistributeToMemoryConfigDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config));
}

RedistributeToMemoryConfigDeviceOperation::tensor_return_value_t
RedistributeToMemoryConfigDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

ttsl::hash::hash_t RedistributeToMemoryConfigDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::operation::hash_operation<RedistributeToMemoryConfigDeviceOperation>(
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
}

Tensor redistribute_to_memory_config(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<RedistributeToMemoryConfigDeviceOperation>(
        RedistributeToMemoryConfigOperationAttributes{output_mem_config, output_dtype},
        RedistributeToMemoryConfigTensorArgs{input_tensor, preallocated_output});
}
}  // namespace ttnn::prim

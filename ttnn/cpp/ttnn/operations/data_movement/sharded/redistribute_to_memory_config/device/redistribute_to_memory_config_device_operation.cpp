// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "redistribute_to_memory_config_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <ttnn/operation.hpp>
#include "redistribute_to_memory_config_device_operation_types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

RedistributeToMemoryConfigDeviceOperation::program_factory_t
RedistributeToMemoryConfigDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    if (operation_attributes.output_mem_config.is_sharded()) {
        if (input_tensor.layout() == Layout::TILE) {
            return RedistributeToMemoryConfigTilizedShardedProgramFactory{};
        }
        return RedistributeToMemoryConfigRowMajorShardedProgramFactory{};
    }
    return RedistributeToMemoryConfigRowMajorShardedProgramFactory{};  // TODO: will implement to interleaved program
                                                                       // factories in follow up PR!
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

    TT_FATAL(
        output_mem_config.is_sharded(),
        "Output memory config must be sharded");  // TODO: Add path to support interleaved output in subsequent PR

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
        TensorLayout(
            operation_attributes.output_dtype,
            PageConfig(input_tensor.layout()),
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

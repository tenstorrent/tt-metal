// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_sharded_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <ttnn/operation.hpp>
#include "to_sharded_device_operation_types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

ToShardedDeviceOperation::program_factory_t ToShardedDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    if (input_tensor.layout() == Layout::TILE) {
        return ToShardedTilizedProgramFactory{};
    }
    return ToShardedRowMajorProgramFactory{};
}

void ToShardedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_mem_config = operation_attributes.output_mem_config;
    const auto& output_dtype = operation_attributes.output_dtype;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    if (tensor_args.output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.output_tensor.value();
        TT_FATAL(output_tensor.logical_shape() == input_tensor.logical_shape(), "Mismatched output shape");
        TT_FATAL(output_tensor.memory_config() == output_mem_config, "Mismatched output memory config");
        TT_FATAL(output_tensor.dtype() == output_dtype, "Mismatched output dtype");
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
        TT_FATAL(output_tensor.device() == input_tensor.device(), "Operands to shard need to be on the same device!");
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
                : TensorLayout(output_dtype, PageConfig(input_tensor.layout()), output_mem_config).get_tile();
        TT_FATAL(
            input_tensor.tensor_spec().tile().get_tile_shape() == output_tile.get_tile_shape(),
            "Input and output tensors must have the same tile shape when layout is TILE");
    }
}

ToShardedDeviceOperation::spec_return_value_t ToShardedDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto output_layout = TensorLayout(
        operation_attributes.output_dtype, PageConfig(input_tensor.layout()), operation_attributes.output_mem_config);
    auto output_padded_shape = output_layout.compute_padded_shape(
        input_tensor.logical_shape());  // We need to account for the fact that the output tensor may have a
    // different padded_shape due to having a different shard_spec.

    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            output_padded_shape));
}

ToShardedDeviceOperation::tensor_return_value_t ToShardedDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

ttsl::hash::hash_t ToShardedDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::operation::hash_operation<ToShardedDeviceOperation>(
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
}

Tensor to_sharded(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<ToShardedDeviceOperation>(
        operation_attributes_t{output_mem_config, output_dtype},  // keep_l1_aligned},
        tensor_args_t{input_tensor, preallocated_output});
}
}  // namespace ttnn::prim

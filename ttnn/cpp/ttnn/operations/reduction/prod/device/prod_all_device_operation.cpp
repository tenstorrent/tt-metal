// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_all_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {
void ProdAllDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(
        input.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Operands need to be on device! Got storage type: {}",
        input.storage_type());
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::TILE, "Input Layout must be tilized, got layout: {}", input.layout());
    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "Memory layout must be INTERLEAVED, got: {}",
        input.memory_config().memory_layout());
    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Error - unsupported data type for prod, expected BFLOAT16 but got {}.",
        input.dtype());

    {
        const auto prod_all_logical_rank = tensor_args.input.logical_shape().rank();
        TT_FATAL(
            prod_all_logical_rank >= 1,
            "Prod_all expects logical rank >= 1 at device entry, got {}",
            prod_all_logical_rank);
    }

    {
        const auto& prod_all_padded_shape = tensor_args.input.padded_shape();
        const uint32_t prod_all_tile_height = tensor_args.input.tensor_spec().tile().get_height();
        const uint32_t prod_all_tile_width = tensor_args.input.tensor_spec().tile().get_width();
        TT_FATAL(
            prod_all_padded_shape.rank() >= 2,
            "Prod_all padded_shape rank {} must be at least 2 for H/W tile checks",
            prod_all_padded_shape.rank());
        TT_FATAL(
            prod_all_padded_shape[-2] > 0 && prod_all_padded_shape[-1] > 0,
            "Prod_all padded spatial dims must be positive: height={}, width={}",
            prod_all_padded_shape[-2],
            prod_all_padded_shape[-1]);
        TT_FATAL(
            prod_all_padded_shape[-2] % prod_all_tile_height == 0,
            "Prod_all padded_height={} must be tile-height-aligned ({})",
            prod_all_padded_shape[-2],
            prod_all_tile_height);
        TT_FATAL(
            prod_all_padded_shape[-1] % prod_all_tile_width == 0,
            "Prod_all padded_width={} must be tile-width-aligned ({})",
            prod_all_padded_shape[-1],
            prod_all_tile_width);
    }
}

ProdAllDeviceOperation::spec_return_value_t ProdAllDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        ttnn::Shape({1, 1, 1, input.tensor_spec().tile().get_tile_hw()}),
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

ProdAllDeviceOperation::tensor_return_value_t ProdAllDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttnn::Tensor prod_all(const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ProdAllDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim

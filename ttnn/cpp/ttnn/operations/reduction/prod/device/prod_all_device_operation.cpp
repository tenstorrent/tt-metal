// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "prod_all_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction::prod_all {

ProdAllDeviceOperation::program_factory_t ProdAllDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::ProdAllProgramFactory{};
}

void ProdAllDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

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
}

ProdAllDeviceOperation::spec_return_value_t ProdAllDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        ttnn::Shape({1, 1, 1, tt::constants::TILE_HW}),
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

ProdAllDeviceOperation::tensor_return_value_t ProdAllDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::reduction::prod_all

namespace ttnn::prim {
ttnn::Tensor prod_all(const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::reduction::prod_all::ProdAllDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim

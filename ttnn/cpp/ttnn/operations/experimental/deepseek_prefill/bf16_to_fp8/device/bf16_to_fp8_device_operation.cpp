// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bf16_to_fp8_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8 {

namespace {
bool is_dram_interleaved(const ttnn::Tensor& tensor) {
    const auto& mem_cfg = tensor.memory_config();
    return mem_cfg.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_cfg.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}
}  // namespace

void Bf16ToFp8DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor must have a buffer");
    TT_FATAL(input.dtype() == tt::tt_metal::DataType::BFLOAT16, "input_tensor must be BFLOAT16, got {}", input.dtype());
    TT_FATAL(input.layout() == tt::tt_metal::Layout::TILE, "input_tensor must be TILE layout, got {}", input.layout());
    TT_FATAL(is_dram_interleaved(input), "input_tensor must be DRAM interleaved");

    const auto& shape = input.logical_shape();
    const uint32_t tile_h = tt::constants::TILE_HEIGHT;
    const uint32_t tile_w = tt::constants::TILE_WIDTH;
    TT_FATAL(shape[-2] % tile_h == 0, "input rows ({}) must be a multiple of TILE_HEIGHT ({})", shape[-2], tile_h);
    TT_FATAL(shape[-1] % tile_w == 0, "input cols ({}) must be a multiple of TILE_WIDTH ({})", shape[-1], tile_w);
}

void Bf16ToFp8DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

Bf16ToFp8DeviceOperation::spec_return_value_t Bf16ToFp8DeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    // FP8 dispatch trick: allocate as UINT8 (1 byte/element) — kernel writes Fp8_e4m3 bytes.
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT8, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));
}

Bf16ToFp8DeviceOperation::tensor_return_value_t Bf16ToFp8DeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8

namespace ttnn::prim {

ttnn::Tensor prefill_bf16_to_fp8(const ttnn::Tensor& input_tensor) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8::Bf16ToFp8DeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{.input_tensor = input_tensor});
}

}  // namespace ttnn::prim

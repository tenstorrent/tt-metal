// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/common/fp8_quant_common.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

namespace common = ttnn::operations::experimental::deepseek_prefill::fp8_quant_common;

PerTokenCastBackDeviceOperation::program_factory_t PerTokenCastBackDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PerTokenCastBackProgramFactory{};
}

void PerTokenCastBackDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& e4m3 = tensor_args.input_e4m3;
    const auto& scale = tensor_args.input_scale;

    TT_FATAL(
        e4m3.dtype() == tt::tt_metal::DataType::FP8_E4M3, "per_token_cast_back: input_e4m3 dtype must be FP8_E4M3");
    TT_FATAL(
        scale.dtype() == tt::tt_metal::DataType::FLOAT32, "per_token_cast_back: input_scale dtype must be FLOAT32");
    TT_FATAL(e4m3.layout() == tt::tt_metal::Layout::ROW_MAJOR, "per_token_cast_back: input_e4m3 must be ROW_MAJOR");
    TT_FATAL(scale.layout() == tt::tt_metal::Layout::ROW_MAJOR, "per_token_cast_back: input_scale must be ROW_MAJOR");
    TT_FATAL(
        attrs.output_dtype == tt::tt_metal::DataType::BFLOAT16 || attrs.output_dtype == tt::tt_metal::DataType::FLOAT32,
        "per_token_cast_back: output_dtype must be BFLOAT16 or FLOAT32");

    const auto& e4m3_shape = e4m3.logical_shape();
    const auto& scale_shape = scale.logical_shape();
    TT_FATAL(e4m3_shape.size() >= 2, "per_token_cast_back: input_e4m3 rank must be >= 2");
    TT_FATAL(
        e4m3_shape.size() == scale_shape.size(),
        "per_token_cast_back: input_e4m3 ({}D) and input_scale ({}D) must have the same rank",
        e4m3_shape.size(),
        scale_shape.size());

    for (size_t i = 0; i + 1 < e4m3_shape.size(); ++i) {
        TT_FATAL(
            e4m3_shape[i] == scale_shape[i],
            "per_token_cast_back: leading dim {} mismatch ({} vs {})",
            i,
            e4m3_shape[i],
            scale_shape[i]);
    }

    const uint32_t H = static_cast<uint32_t>(e4m3_shape[-1]);
    const uint32_t H_scale = static_cast<uint32_t>(scale_shape[-1]);
    TT_FATAL(
        H == H_scale * common::SCALE_GROUP_SIZE,
        "per_token_cast_back: e4m3 last dim ({}) must equal scale last dim ({}) * SCALE_GROUP_SIZE ({})",
        H,
        H_scale,
        common::SCALE_GROUP_SIZE);
    TT_FATAL(
        H % common::COL_BLOCK_ELEMS == 0,
        "per_token_cast_back: e4m3 last dim ({}) must be a multiple of {} (LLK column-block width)",
        H,
        common::COL_BLOCK_ELEMS);

    auto [M, _H] = common::infer_M_H(e4m3_shape);
    TT_FATAL(
        M % tt::constants::TILE_HEIGHT == 0,
        "per_token_cast_back: row count M={} must be a multiple of TILE_HEIGHT={}",
        M,
        tt::constants::TILE_HEIGHT);
}

void PerTokenCastBackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

PerTokenCastBackDeviceOperation::spec_return_value_t PerTokenCastBackDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input_e4m3.logical_shape(),
        tt::tt_metal::TensorLayout(
            attrs.output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), attrs.output_memory_config));
}

PerTokenCastBackDeviceOperation::tensor_return_value_t PerTokenCastBackDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_e4m3.device());
}

tt::stl::hash::hash_t PerTokenCastBackDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<PerTokenCastBackDeviceOperation>(
        attrs,
        tensor_args.input_e4m3.dtype(),
        tensor_args.input_e4m3.memory_config(),
        tensor_args.input_e4m3.logical_shape(),
        tensor_args.input_scale.logical_shape());
}

}  // namespace ttnn::experimental::prim::per_token_cast_back

namespace ttnn::prim {

ttnn::Tensor per_token_cast_back(
    const Tensor& input_e4m3,
    const Tensor& input_scale,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType = ttnn::experimental::prim::per_token_cast_back::PerTokenCastBackDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = output_dtype, .output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{.input_e4m3 = input_e4m3, .input_scale = input_scale};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/common/fp8_quant_common.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

namespace common = ttnn::operations::experimental::deepseek_prefill::fp8_quant_common;

namespace {

bool is_dram_interleaved(const tt::tt_metal::MemoryConfig& mem_config) {
    return mem_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_device_tensor(const Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR layout", name);
}

}  // namespace

PerTokenCastToFp8DeviceOperation::program_factory_t PerTokenCastToFp8DeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PerTokenCastToFp8ProgramFactory{};
}

void PerTokenCastToFp8DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    validate_device_tensor(input, "per_token_cast_to_fp8: input_tensor");
    TT_FATAL(
        is_dram_interleaved(attrs.output_memory_config),
        "per_token_cast_to_fp8: output memory config must be DRAM interleaved");
    TT_FATAL(
        input.device()->arch() == tt::ARCH::BLACKHOLE,
        "per_token_cast_to_fp8: FP8_E4M3 path requires Blackhole hardware, got arch {}",
        input.device()->arch());
    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16 || input.dtype() == tt::tt_metal::DataType::FLOAT32,
        "per_token_cast_to_fp8: input dtype must be BFLOAT16 or FLOAT32, got {}",
        static_cast<int>(input.dtype()));

    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    TT_FATAL(
        tile_h * tile_w == common::ROW_MAJOR_TILE_ELEMS,
        "per_token_cast_to_fp8: tile_h * tile_w must equal ROW_MAJOR_TILE_ELEMS={} for row-major block tilization, got "
        "{}x{}",
        common::ROW_MAJOR_TILE_ELEMS,
        tile_h,
        tile_w);
    TT_FATAL(
        common::SCALE_GROUP_SIZE % tile_w == 0,
        "per_token_cast_to_fp8: tile width {} must divide SCALE_GROUP_SIZE={}",
        tile_w,
        common::SCALE_GROUP_SIZE);

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.size() >= 2, "per_token_cast_to_fp8: input rank must be >= 2, got {}", shape.size());

    auto [M, H] = common::infer_M_H(shape);
    // M and H are arbitrary (the kernels zero-pad the partial last tile-row / column-block). H must
    // stay a multiple of the 128-element scale group so groups are always full.
    TT_FATAL(
        H % common::SCALE_GROUP_SIZE == 0,
        "per_token_cast_to_fp8: hidden dim H={} must be a multiple of SCALE_GROUP_SIZE={}",
        H,
        common::SCALE_GROUP_SIZE);
    TT_FATAL(M > 0, "per_token_cast_to_fp8: M must be > 0");
}

void PerTokenCastToFp8DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

PerTokenCastToFp8DeviceOperation::spec_return_value_t PerTokenCastToFp8DeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& input_shape = input.logical_shape();

    TensorSpec output_e4m3_spec(
        input_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FP8_E4M3,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_memory_config));

    TensorSpec scale_spec(
        common::scale_shape_from_input(input_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_memory_config));

    return {output_e4m3_spec, scale_spec};
}

PerTokenCastToFp8DeviceOperation::tensor_return_value_t PerTokenCastToFp8DeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [output_e4m3_spec, scale_spec] = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return {create_device_tensor(output_e4m3_spec, device), create_device_tensor(scale_spec, device)};
}

tt::stl::hash::hash_t PerTokenCastToFp8DeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    return tt::tt_metal::operation::hash_operation<PerTokenCastToFp8DeviceOperation>(
        attrs,
        input.dtype(),
        input.memory_config(),
        input.logical_shape(),
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]);
}

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8

namespace ttnn::prim {

std::tuple<ttnn::Tensor, ttnn::Tensor> per_token_cast_to_fp8(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType = ttnn::experimental::prim::per_token_cast_to_fp8::PerTokenCastToFp8DeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{.output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

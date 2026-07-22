// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_device_operation.hpp"

#include <cstdint>
#include <limits>

#include <tt-metalium/constants.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

bool is_dram_interleaved(const tt::tt_metal::MemoryConfig& mem_config) {
    return mem_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_device_tensor(const Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
}

ttnn::Shape scale_output_shape(const ttnn::Shape& input_shape) {
    const auto rank = input_shape.size();
    const uint32_t H = static_cast<uint32_t>(input_shape[rank - 1]);
    ttsl::SmallVector<uint32_t> dims;
    dims.reserve(rank);
    for (size_t i = 0; i + 1 < rank; ++i) {
        dims.push_back(static_cast<uint32_t>(input_shape[i]));
    }
    dims.push_back(H / fp8::BLOCK_W);
    return ttnn::Shape(std::move(dims));
}

}  // namespace

PerTokenCastToFp8DeviceOperation::program_factory_t PerTokenCastToFp8DeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    // Single factory for both layouts; create() branches on input.layout(). ROW_MAJOR/TILE stay separate
    // program-cache entries because compute_program_hash hashes input.layout().
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
    // Row-major circular-buffer pages still use one logical tile. The quantization kernels then stream
    // those pages as tile-height batches of 128-element scale blocks.
    constexpr uint32_t ROW_MAJOR_TILE_ELEMS = 1024;
    TT_FATAL(
        tile_h * tile_w == ROW_MAJOR_TILE_ELEMS,
        "per_token_cast_to_fp8: tile_h * tile_w must equal ROW_MAJOR_TILE_ELEMS={} for row-major block tilization, got "
        "{}x{}",
        ROW_MAJOR_TILE_ELEMS,
        tile_h,
        tile_w);
    TT_FATAL(
        fp8::BLOCK_W % tile_w == 0,
        "per_token_cast_to_fp8: tile width {} must divide BLOCK_W={}",
        tile_w,
        fp8::BLOCK_W);

    const auto& shape = input.logical_shape();
    const auto rank = shape.size();
    TT_FATAL(rank >= 2, "per_token_cast_to_fp8: input rank must be >= 2, got {}", rank);

    uint64_t M = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        M *= static_cast<uint64_t>(shape[i]);
        TT_FATAL(
            M <= std::numeric_limits<uint32_t>::max(),
            "per_token_cast_to_fp8: folded row count M={} exceeds uint32_t range",
            M);
    }
    TT_FATAL(
        static_cast<uint64_t>(shape[rank - 1]) <= std::numeric_limits<uint32_t>::max(),
        "per_token_cast_to_fp8: hidden dim H={} exceeds uint32_t range",
        shape[rank - 1]);

    const uint32_t folded_M = static_cast<uint32_t>(M);
    const uint32_t H = static_cast<uint32_t>(shape[rank - 1]);
    // M and H are arbitrary (the kernels zero-pad the partial last tile-row / column-block). H must
    // stay a multiple of the 128-element block width so scale blocks are always full.
    TT_FATAL(
        H % fp8::BLOCK_W == 0,
        "per_token_cast_to_fp8: hidden dim H={} must be a multiple of BLOCK_W={}",
        H,
        fp8::BLOCK_W);
    TT_FATAL(folded_M > 0, "per_token_cast_to_fp8: M must be > 0");
}

void PerTokenCastToFp8DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

PerTokenCastToFp8DeviceOperation::spec_return_value_t PerTokenCastToFp8DeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& input_shape = input.logical_shape();

    tt::tt_metal::TensorSpec output_e4m3_spec(
        input_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FP8_E4M3,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attrs.output_memory_config));

    tt::tt_metal::TensorSpec scale_spec(
        scale_output_shape(input_shape),
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

ttsl::hash::hash_t PerTokenCastToFp8DeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    return tt::tt_metal::operation::hash_operation<PerTokenCastToFp8DeviceOperation>(
        attrs,
        input.dtype(),
        input.layout(),  // ROW_MAJOR and TILE select different program factories
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
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    bool round_scale_to_power_of_two) {
    using OperationType = ttnn::experimental::prim::per_token_cast_to_fp8::PerTokenCastToFp8DeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_memory_config = output_memory_config, .round_scale_to_power_of_two = round_scale_to_power_of_two};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

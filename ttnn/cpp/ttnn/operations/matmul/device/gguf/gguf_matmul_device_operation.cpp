// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

GgufMatmulDeviceOperation::program_factory_t GgufMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GgufMatmulProgramFactory{};
}

void GgufMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    const Tensor& input = args.input;
    const Tensor& weight = args.weight;
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "GGUF matmul: input and weight must be on device");
    TT_FATAL(input.is_allocated() && weight.is_allocated(), "GGUF matmul: input and weight must be allocated");
    TT_FATAL(input.device() == weight.device(), "GGUF matmul: input and weight must be on the same device");
    TT_FATAL(input.device()->arch() == tt::ARCH::BLACKHOLE, "GGUF matmul: GGUF weights need Blackhole");
    TT_FATAL(is_gguf(weight.dtype()), "GGUF matmul: the weight must have a GGUF dtype, got {}", weight.dtype());
    TT_FATAL(
        weight.memory_config().buffer_type() == BufferType::DRAM &&
            weight.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "GGUF matmul: the weight must be DRAM interleaved, got {}",
        weight.memory_config());
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 && input.layout() == Layout::TILE,
        "GGUF matmul: the input must be BFLOAT16 TILE, got {} {}",
        input.dtype(),
        input.layout());
    TT_FATAL(
        input.tensor_spec().tile() == tt::tt_metal::Tile(),
        "GGUF matmul: the input must use 32x32 tiles, got {}",
        input.tensor_spec().tile());
    const auto& w_shape = weight.logical_shape();
    TT_FATAL(w_shape.rank() == 2, "GGUF matmul: the weight must be [K, N], got {}", w_shape);
    const uint32_t k = w_shape[0];
    TT_FATAL(
        input.logical_shape()[-1] == k && input.padded_shape()[-1] == k,
        "GGUF matmul: the input's last dimension {} must equal the weight's K {}",
        input.logical_shape()[-1],
        k);
    TT_FATAL(
        input.padded_shape().volume() / k == tt::constants::TILE_HEIGHT,
        "GGUF matmul: the input must hold one row block of {} rows, got {}",
        tt::constants::TILE_HEIGHT,
        input.padded_shape());
    TT_FATAL(
        attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "GGUF matmul: the output must be interleaved, got {}",
        attributes.output_mem_config);
}

void GgufMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    validate_on_program_cache_miss(attributes, args);
}

GgufMatmulDeviceOperation::spec_return_value_t GgufMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    ttnn::Shape shape = args.input.logical_shape();
    shape[-1] = args.weight.logical_shape()[1];
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attributes.output_mem_config));
}

GgufMatmulDeviceOperation::tensor_return_value_t GgufMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attributes, args), args.input.device());
}

Tensor gguf_matmul(const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config) {
    return ttnn::device_operation::launch<GgufMatmulDeviceOperation>(
        GgufMatmulParams{output_mem_config}, GgufMatmulInputs{input, weight});
}

}  // namespace ttnn::prim

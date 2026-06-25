// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_heads_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

void ConcatHeadsMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& attn = tensor_args.attn;
    const auto& weight = tensor_args.weight;
    TT_FATAL(
        attn.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "attn and weight must be on device");
    TT_FATAL(attn.layout() == Layout::TILE && weight.layout() == Layout::TILE, "inputs must be tilized");
    TT_FATAL(
        attn.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            weight.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "inputs must be INTERLEAVED");
    TT_FATAL(attn.padded_shape().rank() == 4, "attn must be rank-4 [1, nh, seq, hd]");
    TT_FATAL(
        attn.padded_shape()[2] == TILE_HEIGHT,
        "concat_heads_matmul requires seq <= one tile (Mt==1); got seq {}",
        attn.padded_shape()[2]);
    uint32_t K = attn.padded_shape()[1] * attn.padded_shape()[3];  // nh * hd
    TT_FATAL(
        weight.padded_shape()[-2] == K,
        "weight K ({}) must equal num_heads*head_dim ({})",
        weight.padded_shape()[-2],
        K);
    TT_FATAL(args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "output must be INTERLEAVED");
}

ConcatHeadsMatmulDeviceOperation::spec_return_value_t ConcatHeadsMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    uint32_t seq = tt::round_up(args.seq_len, TILE_HEIGHT);
    uint32_t N = tensor_args.weight.padded_shape()[-1];
    return TensorSpec(
        ttnn::Shape({1, 1, seq, N}),
        tt::tt_metal::TensorLayout(args.output_dtype, tt::tt_metal::PageConfig(Layout::TILE), args.output_mem_config));
}

ConcatHeadsMatmulDeviceOperation::tensor_return_value_t ConcatHeadsMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.attn.device());
}

ttsl::hash::hash_t ConcatHeadsMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<ConcatHeadsMatmulDeviceOperation>(
        args.seq_len, args.output_mem_config, args.output_dtype, tensor_args.attn, tensor_args.weight);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor concat_heads_matmul(
    const Tensor& attn,
    const Tensor& weight,
    uint32_t seq_len,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    tt::tt_metal::DataType output_dtype,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config) {
    using OperationType = ttnn::experimental::prim::ConcatHeadsMatmulDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .seq_len = seq_len,
        .output_mem_config = output_mem_config,
        .output_dtype = output_dtype,
        .compute_kernel_config = compute_kernel_config,
        .program_config = std::move(program_config),
    };
    auto inputs = OperationType::tensor_args_t{.attn = attn, .weight = weight};
    return ttnn::device_operation::launch<OperationType>(attrs, inputs);
}

}  // namespace ttnn::prim

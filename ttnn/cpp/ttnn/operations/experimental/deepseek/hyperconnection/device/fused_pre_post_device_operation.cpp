// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_pre_post_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

void validate_tensors(const FusedPrePostInputs& tensor_args) {
    const auto& pre_w = tensor_args.pre_w;
    const auto& post_w = tensor_args.post_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;

    TT_FATAL(pre_w.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: pre_w must be on device");
    TT_FATAL(post_w.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: post_w must be on device");
    TT_FATAL(
        pre_bias.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: pre_bias must be on device");
    TT_FATAL(
        post_bias.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: post_bias must be on device");
    TT_FATAL(
        hidden_streams.storage_type() == StorageType::DEVICE,
        "fused_hyperconnection_pre_post: hidden_streams must be on device");

    TT_FATAL(pre_w.layout() == Layout::TILE, "fused_hyperconnection_pre_post: pre_w must be TILE layout");
    TT_FATAL(post_w.layout() == Layout::TILE, "fused_hyperconnection_pre_post: post_w must be TILE layout");
    TT_FATAL(pre_bias.layout() == Layout::TILE, "fused_hyperconnection_pre_post: pre_bias must be TILE layout");
    TT_FATAL(post_bias.layout() == Layout::TILE, "fused_hyperconnection_pre_post: post_bias must be TILE layout");
    TT_FATAL(
        hidden_streams.layout() == Layout::TILE, "fused_hyperconnection_pre_post: hidden_streams must be TILE layout");

    TT_FATAL(pre_w.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: pre_w must be BFLOAT16");
    TT_FATAL(post_w.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: post_w must be BFLOAT16");
    TT_FATAL(pre_bias.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: pre_bias must be BFLOAT16");
    TT_FATAL(post_bias.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: post_bias must be BFLOAT16");
    TT_FATAL(
        hidden_streams.dtype() == DataType::BFLOAT16,
        "fused_hyperconnection_pre_post: hidden_streams must be BFLOAT16");

    TT_FATAL(
        pre_w.logical_shape() == post_w.logical_shape(),
        "fused_hyperconnection_pre_post: pre_w and post_w must have the same shape");
    TT_FATAL(
        pre_bias.logical_shape() == post_bias.logical_shape(),
        "fused_hyperconnection_pre_post: pre_bias and post_bias must have the same shape");

    const auto& pre_shape = pre_w.logical_shape();
    const auto& bias_shape = pre_bias.logical_shape();
    const auto& hidden_shape = hidden_streams.logical_shape();
    TT_FATAL(
        pre_shape.rank() == 4,
        "fused_hyperconnection_pre_post: pre_w must be rank-4 [1,1,T,H], got rank {}",
        pre_shape.rank());
    TT_FATAL(
        bias_shape.rank() == 4,
        "fused_hyperconnection_pre_post: pre_bias must be rank-4 [1,1,1,H], got rank {}",
        bias_shape.rank());
    TT_FATAL(
        hidden_shape.rank() == 4,
        "fused_hyperconnection_pre_post: hidden_streams must be rank-4 [1,1,H,D], got rank {}",
        hidden_shape.rank());
    TT_FATAL(pre_shape[0] == 1 && pre_shape[1] == 1, "fused_hyperconnection_pre_post: pre_w must be [1,1,T,H]");
    // Decode-only fused op: a single token (T == 1) so the stream collapse is a [1,H] x [H,D] matmul.
    TT_FATAL(
        pre_shape[2] == 1, "fused_hyperconnection_pre_post: only T==1 (decode) is supported, got T={}", pre_shape[2]);
    TT_FATAL(
        bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1,
        "fused_hyperconnection_pre_post: pre_bias must be [1,1,1,H]");
    TT_FATAL(
        pre_shape[-1] == bias_shape[-1],
        "fused_hyperconnection_pre_post: pre_w and pre_bias last dim must match ({} vs {})",
        pre_shape[-1],
        bias_shape[-1]);
    TT_FATAL(
        hidden_shape[0] == 1 && hidden_shape[1] == 1,
        "fused_hyperconnection_pre_post: hidden_streams must be [1,1,H,D] (decode, T==1)");
    TT_FATAL(
        hidden_shape[2] == pre_shape[-1],
        "fused_hyperconnection_pre_post: hidden_streams stream dim ({}) must match H ({})",
        hidden_shape[2],
        pre_shape[-1]);
}

}  // namespace

void FusedPrePostDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(tensor_args);
    (void)attributes;
}

void FusedPrePostDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(tensor_args);
    (void)attributes;
}

FusedPrePostDeviceOperation::spec_return_value_t FusedPrePostDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& post_w = tensor_args.post_w;
    const auto& hidden_streams = tensor_args.hidden_streams;
    const auto output_layout = tt::tt_metal::TensorLayout(
        post_w.dtype(), tt::tt_metal::PageConfig(post_w.layout()), operation_attributes.output_mem_config);

    // post = 2 * sigmoid(post_w * post_scale + post_bias), shape [1,1,1,H] (same as post_w).
    // collapsed = pre[1,H] @ hidden[H,D] -> [1,1,1,D].
    const auto& hidden_shape = hidden_streams.logical_shape();
    const ttnn::Shape collapsed_shape({1, 1, 1, hidden_shape[-1]});
    return {TensorSpec(post_w.logical_shape(), output_layout), TensorSpec(collapsed_shape, output_layout)};
}

FusedPrePostDeviceOperation::tensor_return_value_t FusedPrePostDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(specs[0], tensor_args.post_w.device()),
        create_device_tensor(specs[1], tensor_args.hidden_streams.device())};
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

std::array<Tensor, 2> fused_hyperconnection_pre_post(
    const Tensor& pre_w,
    const Tensor& post_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& hidden_streams,
    float pre_scale,
    float post_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::hyperconnection::FusedPrePostDeviceOperation;
    const MemoryConfig output_mem_config = memory_config.value_or(pre_w.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{
        .pre_scale = pre_scale,
        .post_scale = post_scale,
        .eps = eps,
        .output_mem_config = output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .pre_w = pre_w,
        .post_w = post_w,
        .pre_bias = pre_bias,
        .post_bias = post_bias,
        .hidden_streams = hidden_streams,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

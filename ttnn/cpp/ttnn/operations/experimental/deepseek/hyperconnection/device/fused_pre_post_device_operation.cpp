// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_pre_post_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

void validate_tensors(const FusedPrePostParams& attributes, const FusedPrePostInputs& tensor_args) {
    const auto& fused_w = tensor_args.fused_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;

    TT_FATAL(
        fused_w.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: fused_w must be on device");
    TT_FATAL(
        pre_bias.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: pre_bias must be on device");
    TT_FATAL(
        post_bias.storage_type() == StorageType::DEVICE, "fused_hyperconnection_pre_post: post_bias must be on device");
    TT_FATAL(
        hidden_streams.storage_type() == StorageType::DEVICE,
        "fused_hyperconnection_pre_post: hidden_streams must be on device");

    TT_FATAL(fused_w.layout() == Layout::TILE, "fused_hyperconnection_pre_post: fused_w must be TILE layout");
    TT_FATAL(pre_bias.layout() == Layout::TILE, "fused_hyperconnection_pre_post: pre_bias must be TILE layout");
    TT_FATAL(post_bias.layout() == Layout::TILE, "fused_hyperconnection_pre_post: post_bias must be TILE layout");
    TT_FATAL(
        hidden_streams.layout() == Layout::TILE, "fused_hyperconnection_pre_post: hidden_streams must be TILE layout");

    TT_FATAL(fused_w.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: fused_w must be BFLOAT16");
    TT_FATAL(pre_bias.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: pre_bias must be BFLOAT16");
    TT_FATAL(post_bias.dtype() == DataType::BFLOAT16, "fused_hyperconnection_pre_post: post_bias must be BFLOAT16");
    TT_FATAL(
        hidden_streams.dtype() == DataType::BFLOAT16,
        "fused_hyperconnection_pre_post: hidden_streams must be BFLOAT16");

    TT_FATAL(
        pre_bias.logical_shape() == post_bias.logical_shape(),
        "fused_hyperconnection_pre_post: pre_bias and post_bias must have the same shape");

    const uint32_t hc = attributes.num_streams;
    const auto& fused_shape = fused_w.logical_shape();
    const auto& bias_shape = pre_bias.logical_shape();
    const auto& hidden_shape = hidden_streams.logical_shape();

    TT_FATAL(
        fused_shape.rank() == 4,
        "fused_hyperconnection_pre_post: fused_w must be rank-4 [1,1,T,(2+H)*H], got rank {}",
        fused_shape.rank());
    TT_FATAL(
        fused_shape[0] == 1 && fused_shape[1] == 1, "fused_hyperconnection_pre_post: fused_w must be [1,1,T,(2+H)*H]");
    TT_FATAL(
        fused_shape[-1] == (2 + hc) * hc,
        "fused_hyperconnection_pre_post: fused_w last dim must be (2+H)*H = {}, got {}",
        (2 + hc) * hc,
        fused_shape[-1]);
    const uint32_t num_tokens = static_cast<uint32_t>(fused_shape[2]);
    TT_FATAL(num_tokens >= 1, "fused_hyperconnection_pre_post: fused_w must carry at least one token row");

    TT_FATAL(
        bias_shape.rank() == 4,
        "fused_hyperconnection_pre_post: pre_bias must be rank-4 [1,1,1,H], got rank {}",
        bias_shape.rank());
    TT_FATAL(
        bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1,
        "fused_hyperconnection_pre_post: pre_bias must be [1,1,1,H]");
    TT_FATAL(
        bias_shape[-1] == hc,
        "fused_hyperconnection_pre_post: pre_bias last dim must be H={}, got {}",
        hc,
        bias_shape[-1]);

    TT_FATAL(
        hidden_shape.rank() == 4,
        "fused_hyperconnection_pre_post: hidden_streams must be rank-4 [B,S,H,D], got rank {}",
        hidden_shape.rank());
    // H <= 32 means every token's [H,D] slab occupies exactly one tile row, so the B and S dims
    // only decide how many such slabs there are: token t is tile row t of the flattened grid.
    TT_FATAL(
        static_cast<uint32_t>(hidden_shape[0]) * static_cast<uint32_t>(hidden_shape[1]) == num_tokens,
        "fused_hyperconnection_pre_post: hidden_streams B*S ({}*{}) must equal fused_w's token count T={}",
        hidden_shape[0],
        hidden_shape[1],
        num_tokens);
    TT_FATAL(
        hidden_shape[2] == hc,
        "fused_hyperconnection_pre_post: hidden_streams stream dim ({}) must match H ({})",
        hidden_shape[2],
        hc);
    TT_FATAL(hc <= 32, "fused_hyperconnection_pre_post: only H<=32 (single comb tile) is supported, got H={}", hc);
}

}  // namespace

void FusedPrePostDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

void FusedPrePostDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

FusedPrePostDeviceOperation::spec_return_value_t FusedPrePostDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& fused_w = tensor_args.fused_w;
    const auto& hidden_streams = tensor_args.hidden_streams;
    const uint32_t hc = operation_attributes.num_streams;
    const auto output_layout = tt::tt_metal::TensorLayout(
        fused_w.dtype(), tt::tt_metal::PageConfig(fused_w.layout()), operation_attributes.output_mem_config);

    // Per token t of the T == B*S tokens:
    //   post      = 2 * sigmoid(post_w * post_scale + post_bias), emitted as a column [H,1].
    //   collapsed = pre[1,H] @ hidden[H,D] -> [1,D].
    //   comb_w_mat = comb_w slice of fused_w, laid out as the [H,H] grid (one tile).
    // Tokens stay on dim 1 so the caller can split it back into [B,S,...] with a view reshape.
    const auto& hidden_shape = hidden_streams.logical_shape();
    const uint32_t num_tokens = static_cast<uint32_t>(fused_w.logical_shape()[2]);
    const ttnn::Shape post_shape({1, num_tokens, hc, 1});
    const ttnn::Shape collapsed_shape({1, num_tokens, 1, hidden_shape[-1]});
    const ttnn::Shape comb_shape({1, num_tokens, hc, hc});
    return {
        tt::tt_metal::TensorSpec(post_shape, output_layout),
        tt::tt_metal::TensorSpec(collapsed_shape, output_layout),
        tt::tt_metal::TensorSpec(comb_shape, output_layout)};
}

FusedPrePostDeviceOperation::tensor_return_value_t FusedPrePostDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.fused_w.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

std::array<Tensor, 3> fused_hyperconnection_pre_post(
    const Tensor& fused_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& hidden_streams,
    uint32_t num_streams,
    float pre_scale,
    float post_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::hyperconnection::FusedPrePostDeviceOperation;
    const MemoryConfig output_mem_config = memory_config.value_or(fused_w.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = num_streams,
        .pre_scale = pre_scale,
        .post_scale = post_scale,
        .eps = eps,
        .output_mem_config = output_mem_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .fused_w = fused_w,
        .pre_bias = pre_bias,
        .post_bias = post_bias,
        .hidden_streams = hidden_streams,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

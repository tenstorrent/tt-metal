// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deltanet {

void DeltaNetRecurrenceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& /*args*/) {
    // Validate tensor shapes and dtypes
    TT_FATAL(attrs.head_k_dim % 32 == 0, "head_k_dim must be tile-aligned (multiple of 32)");
    TT_FATAL(attrs.head_v_dim % 32 == 0, "head_v_dim must be tile-aligned (multiple of 32)");
}

DeltaNetRecurrenceOperation::spec_return_value_t DeltaNetRecurrenceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // Output: (1, 1, B_pad, value_dim=6144)
    uint32_t value_dim = attrs.num_heads * attrs.head_v_dim;
    auto input_shape = args.conv_out.padded_shape();
    auto output_shape = ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], value_dim});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(args.conv_out.dtype(), tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
}

DeltaNetRecurrenceOperation::tensor_return_value_t DeltaNetRecurrenceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.conv_out.device());
}

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {
ttnn::operations::experimental::deltanet::DeltaNetRecurrenceOperation::tensor_return_value_t deltanet_recurrence(
    const Tensor& conv_out,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& z_proj,
    const Tensor& dt_bias,
    const Tensor& A_exp,
    const Tensor& norm_weight,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t head_k_dim,
    uint32_t head_v_dim,
    uint32_t num_k_heads,
    uint32_t gqa_ratio,
    float scale,
    float norm_eps) {
    return ttnn::device_operation::launch<ttnn::operations::experimental::deltanet::DeltaNetRecurrenceOperation>(
        ttnn::operations::experimental::deltanet::DeltaNetRecurrenceOperation::operation_attributes_t{
            .num_heads = num_heads,
            .head_k_dim = head_k_dim,
            .head_v_dim = head_v_dim,
            .num_k_heads = num_k_heads,
            .gqa_ratio = gqa_ratio,
            .scale = scale,
            .norm_eps = norm_eps,
        },
        ttnn::operations::experimental::deltanet::DeltaNetRecurrenceOperation::tensor_args_t{
            .conv_out = conv_out,
            .b_proj = b_proj,
            .a_proj = a_proj,
            .z_proj = z_proj,
            .dt_bias = dt_bias,
            .A_exp = A_exp,
            .norm_weight = norm_weight,
            .state = state,
        });
}
}  // namespace ttnn::prim

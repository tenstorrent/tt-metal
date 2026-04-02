// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "sparse_moe_device_operation.hpp"

namespace ttnn::operations::experimental::sparse_moe {

SparseMoeExpertOperation::program_factory_t SparseMoeExpertOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*args*/) {
    return SingleCore{};
}

void SparseMoeExpertOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& /*args*/) {
    TT_FATAL(attrs.hidden_dim % 32 == 0, "hidden_dim must be tile-aligned");
    TT_FATAL(attrs.expert_inter_dim % 32 == 0, "expert_inter_dim must be tile-aligned");
}

void SparseMoeExpertOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& /*args*/) {
    TT_FATAL(attrs.hidden_dim % 32 == 0, "hidden_dim must be tile-aligned");
    TT_FATAL(attrs.expert_inter_dim % 32 == 0, "expert_inter_dim must be tile-aligned");
    TT_FATAL(attrs.batch_size % 32 == 0, "batch_size must be tile-aligned");
}

SparseMoeExpertOperation::spec_return_value_t SparseMoeExpertOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // Output: (1, 1, batch, num_experts * 2 * expert_inter_dim) = same as ttnn.linear output
    uint32_t output_width = attrs.num_experts * 2 * attrs.expert_inter_dim;
    auto input_shape = args.input.padded_shape();
    auto output_shape = ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], output_width});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(args.input.dtype(), tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
}

SparseMoeExpertOperation::tensor_return_value_t SparseMoeExpertOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

}  // namespace ttnn::operations::experimental::sparse_moe

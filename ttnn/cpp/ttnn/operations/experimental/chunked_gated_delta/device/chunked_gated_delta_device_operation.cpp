// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunked_gated_delta_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

ChunkedGatedDeltaDeviceOperation::operation_attributes_t ChunkedGatedDeltaDeviceOperation::compute_operation_attributes(
    const tensor_args_t& tensor_args) {
    const auto& g_exp_shape = tensor_args.g_exp.logical_shape();
    const auto& factor_shape = tensor_args.factor.logical_shape();
    const auto& bktv_shape = tensor_args.bktv.logical_shape();

    return {
        .total_num_heads = g_exp_shape[0],
        .seq_len = g_exp_shape[1],
        .dim_k = factor_shape[2],
        .dim_v = bktv_shape[3],
    };
}

ChunkedGatedDeltaDeviceOperation::program_factory_t ChunkedGatedDeltaDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // Each head is independent, so any time we have more than one head we get
    // strictly more parallelism by spreading them across cores. Fall back to
    // the single-core factory only for the trivial 1-head case.
    if (operation_attributes.total_num_heads > 1) {
        return MultiCore{};
    }
    return SingleCore{};
}

void ChunkedGatedDeltaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& g_exp = tensor_args.g_exp;
    const auto& factor = tensor_args.factor;
    const auto& bktv = tensor_args.bktv;
    const auto& state = tensor_args.state;

    const uint32_t total_num_heads = operation_attributes.total_num_heads;
    const uint32_t seq_len = operation_attributes.seq_len;
    const uint32_t dim_k = operation_attributes.dim_k;
    const uint32_t dim_v = operation_attributes.dim_v;

    const auto& g_exp_shape = g_exp.logical_shape();
    const auto& factor_shape = factor.logical_shape();
    const auto& bktv_shape = bktv.logical_shape();
    const auto& state_shape = state.logical_shape();

    TT_FATAL(g_exp.layout() == Layout::TILE, "g_exp must be on tile layout");
    TT_FATAL(factor.layout() == Layout::TILE, "factor must be on tile layout");
    TT_FATAL(bktv.layout() == Layout::TILE, "bktv must be on tile layout");
    TT_FATAL(state.layout() == Layout::TILE, "state must be on tile layout");
    TT_FATAL(g_exp.dtype() == factor.dtype(), "g_exp and factor must have the same dtype");
    TT_FATAL(g_exp.dtype() == bktv.dtype(), "g_exp and bktv must have the same dtype");
    TT_FATAL(g_exp.dtype() == state.dtype(), "g_exp and state must have the same dtype");

    TT_FATAL(
        g_exp_shape[0] == total_num_heads,
        "g_exp dim 0 must be total_num_heads ({}), got {}",
        total_num_heads,
        g_exp_shape[0]);
    TT_FATAL(g_exp_shape[1] == seq_len, "g_exp dim 1 must be seq_len ({}), got {}", seq_len, g_exp_shape[1]);
    TT_FATAL(g_exp_shape[2] == 1, "g_exp dim 2 must be 1, got {}, for shape {}", g_exp_shape[2], g_exp_shape);
    TT_FATAL(g_exp_shape[3] == 1, "g_exp dim 3 must be 1, got {}, for shape {}", g_exp_shape[3], g_exp_shape);

    TT_FATAL(
        factor_shape[0] == total_num_heads,
        "factor dim 0 must be total_num_heads ({}), got {}",
        total_num_heads,
        factor_shape[0]);
    TT_FATAL(factor_shape[1] == seq_len, "factor dim 1 must be seq_len ({}), got {}", seq_len, factor_shape[1]);
    TT_FATAL(factor_shape[2] == dim_k, "factor dim 2 must be dim_k ({}), got {}", dim_k, factor_shape[2]);
    TT_FATAL(factor_shape[3] == dim_k, "factor dim 3 must be dim_k ({}), got {}", dim_k, factor_shape[3]);

    TT_FATAL(
        bktv_shape[0] == total_num_heads,
        "bktv dim 0 must be total_num_heads ({}), got {}",
        total_num_heads,
        bktv_shape[0]);
    TT_FATAL(bktv_shape[1] == seq_len, "bktv dim 1 must be seq_len ({}), got {}", seq_len, bktv_shape[1]);
    TT_FATAL(bktv_shape[2] == dim_k, "bktv dim 2 must be dim_k ({}), got {}", dim_k, bktv_shape[2]);
    TT_FATAL(bktv_shape[3] == dim_v, "bktv dim 3 must be dim_v ({}), got {}", dim_v, bktv_shape[3]);

    TT_FATAL(
        state_shape[0] == total_num_heads,
        "state dim 0 must be total_num_heads ({}), got {}",
        total_num_heads,
        state_shape[0]);
    TT_FATAL(state_shape[1] == 1, "state dim 1 must be 1, got {}, for shape {}", state_shape[1], state_shape);
    TT_FATAL(state_shape[2] == dim_k, "state dim 2 must be dim_k ({}), got {}", dim_k, state_shape[2]);
    TT_FATAL(state_shape[3] == dim_v, "state dim 3 must be dim_v ({}), got {}", dim_v, state_shape[3]);

    TT_FATAL(g_exp.device() != nullptr, "g_exp must be on device");
    TT_FATAL(g_exp.device() == factor.device(), "g_exp and factor must be on the same device");
    TT_FATAL(g_exp.device() == bktv.device(), "g_exp and bktv must be on the same device");
    TT_FATAL(g_exp.device() == state.device(), "g_exp and state must be on the same device");
}

ChunkedGatedDeltaDeviceOperation::spec_return_value_t ChunkedGatedDeltaDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        ttnn::Shape{
            operation_attributes.total_num_heads,
            operation_attributes.seq_len,
            operation_attributes.dim_k,
            operation_attributes.dim_v},
        tt::tt_metal::TensorLayout(
            tensor_args.g_exp.dtype(), tt::tt_metal::PageConfig(tensor_args.g_exp.layout()), MemoryConfig{}));
}

ChunkedGatedDeltaDeviceOperation::tensor_return_value_t ChunkedGatedDeltaDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.g_exp.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::ChunkedGatedDeltaDeviceOperation::tensor_return_value_t chunked_gated_delta(
    const Tensor& g_exp, const Tensor& factor, const Tensor& bktv, const Tensor& state) {
    using OperationType = ttnn::experimental::prim::ChunkedGatedDeltaDeviceOperation;
    auto tensor_args = OperationType::tensor_args_t{g_exp, factor, bktv, state};
    auto operation_attributes = OperationType::compute_operation_attributes(tensor_args);

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

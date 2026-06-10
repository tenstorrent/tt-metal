// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::indexer {

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

void IndexerScoreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    const auto& w_shape = tensor_args.weights.logical_shape();

    TT_FATAL(q_shape.rank() == 4 && k_shape.rank() == 4 && w_shape.rank() == 4, "q, k, weights must be rank 4");
    TT_FATAL(k_shape[1] == 1, "k must be single-head [B, 1, T, D], got {} heads", k_shape[1]);
    TT_FATAL(q_shape[3] == k_shape[3], "q head dim {} != k head dim {}", q_shape[3], k_shape[3]);
    TT_FATAL(
        w_shape[1] == q_shape[1] && w_shape[2] == q_shape[2] && w_shape[3] == 1,
        "weights must be [B, Hi, Sq, 1] matching q [B, Hi, Sq, D]");
    TT_FATAL(q_shape[0] == k_shape[0] && q_shape[0] == w_shape[0], "batch mismatch");
}

IndexerScoreDeviceOperation::spec_return_value_t IndexerScoreDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    // score [B, 1, Sq, T], row-major bf16 (consumed by the row-major topk)
    ttnn::Shape out_shape({q_shape[0], 1, q_shape[2], k_shape[2]});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), tensor_args.q.memory_config()));
}

IndexerScoreDeviceOperation::tensor_return_value_t IndexerScoreDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

std::tuple<IndexerScoreDeviceOperation::operation_attributes_t, IndexerScoreDeviceOperation::tensor_args_t>
IndexerScoreDeviceOperation::invoke(
    const Tensor& q, const Tensor& k, const Tensor& weights, bool is_causal, uint32_t chunk_start_idx) {
    return {
        operation_attributes_t{.is_causal = is_causal, .chunk_start_idx = chunk_start_idx},
        tensor_args_t{.q = q, .k = k, .weights = weights}};
}

}  // namespace ttnn::operations::experimental::deepseek::indexer

namespace ttnn::prim {

ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    bool is_causal,
    uint32_t chunk_start_idx) {
    using OperationType = ttnn::operations::experimental::deepseek::indexer::IndexerScoreDeviceOperation;
    auto operation_attributes =
        OperationType::operation_attributes_t{.is_causal = is_causal, .chunk_start_idx = chunk_start_idx};
    auto tensor_args = OperationType::tensor_args_t{.q = q, .k = k, .weights = weights};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

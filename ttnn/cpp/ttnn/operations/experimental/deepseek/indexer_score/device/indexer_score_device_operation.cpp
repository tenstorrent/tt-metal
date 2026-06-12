// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::deepseek::indexer {

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

void IndexerScoreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    const auto& w_shape = tensor_args.weights.logical_shape();

    // Shapes: q [B, Hi, Sq, D], k [B, 1, T, D] (single shared head), weights [B, Hi, Sq, 1].
    TT_FATAL(q_shape.rank() == 4 && k_shape.rank() == 4 && w_shape.rank() == 4, "q, k, weights must be rank 4");
    TT_FATAL(k_shape[1] == 1, "k must be single-head [B, 1, T, D], got {} heads", k_shape[1]);
    TT_FATAL(q_shape[3] == k_shape[3], "q head dim {} != k head dim {}", q_shape[3], k_shape[3]);
    TT_FATAL(
        w_shape[1] == q_shape[1] && w_shape[2] == q_shape[2] && w_shape[3] == 1,
        "weights must be [B, Hi, Sq, 1] matching q [B, Hi, Sq, D]");
    TT_FATAL(q_shape[0] == k_shape[0] && q_shape[0] == w_shape[0], "batch mismatch");
    TT_FATAL(q_shape[0] == 1, "batch 1 only, got {}", q_shape[0]);
    TT_FATAL(attrs.is_causal, "non-causal not implemented");

    // q/weights are packed and read as bf16 tiles; k is matmul srcA only (never packed),
    // so it may also be bfp8_b -- halves k DRAM/L1 BW. Mismatched dtype/layout corrupts or hangs.
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const auto& w = tensor_args.weights;
    TT_FATAL(
        q.dtype() == DataType::BFLOAT16 && w.dtype() == DataType::BFLOAT16,
        "q, weights must be bfloat16 (got q={}, weights={})",
        q.dtype(),
        w.dtype());
    TT_FATAL(
        k.dtype() == DataType::BFLOAT16 || k.dtype() == DataType::BFLOAT8_B,
        "k must be bfloat16 or bfloat8_b (got {})",
        k.dtype());
    TT_FATAL(
        q.layout() == Layout::TILE && k.layout() == Layout::TILE && w.layout() == Layout::TILE,
        "q, k, weights must be TILE layout");

    // Tile alignment and the causal chunk window.
    const uint32_t Hi = q_shape[1];
    const uint32_t Sq = q_shape[2];
    const uint32_t D = q_shape[3];
    const uint32_t T = k_shape[2];
    TT_FATAL(
        Sq % TILE_HEIGHT == 0 && T % TILE_WIDTH == 0 && D % TILE_WIDTH == 0,
        "Sq {}, T {}, D {} must be tile-aligned",
        Sq,
        T,
        D);
    TT_FATAL(attrs.chunk_start_idx % TILE_WIDTH == 0, "chunk_start_idx {} must be tile-aligned", attrs.chunk_start_idx);
    TT_FATAL(
        attrs.chunk_start_idx + Sq <= T,
        "chunk window [{}, {}+{}) exceeds T={}",
        attrs.chunk_start_idx,
        attrs.chunk_start_idx,
        Sq,
        T);

    // Work-unit knobs (elements, tile-aligned); see IndexerScoreProgramConfig.
    const auto& cfg = attrs.program_config;
    TT_FATAL(
        cfg.q_chunk_size % TILE_HEIGHT == 0 && cfg.k_chunk_size % TILE_WIDTH == 0,
        "q_chunk_size {} / k_chunk_size {} must be tile-aligned",
        cfg.q_chunk_size,
        cfg.k_chunk_size);
    const uint32_t QC = cfg.q_chunk_size / TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);
    TT_FATAL(QC > 0 && (Sq / TILE_HEIGHT) % QC == 0, "q_chunk_size {} must divide Sq {}", cfg.q_chunk_size, Sq);
    TT_FATAL(KC > 0 && KC <= T / TILE_WIDTH, "k_chunk_size {} out of range (T={})", cfg.k_chunk_size, T);
    TT_FATAL(HB > 0 && Hi % HB == 0, "head_group_size {} must divide Hi {}", HB, Hi);
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
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

std::tuple<IndexerScoreDeviceOperation::operation_attributes_t, IndexerScoreDeviceOperation::tensor_args_t>
IndexerScoreDeviceOperation::invoke(
    const Tensor& q,
    const Tensor& k,
    const Tensor& weights,
    bool is_causal,
    uint32_t chunk_start_idx,
    const IndexerScoreProgramConfig& program_config) {
    return {
        operation_attributes_t{
            .is_causal = is_causal, .chunk_start_idx = chunk_start_idx, .program_config = program_config},
        tensor_args_t{.q = q, .k = k, .weights = weights}};
}

}  // namespace ttnn::operations::experimental::deepseek::indexer

namespace ttnn::prim {

ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    bool is_causal,
    uint32_t chunk_start_idx,
    const ttnn::operations::experimental::deepseek::indexer::IndexerScoreProgramConfig& program_config) {
    using OperationType = ttnn::operations::experimental::deepseek::indexer::IndexerScoreDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .is_causal = is_causal, .chunk_start_idx = chunk_start_idx, .program_config = program_config};
    auto tensor_args = OperationType::tensor_args_t{.q = q, .k = k, .weights = weights};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

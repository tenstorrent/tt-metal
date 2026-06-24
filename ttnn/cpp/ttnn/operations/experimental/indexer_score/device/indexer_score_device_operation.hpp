// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "indexer_score_device_operation_types.hpp"
#include "indexer_score_program_factory.hpp"

namespace ttnn::operations::experimental::indexer_score {

struct IndexerScoreDeviceOperation {
    using operation_attributes_t = indexer_score::operation_attributes_t;
    using tensor_args_t = indexer_score::tensor_args_t;
    using spec_return_value_t = indexer_score::spec_return_value_t;
    using tensor_return_value_t = indexer_score::tensor_return_value_t;
    using program_factory_t = std::variant<program::IndexerScoreProgramFactory>;

    // No custom compute_program_hash: operation_attributes_t (chunk_start_idx + every config field)
    // is a reflectable aggregate, so the default reflection hash already keys distinct programs on all
    // fields. Do not add a hand-rolled hash that drops fields.
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Matmul-FLOP performance model (mirrors SDPAOperation::create_op_performance_model): reports ideal
    // matmul cycles so tracy's ideal/actual utilization equals the math-util test's mm_flops/(cores x
    // cycles x peak). FLOPs count causal-valid tiles only; cores/fidelity match the factory.
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& q,
        const Tensor& k,
        const Tensor& weights,
        uint32_t chunk_start_idx,
        bool apply_relu,
        uint32_t num_groups,
        uint32_t block_size,
        const IndexerScoreProgramConfig& program_config,
        const DeviceComputeKernelConfig& compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

// Two public frontends over one shared device operation (IndexerScoreDeviceOperation): the lightning
// indexer comes in two flavours that differ only in fixed knobs, so each model family gets its own
// callable rather than one op with mode flags. Both share the program factory and the 3 kernels (the
// flavour is just compile-time args), and both produce a row-major bf16 score the downstream topk reads.
// Causality (shared): key t visible to query s iff t <= chunk_start_idx + s; future -> -inf.

// DeepSeek-V3.2 DSA / GLM-5 lightning indexer (ttnn.experimental.indexer_score_dsa):
//   score[b, 0, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
// q [B, Hi, Sq, D], k [B, 1, T, D], weights [B, Hi, Sq, 1] (learned per-head gates, scale pre-folded)
//   -> score [B, 1, Sq, T]. All Hi heads relu'd and summed into one shared selection row.
ttnn::Tensor indexer_score_dsa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    uint32_t chunk_start_idx = 0,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

// MiniMax-M3 MSA lightning indexer (ttnn.experimental.indexer_score_msa):
//   score[b, g, s, t] = sum_{h in group g} (q[b,h,s,:] . k[b,t,:]) * scale
// Raw dot (no relu), NO learned gates -- only a 1/sqrt(d) `scale` (run as a synthesized constant gate),
// so there is no weights tensor. q [B, Hi, Sq, D], k [B, 1, T, D] -> score [B, num_groups, Sq, T_out].
// num_groups: keep the GQA groups separate (G output planes, no cross-group sum) for per-group selection;
//   G==1 is the TP=4 group-aligned deployment (Hi=1/device). G>1 needs all heads resident, k_chunk>=64.
// block_size: 0 = no pooling -> [B,G,Sq,T]. >0 = block-max-pool over block_size keys -> [B,G,Sq,T/block_size]
//   (M3 block selection; the downstream topk picks per-group top-k blocks).
ttnn::Tensor indexer_score_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    uint32_t chunk_start_idx = 0,
    float scale = 1.0f,
    uint32_t num_groups = 1,
    uint32_t block_size = 0,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental

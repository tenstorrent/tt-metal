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

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Custom hash: runtime values (cache_batch_idx / kv_len / chunk_start_idx) are excluded so they reuse
    // one program; cluster_axis IS hashed.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks the runtime values that can change on a hit (slot < B, kv_len bounds, chunk_start window).
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Matmul-FLOP performance model: reports ideal matmul cycles so tracy's util matches the math-util
    // test's mm_flops/(cores x cycles x peak). FLOPs count causal-valid tiles only.
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
        bool synthesize_gate,
        float gate_scale,
        const IndexerScoreProgramConfig& program_config,
        const DeviceComputeKernelConfig& compute_kernel_config,
        std::optional<uint32_t> cache_batch_idx,
        std::optional<uint32_t> kv_len,
        std::optional<uint32_t> cluster_axis);
};

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

// Two public frontends over one shared device op: the lightning indexer's two flavours differ only in
// fixed knobs, so each gets its own callable. Both share the program factory + 3 kernels (flavour = compile-
// time args) and produce a row-major bf16 score. Causality: key t visible to query s iff t <= chunk_start + s.

// DeepSeek-V3.2 DSA / GLM-5 (ttnn.experimental.indexer_score_dsa):
//   score[b, 0, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
// q [B,Hi,Sq,D], k [B,1,T,D], weights [B,Hi,Sq,1] -> score [B,1,Sq,T] (all heads relu'd + summed).
// cache_batch_idx/kv_len/chunk_start_idx are re-applied each dispatch and hash-excluded (no recompile); see
// the nanobind docs for their full semantics. OMIT chunk_start_idx on a mesh (deduced as T - sp_ring*Sq).
ttnn::Tensor indexer_score_dsa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_len = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

// MiniMax-M3 MSA (ttnn.experimental.indexer_score_msa):
//   score[b, g, s, t] = sum_{h in group g} (q[b,h,s,:] . k[b,t,:]) * scale
// Raw dot, no learned gates -- only a 1/sqrt(d) `scale` (synthesized as a constant gate, so no weights
// tensor). q [B,Hi,Sq,D], k [B,1,T,D] -> score [B,num_groups,Sq,T_out].
// num_groups: G output planes (no cross-group sum); G==1 = TP=4 group-aligned (Hi=1/device), G>1 needs all
//   heads resident + k_chunk>=64. block_size: 0 = full [B,G,Sq,T]; >0 = block-max-pool -> [B,G,Sq,T/bs].
// chunk_start_idx / cluster_axis: same SP-ring semantics as indexer_score_dsa.
// num_groups is required (no default): per-GQA-group selection is MSA's purpose, so the caller must state
// the group count explicitly. It is placed before the defaulted optionals so the signature stays well-formed.
ttnn::Tensor indexer_score_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    uint32_t num_groups,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    float scale = 1.0f,
    uint32_t block_size = 0,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental

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

    // Custom hash: the runtime values (cache_batch_idx / kv_len / chunk_start_idx) are excluded so they reuse
    // one compiled program; cluster_axis IS hashed. See the definition for the full rationale.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Re-checks the runtime values that can change on a hit: persistent-cache values (slot < B, kv_len bounds)
    // and the hash-excluded chunk_start window/alignment (mirrors ring_joint_sdpa's runtime revalidation).
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
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
        const IndexerScoreProgramConfig& program_config,
        const DeviceComputeKernelConfig& compute_kernel_config,
        std::optional<uint32_t> cache_batch_idx,
        std::optional<uint32_t> kv_len,
        std::optional<uint32_t> cluster_axis);
};

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

// DeepSeek-V3.2 DSA lightning-indexer scorer (the public callable, ttnn.experimental.indexer_score):
//   score[b, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
// q [B, Hi, Sq, D], k [B, 1, T, D], weights [B, Hi, Sq, 1] -> score [B, 1, Sq, T] (row-major bf16).
// Causality from chunk_start_idx: key t visible to query s iff t <= chunk_start_idx + s.
//
// cache_batch_idx: when set, k is a shared [B, 1, T, D] cache and this selects the batch slot to score
// against (k may then also be ND-sharded across DRAM banks).
// kv_len: when set, only the first kv_len key positions of k are valid this dispatch (rest masked out);
// must be tile-aligned, in (0, T], with chunk_start_idx + Sq <= kv_len. Output is still [B, 1, Sq, T] with
// only columns [0, kv_len) written.
// chunk_start_idx is the ABSOLUTE chunk_start of rank 0 (lowest cluster_axis coord); rank r uses
// chunk_start_idx + r*Sq, so one dispatch gives each SP rank its own chunk_start. OMIT it on multichip: the
// base is deduced as T - sp_ring*Sq (sp_ring = mesh extent along cluster_axis, whole mesh if unset -- a 2D
// SP x TP mesh uses the SP-axis length, not total devices). On single chip (one device = rank 0) set it to
// history + rank*Sq to simulate any rank. cluster_axis selects the SP ring axis.
// All of cache_batch_idx / kv_len / chunk_start are re-applied each dispatch and excluded from the program
// hash, so switching slot, growing kv_len, or changing chunk_start reuses ONE program -- no recompile.
ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_len = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental

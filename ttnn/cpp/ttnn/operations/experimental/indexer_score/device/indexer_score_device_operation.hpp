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

    // Custom hash: chunk_start_idx / chunk_start_stride are RUNTIME values (per-device, derived from the
    // mesh coordinate), so they are EXCLUDED from the key -- distinct chunk_start values reuse one
    // compiled program. cluster_axis IS hashed (it fixes each device's stored linearized index). Do not
    // re-add chunk_start_idx/stride here or every value will recompile.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // chunk_start is hash-excluded, so its window/alignment check is re-run on cache hits too (mirrors
    // ring_joint_sdpa's runtime-patched-scalar revalidation).
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
        std::optional<uint32_t> cluster_axis);
};

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

// DeepSeek-V3.2 DSA lightning-indexer scorer (the public callable, ttnn.experimental.indexer_score):
//   score[b, s, t] = sum_h relu(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
// q [B, Hi, Sq, D], k [B, 1, T, D], weights [B, Hi, Sq, 1] -> score [B, 1, Sq, T] (row-major bf16).
// Causality from chunk_start_idx: key t visible to query s iff t <= chunk_start_idx + s.
// chunk_start_idx is the device-0 base; device r uses base + r*Sq (r = linearized index along
// cluster_axis, Sq = q seq len), so one dispatch gives each SP rank its own chunk_start.
// OMIT chunk_start_idx on multichip: the base is then deduced from the shapes as T - num_devices*Sq
// (the chunked-prefill invariant K = history + the SP-gathered chunk). Pass it on single chip, where
// the device count does not equal the SP ring size (e.g. simulating one rank).
ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"

#include "sdpa_ksplit_merge_device_operation_types.hpp"

namespace ttnn::prim {

// Exact merge of the K-split partitions of ring joint SDPA:
//   M = max_p m_p, a_p = exp(scale (m_p - M)), out = sum_p a_p O_p / sum_p a_p l_p
// One fused program over the whole worker grid (fp32 DEST); work item = (batch, head, 32-row tile).
struct SDPAKSplitMergeDeviceOperation {
    using operation_attributes_t = SDPAKSplitMergeParams;
    using tensor_args_t = SDPAKSplitMergeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

// Largest k_split the merge supports (DEST: S coefficient tiles + 2 scratch tiles in 8 fp32 tiles).
inline constexpr uint32_t SDPA_KSPLIT_MERGE_MAX_SPLIT = 6;

Tensor sdpa_k_split_merge(
    const Tensor& partial_output,
    const Tensor& partial_stats,
    uint32_t k_split,
    float scale,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim

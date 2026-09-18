// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include "unified_routed_expert_ffn_program_factory.hpp"
#include "unified_routed_expert_ffn_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

struct UnifiedRoutedExpertFfnDeviceOperation {
    using operation_attributes_t = UnifiedRoutedExpertFfnParams;
    using tensor_args_t = UnifiedRoutedExpertFfnInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<UnifiedRoutedExpertFfnProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn

namespace ttnn::prim {

// Single-program multi-expert MoE FFN. The device program's kernels loop over all
// `experts_per_chip` local experts, running the full gate/up/down FFN for each.
ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& x,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& output,
    uint32_t m_tiles,
    uint32_t experts_per_chip,
    bool x_is_row_major,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation activation =
        ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation::Silu,
    const std::vector<ttnn::Tensor>& gate_biases = {},
    const std::vector<ttnn::Tensor>& up_biases = {},
    const std::vector<ttnn::Tensor>& down_biases = {});

}  // namespace ttnn::prim

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

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    uint32_t chunk_M_tiles,
    uint32_t m_tiles,
    bool read_x_at_offset,
    bool x_is_row_major,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& optional_output,
    const std::optional<ttnn::Tensor>& expert_region_offsets = std::nullopt,
    ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation activation =
        ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation::Silu,
    const std::optional<ttnn::Tensor>& gate_bias = std::nullopt,
    const std::optional<ttnn::Tensor>& up_bias = std::nullopt,
    const std::optional<ttnn::Tensor>& down_bias = std::nullopt);

}  // namespace ttnn::prim

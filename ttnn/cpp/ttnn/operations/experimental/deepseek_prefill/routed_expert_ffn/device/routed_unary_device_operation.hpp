// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "routed_unary_types.hpp"
#include "routed_unary_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

struct RoutedUnaryDeviceOperation {
    using operation_attributes_t = RoutedUnaryParams;
    using tensor_args_t = RoutedUnaryInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<RoutedUnaryProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Exclude local_expert_idx / curr_expert_iter / expert_iter_length from the
    // hash so the same cached program is reused across chunk iterations; only
    // guard runtime args change between dispatches.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device

namespace ttnn::prim {

ttnn::Tensor routed_unary(
    const ttnn::Tensor& input,
    const ttnn::operations::unary::EltwiseUnaryWithParam& op,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt);

}  // namespace ttnn::prim

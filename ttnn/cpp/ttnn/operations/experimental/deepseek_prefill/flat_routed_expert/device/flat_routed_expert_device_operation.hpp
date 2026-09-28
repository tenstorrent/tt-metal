// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "flat_routed_expert_program_factory.hpp"
#include "flat_routed_expert_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

struct FlatRoutedExpertDeviceOperation {
    using operation_attributes_t = FlatRoutedExpertParams;
    using tensor_args_t = FlatRoutedExpertInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<FlatRoutedExpertProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert

namespace ttnn::prim {

ttnn::Tensor flat_routed_expert(
    const ttnn::operations::experimental::deepseek_prefill::flat_routed_expert::FlatRoutedExpertConfig& config,
    const ttnn::operations::experimental::deepseek_prefill::flat_routed_expert::FlatRoutedExpertInputs& inputs);

}  // namespace ttnn::prim

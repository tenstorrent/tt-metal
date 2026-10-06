// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "practice_routed_expert_types.hpp"
#include "practice_routed_expert_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

struct PracticeRoutedExpertDeviceOperation {
    using operation_attributes_t = PracticeRoutedExpertParams;
    using tensor_args_t = PracticeRoutedExpertInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<PracticeRoutedExpertProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert

namespace ttnn::prim {

ttnn::Tensor practice_routed_expert(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim

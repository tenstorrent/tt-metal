// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "bf16_to_fp8_types.hpp"
#include "bf16_to_fp8_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8 {

struct Bf16ToFp8DeviceOperation {
    using operation_attributes_t = Bf16ToFp8Params;
    using tensor_args_t = Bf16ToFp8Inputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<Bf16ToFp8ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8

namespace ttnn::prim {

ttnn::Tensor prefill_bf16_to_fp8(const ttnn::Tensor& input_tensor);

}  // namespace ttnn::prim

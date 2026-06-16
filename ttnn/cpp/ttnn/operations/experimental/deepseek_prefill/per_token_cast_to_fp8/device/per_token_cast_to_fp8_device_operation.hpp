// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "per_token_cast_to_fp8_device_operation_types.hpp"
#include "per_token_cast_to_fp8_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

struct PerTokenCastToFp8DeviceOperation {
    using operation_attributes_t = PerTokenCastToFp8Params;
    using tensor_args_t = PerTokenCastToFp8Inputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = std::variant<PerTokenCastToFp8ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8

namespace ttnn::prim {
std::tuple<ttnn::Tensor, ttnn::Tensor> per_token_cast_to_fp8(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_memory_config);
}  // namespace ttnn::prim

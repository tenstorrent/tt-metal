// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "gelu_backward_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "gelu_backward_device_operation_types.hpp"

namespace ttnn::operations::experimental::gelu_backward {

struct GeluBackwardDeviceOperation {
    using operation_attributes_t = gelu_backward::operation_attributes_t;
    using tensor_args_t = gelu_backward::tensor_args_t;
    using spec_return_value_t = gelu_backward::spec_return_value_t;
    using tensor_return_value_t = gelu_backward::tensor_return_value_t;
    using program_factory_t = std::variant<program::GeluBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& grad_output,
        const Tensor& input,
        const std::string& approximate,
        DataType output_dtype,
        const MemoryConfig& output_memory_config,
        const std::optional<Tensor>& preallocated_input_grad);
};

}  // namespace ttnn::operations::experimental::gelu_backward

namespace ttnn::prim {
constexpr auto gelu_bw = ttnn::register_operation<
    "ttnn::prim::gelu_bw",
    ttnn::operations::experimental::gelu_backward::GeluBackwardDeviceOperation>();
}  // namespace ttnn::prim

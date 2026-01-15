// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    using operation_attributes_t = GeluBackwardParams;
    using tensor_args_t = GeluBackwardInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::GeluBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::gelu_backward

namespace ttnn::prim {

ttnn::operations::experimental::gelu_backward::GeluBackwardDeviceOperation::tensor_return_value_t gelu_bw(
    const Tensor& grad_output,
    const Tensor& input,
    const std::string& approximate,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output);

}  // namespace ttnn::prim

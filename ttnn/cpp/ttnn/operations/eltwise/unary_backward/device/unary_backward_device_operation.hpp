// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "unary_backward_device_operation_types.hpp"
#include "unary_backward_op_types.hpp"
#include "unary_backward_program_factory.hpp"

namespace ttnn::operations::unary_backward {

// Shared device operation for unary backward ops: one grad_output and one input of the same
// shape in, one input gradient out. Which gradient is computed comes from
// operation_attributes_t::op_type, so validation, output specs, the program hash and the
// program factory are written once for every op in UnaryBackwardOpType.
struct UnaryBackwardDeviceOperation {
    using operation_attributes_t = UnaryBackwardParams;
    using tensor_args_t = UnaryBackwardInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<UnaryBackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Entry point for the composite-op layer in unary_backward.cpp.
Tensor launch_unary_backward(
    UnaryBackwardOpType op_type,
    const Tensor& grad_output,
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_input_grad = std::nullopt);

}  // namespace ttnn::operations::unary_backward

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "binary_backward_device_operation_types.hpp"
#include "binary_backward_op_types.hpp"
#include "binary_backward_program_factory.hpp"

namespace ttnn::operations::binary_backward {

struct BinaryBackwardDeviceOperation {
    using operation_attributes_t = BinaryBackwardParams;
    using tensor_args_t = BinaryBackwardInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<BinaryBackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> launch_binary_backward(
    BinaryBackwardOpType op_type,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& other,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    std::array<bool, 2> are_required_outputs,
    const std::optional<Tensor>& preallocated_input_grad,
    const std::optional<Tensor>& preallocated_other_grad);

}  // namespace ttnn::operations::binary_backward

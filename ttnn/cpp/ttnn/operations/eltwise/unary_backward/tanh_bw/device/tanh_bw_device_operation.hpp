// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "tanh_bw_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "tanh_bw_device_operation_types.hpp"

namespace ttnn::operations::unary_backward::tanh_bw {

struct TanhBwDeviceOperation {
    using operation_attributes_t = TanhBwParams;
    using tensor_args_t = TanhBwInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<TanhBwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor launch_tanh_bw(
    const Tensor& grad_output,
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output);

}  // namespace ttnn::operations::unary_backward::tanh_bw

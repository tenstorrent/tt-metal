// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "clip_grad_norm_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "clip_grad_norm_device_operation_types.hpp"

namespace ttnn::operations::experimental::clip_grad_norm {

struct ClipGradNormDeviceOperation {
    using operation_attributes_t = clip_grad_norm::operation_attributes_t;
    using tensor_args_t = clip_grad_norm::tensor_args_t;
    using spec_return_value_t = clip_grad_norm::spec_return_value_t;
    using tensor_return_value_t = clip_grad_norm::tensor_return_value_t;
    using program_factory_t = std::variant<program::ClipGradNormProgramFactory>;
    using shared_variables_t = program::ClipGradNormProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        float max_norm,
        float p,
        float eps,
        DataType output_dtype,
        const MemoryConfig& output_memory_config = MemoryConfig(),
        const std::optional<Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttnn::operations::experimental::clip_grad_norm

namespace ttnn::prim {
constexpr auto clip_grad_norm = ttnn::register_operation<
    "ttnn::prim::clip_grad_norm",
    ttnn::operations::experimental::clip_grad_norm::ClipGradNormDeviceOperation>();
}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {

struct MorehDotBackwardOperation {
    struct operation_attributes_t {
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& output_grad;
        const Tensor& input;
        const Tensor& other;

        // (o2buzzle): May I present: thanhnguyen's mistake that costed me 3 hours.
        const std::vector<std::optional<Tensor>> output_tensors;
    };

    using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    struct ProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::moreh::moreh_dot_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_dot_backward::MorehDotBackwardOperation::tensor_return_value_t moreh_dot_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    std::optional<const Tensor> input_grad,
    std::optional<const Tensor> other_grad,
    const std::optional<MemoryConfig>& memory_config);
}  // namespace ttnn::prim

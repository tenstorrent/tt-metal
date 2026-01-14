// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

namespace ttnn::operations::reduction::sampling {

struct SamplingDeviceOperation {
    using operation_attributes_t = sampling::operation_attributes_t;
    using tensor_args_t = sampling::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::SamplingProgramFactory>;
    using shared_variables_t = program::SamplingProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::reduction::sampling

namespace ttnn::prim {
ttnn::Tensor sampling(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& preallocated_output_tensor = std::nullopt);
}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "rotate_half_device_operation_types.hpp"
#include "rotate_half_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::rotate_half {

struct RotateHalfDeviceOperation {
    using operation_attributes_t = RotateHalfParams;
    using tensor_args_t = RotateHalfInputs;
    using spec_return_value_t = rotate_half::spec_return_value_t;
    using tensor_return_value_t = rotate_half::tensor_return_value_t;
    using program_factory_t = std::variant<program::RotateHalfProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::transformer::rotate_half

namespace ttnn::prim {
ttnn::operations::experimental::transformer::rotate_half::tensor_return_value_t rotate_half(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim

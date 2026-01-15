// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation_types.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"

namespace ttnn::operations::data_movement::nonzero {

struct NonZeroIndicesDeviceOperation {
    using operation_attributes_t = NonzeroParams;
    using tensor_args_t = NonzeroInputs;
    using spec_return_value_t = nonzero::spec_return_value_t;
    using tensor_return_value_t = nonzero::tensor_return_value_t;
    using program_factory_t = std::variant<program::NonZeroIndicesProgramFactory>;
    using shared_variables_t = program::NonZeroIndicesProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::data_movement::nonzero

namespace ttnn::prim {

ttnn::operations::data_movement::nonzero::NonZeroIndicesDeviceOperation::tensor_return_value_t nonzero(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& memory_config);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "step_context_device_operation_types.hpp"
#include "step_context_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::step_context {

struct StepContextDeviceOperation {
    using operation_attributes_t = StepContextParams;
    using tensor_args_t = StepContextInputs;
    using spec_return_value_t = StepContextSpecReturn;
    using tensor_return_value_t = StepContextTensorReturn;
    using program_factory_t = std::variant<StepContextProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::step_context

namespace ttnn::prim {

std::vector<Tensor> decode_step_context(
    const Tensor& packet,
    const std::vector<Tensor>& rope_tables,
    const std::vector<std::vector<uint32_t>>& int_shapes,
    const std::vector<DataType>& int_dtypes,
    const std::vector<std::vector<int32_t>>& int_terms,
    const std::vector<uint32_t>& rope_table_ids,
    const std::vector<uint32_t>& rope_fields,
    const std::vector<int32_t>& rope_offsets);

}  // namespace ttnn::prim

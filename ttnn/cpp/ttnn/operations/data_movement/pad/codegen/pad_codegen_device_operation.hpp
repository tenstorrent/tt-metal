// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operation.hpp"

#include "pad_codegen_device_operation_types.hpp"
#include "pad_codegen_program_factory.hpp"

namespace ttnn::prim {

struct PadCodegenDeviceOperation {
    using operation_attributes_t = PadCodegenParams;
    using tensor_args_t = PadCodegenInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PadCodegenProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor pad_codegen(
    const Tensor& input,
    const PadCodegenDeviceOperation::operation_attributes_t& operation_attributes,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/types.hpp"
#include "untilize_codegen_program_factory.hpp"

namespace ttnn::prim {

struct UntilizeCodegenTensorArgs {
    Tensor input;
};

struct UntilizeCodegenOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
};

using UntilizeCodegenTensorReturnValue = Tensor;
using UntilizeCodegenSpecReturnValue = ttnn::TensorSpec;

struct UntilizeCodegenDeviceOperation {
    using operation_attributes_t = UntilizeCodegenOperationAttributes;
    using tensor_args_t = UntilizeCodegenTensorArgs;
    using spec_return_value_t = UntilizeCodegenSpecReturnValue;
    using tensor_return_value_t = UntilizeCodegenTensorReturnValue;
    using program_factory_t = std::variant<UntilizeCodegenProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor untilize_codegen(const Tensor& input, tt::tt_metal::MemoryConfig output_mem_config);

}  // namespace ttnn::prim

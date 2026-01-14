// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "concatenate_heads_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "concatenate_heads_device_operation_types.hpp"

namespace ttnn::operations::experimental::transformer {

struct ConcatenateHeadsDeviceOperation {
    using operation_attributes_t = transformer::operation_attributes_t;
    using tensor_args_t = transformer::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::ConcatenateHeadsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {

ttnn::operations::experimental::transformer::ConcatenateHeadsDeviceOperation::tensor_return_value_t concatenate_heads(
    const ttnn::Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& preallocated_output);

}  // namespace ttnn::prim

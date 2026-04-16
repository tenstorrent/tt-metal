// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::index_fill {
struct IndexFillOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const std::variant<float, int> value;
        const MemoryConfig memory_config;
    };
    struct tensor_args_t {
        const Tensor& input;
        const Tensor& index;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::index_fill
namespace ttnn::prim {
ttnn::Tensor index_fill(
    const Tensor& input,
    uint32_t dim,
    const Tensor& index,
    std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}

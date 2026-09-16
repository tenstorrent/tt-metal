// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::index_fill_new {

// Descriptor-based port of index_fill::IndexFillOperation (#42392).
//
// Single descriptor, no program_factory_t, no override_runtime_arguments: every per-core runtime
// arg other than the three buffer addresses (input, index, output) is derived from state the
// default program hash already covers (dim / value / memory_config attributes and the input and
// index TensorSpecs), so a cache hit guarantees the work split and per-core args are unchanged.
// The addresses are declared as Buffer* bindings via emplace_runtime_args and patched in place by
// the framework on cache hits (recipe §1.2 branch (b)).
struct IndexFillNewOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const std::variant<float, int> value;
        const MemoryConfig memory_config;
    };
    struct tensor_args_t {
        const Tensor& input;
        const Tensor& index;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
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
}  // namespace ttnn::operations::index_fill_new

namespace ttnn::prim {
ttnn::Tensor index_fill_new(
    const Tensor& input,
    uint32_t dim,
    const Tensor& index,
    std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}

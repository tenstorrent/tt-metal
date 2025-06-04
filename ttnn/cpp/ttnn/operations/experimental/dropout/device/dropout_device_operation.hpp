// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "dropout_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "dropout_device_operation_types.hpp"

namespace ttnn::operations::experimental::dropout {

struct DropoutDeviceOperation {
    using operation_attributes_t = dropout::operation_attributes_t;
    using tensor_args_t = dropout::tensor_args_t;
    using spec_return_value_t = dropout::spec_return_value_t;
    using tensor_return_value_t = dropout::tensor_return_value_t;
    using program_factory_t = std::variant<program::DropoutProgramFactory, program::DropoutMeshWorkloadFactory>;
    using shared_variables_t = program::DropoutProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        float prob,
        float scale,
        uint32_t seed,
        bool use_per_device_seed,
        DataType output_dtype,
        const MemoryConfig& output_memory_config = MemoryConfig(),
        const std::optional<Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttnn::operations::experimental::dropout

namespace ttnn::prim {
constexpr auto dropout =
    ttnn::register_operation<"ttnn::prim::dropout", ttnn::operations::experimental::dropout::DropoutDeviceOperation>();
}  // namespace ttnn::prim

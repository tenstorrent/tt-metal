// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::generic {

struct GenericOpDeviceOperation {
    using operation_attributes_t = tt::tt_metal::ProgramDescriptor;

    using tensor_return_value_t = Tensor;

    using spec_return_value_t = TensorSpec;

    // NOTE: output tensor is the last element in the vector io_tensors
    struct tensor_args_t {
        const std::vector<Tensor>& io_tensors;
        const Tensor& output_tensor;
    };

    struct GenericProgram {
        // to refactor this when we implement caching
        struct shared_variables_t {};
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<GenericProgram>;

    // Mandatory methods
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const std::vector<Tensor>& io_tensors, const operation_attributes_t& operation_attributes);

    // static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};  // struct GenericOpDeviceOperation

}  // namespace ttnn::operations::generic

namespace ttnn::prim {
constexpr auto generic_op =
    ttnn::register_operation<"ttnn::prim::generic_op", ttnn::operations::generic::GenericOpDeviceOperation>();
}  // namespace ttnn::prim

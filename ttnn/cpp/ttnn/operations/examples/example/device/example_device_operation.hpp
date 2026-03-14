// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::examples {

struct ExampleDeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        bool attribute;
        int some_other_attribute;
    };

    // Define the tensor arguments. This is it to store all tensors passed in and/or out of the operation
    // Tensor arguments don't need to be just input tensors, they can be output tensors, input/output tensors, optional
    // tensors, etc.
    struct tensor_args_t {
        // This example will use a tensor that can only be used as an input
        const Tensor& input_tensor;

        // However, the following examples show what else can be done with tensor_args_t

        // An example of the tensor that can be used for input/output or just for pre-allocated output
        // Tensor& io_tensor;

        // An example of an optional tensor
        // std::optional<Tensor> optional_output_tensor;

        // An example of a vector of tensors
        // std::vector<Tensor> vector_of_tensors;

        // An example of a tuple of tensors
        // std::tuple<Tensor, ...> tuple_of_tensors;

        // An example of a vector of optional tensors
        // std::vector<std::optional<Tensor>> vector_of_optional_tensors;

        // An example of a tuple of tensors
        // std::tuple<std::vector<std::optional<Tensor>>, std::optional<Tensor>> some_crazy_tuple_of_tensors;
    };

    // Define the return types for the spec(s) of the operation
    // Can be a single ttnn::TensorSpec, std::optional<ttnn::TensorSpec>, std::vector<ttnn::TensorSpec>,
    // std::tuple<ttnn::TensorSpec> etc.
    using spec_return_value_t = ttnn::TensorSpec;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = Tensor;

    // Note spec_return_value_t and tensor_return_value_t should follow the same pattern
    // i.e. if spec_return_value_t is a std::vector<std::optional<ttnn::TensorSpec>> then tensor_return_value_t should
    // be std::vector<std::optional<Tensor>>

    // =========================================================================
    // ProgramDescriptor-based factories (RECOMMENDED for all new operations)
    //
    // These use `create_descriptor()` which returns a declarative ProgramDescriptor.
    // The framework handles program construction, caching, buffer address
    // patching, and seed patching automatically. No shared_variables_t or
    // cached_program_t needed.
    //
    // Seed handling is automatic: if operation_attributes_t has a uint32_t seed
    // field, the framework excludes it from hashing and patches compute kernel
    // runtime_args[0] with (seed + core_index) on every cache hit.
    // =========================================================================

    struct SingleCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    // Required only when program_factory_t has more than one variant.
    // For single-variant program_factory_t, the framework auto-selects it.
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Also called on cache hit by default.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Optional: override to use lighter validation on cache hit.
    // If not provided, the framework calls validate_on_program_cache_miss.
    // static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output specs based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::examples

namespace ttnn::prim {
ttnn::operations::examples::ExampleDeviceOperation::tensor_return_value_t example(const Tensor& input_tensor);
}  // namespace ttnn::prim
